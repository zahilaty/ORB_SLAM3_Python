"""Frames, covisibility/spanning tree, landmarks and Atlas bookkeeping.

Data and mutations follow Frame/KeyFrame/MapPoint/Map/Atlas source; thread
locks and serialization wrappers are deliberately absent. Sensor preparation
belongs in input.py; tracking/mapping algorithms operate on these records.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from dataclasses import dataclass, field
from collections import Counter
import copy
import numpy as np
from .features import FeatureSet, hamming
from .geometry import inverse_pose, transform_points


@dataclass(eq=False)
class Frame:
    id: int
    timestamp: float
    features: FeatureSet
    camera: object
    width: int
    height: int
    scales: np.ndarray
    inverse_variances: np.ndarray
    baseline_fx: float = 0.
    depth_threshold: float = 0.
    pose: np.ndarray | None = None
    map_points: list = field(default_factory=list)
    right_u: np.ndarray | None = None
    depth: np.ndarray | None = None
    outliers: np.ndarray | None = None
    bow: dict = field(default_factory=dict)
    feature_nodes: dict = field(default_factory=dict)
    raw_features: FeatureSet | None = None
    bounds: tuple | None = None
    n_left: int = -1
    right_camera: object | None = None
    T_right_left: np.ndarray | None = None
    stereo_matches: np.ndarray | None = None
    stereo_points: np.ndarray | None = None
    reference_id: int | None = None
    last_keyframe_id: int | None = None
    body_state: object | None = None
    imu_calibration: object | None = None
    preintegrated: object | None = None
    frame_preintegrated: object | None = None
    state_prior: object | None = None

    def __post_init__(self):
        n = len(self.features)
        self.scales = np.asarray(self.scales)
        self.inverse_variances = np.asarray(self.inverse_variances)
        if not self.map_points:
            self.map_points = [None] * n
        if self.right_u is None:
            self.right_u = np.full(n, -1.)
        if self.depth is None:
            self.depth = np.full(n, -1.)
        if self.outliers is None:
            self.outliers = np.zeros(n, dtype=bool)
        if self.bounds is None:
            self.bounds = (0., float(self.width), 0., float(self.height))
        if self.raw_features is None:
            self.raw_features = self.features
        if self.right_camera is not None and (self.T_right_left is None or self.n_left < 0):
            raise ValueError("Unrectified stereo needs T_right_left and left feature count")

    @property
    def center(self):
        return -self.pose[:3, :3].T @ self.pose[:3, 3]

    @property
    def baseline(self):
        return self.baseline_fx / self.camera.fx

    def camera_for(self, index):
        return self.right_camera if self.n_left >= 0 and index >= self.n_left else self.camera

    def pose_for(self, index):
        if self.n_left >= 0 and index >= self.n_left:
            return self.T_right_left @ self.pose
        return self.pose

    def center_for(self, index):
        pose = self.pose_for(index)
        return -pose[:3, :3].T @ pose[:3, 3]

    def in_image(self, x, y):
        minx, maxx, miny, maxy = self.bounds
        return minx <= x < maxx and miny <= y < maxy

    def get_features_in_area(self, x, y, radius, min_level=-1, max_level=-1, right=False):
        # Equivalent to source grid query's final predicates; O(N) vectorized
        # query avoids a second spatial representation without changing matches.
        points = self.features.points
        mask = (np.abs(points[:, 0] - x) < radius) & (np.abs(points[:, 1] - y) < radius)
        if self.n_left >= 0:
            mask &= (np.arange(len(points)) >= self.n_left) if right else (np.arange(len(points)) < self.n_left)
        if min_level > 0:
            mask &= self.features.octaves >= min_level
        if max_level >= 0:
            mask &= self.features.octaves <= max_level
        minx, maxx, miny, maxy = self.bounds
        gx = (points[:, 0]-minx) * 64/(maxx-minx)
        gy = (points[:, 1]-miny) * 48/(maxy-miny)
        cellx = (np.sign(gx)*np.floor(np.abs(gx)+.5)).astype(int)
        celly = (np.sign(gy)*np.floor(np.abs(gy)+.5)).astype(int)
        mask &= (cellx >= 0) & (cellx < 64) & (celly >= 0) & (celly < 48)
        indices = np.flatnonzero(mask)
        indices = indices[np.lexsort((indices, celly[indices], cellx[indices]))]
        return indices - self.n_left if right and self.n_left >= 0 else indices

    def compute_bow(self, vocabulary):
        if not self.bow:
            self.bow, self.feature_nodes = vocabulary.transform(self.features.descriptors, levelsup=4)

    def unproject_depth(self, index):
        if self.depth[index] <= 0:
            return None
        if self.stereo_points is not None and index < len(self.stereo_points) and np.isfinite(self.stereo_points[index]).all():
            point = self.stereo_points[index]
        else:
            point = self.camera.unproject(self.features.points[index]) * self.depth[index]
        return transform_points(inverse_pose(self.pose), point)


@dataclass(eq=False)
class KeyFrame(Frame):
    map: object | None = field(default=None, repr=False)
    source_frame_id: int = -1
    inertial: bool = False
    covisibility: dict = field(default_factory=dict)
    ordered_covisibility: list = field(default_factory=list)
    parent_id: int | None = None
    children: set = field(default_factory=set)
    loop_edges: set = field(default_factory=set)
    merge_edges: set = field(default_factory=set)
    first_connection: bool = True
    bad: bool = False
    not_erase: bool = False
    to_be_erased: bool = False
    pose_to_parent: np.ndarray | None = None
    previous_id: int | None = None
    next_id: int | None = None

    def best_covisible(self, count):
        return [self.map.keyframes[i] for i in self.ordered_covisibility[:count]
                if i in self.map.keyframes and not self.map.keyframes[i].bad]

    def add_connection(self, other, weight):
        self.covisibility[other.id] = weight
        self.ordered_covisibility = sorted(self.covisibility, key=lambda i: (self.covisibility[i], i), reverse=True)

    def update_connections(self):
        counts = Counter()
        for point in self.map_points:
            if point is None or point.bad:
                continue
            for other_id in point.observations:
                other = self.map.keyframes.get(other_id)
                if other_id != self.id and other is not None and not other.bad:
                    counts[other_id] += 1
        if not counts:
            return
        strong = {i: n for i, n in counts.items() if n >= 15}
        if not strong:
            i = max(counts, key=lambda j: (counts[j], -j))
            strong[i] = counts[i]
        for i, weight in strong.items():
            self.map.keyframes[i].add_connection(self, weight)
        self.covisibility = dict(counts)
        self.ordered_covisibility = sorted(strong, key=lambda i: (strong[i], i), reverse=True)
        if self.first_connection and self.id != self.map.initial_keyframe_id:
            self.change_parent(self.map.keyframes[self.ordered_covisibility[0]])
            self.first_connection = False

    def change_parent(self, parent):
        self.parent_id = parent.id if parent is not None else None
        if parent is not None:
            parent.children.add(self.id)

    def tracked_points(self, minimum_observations=0):
        return sum(p is not None and not p.bad and p.observation_count >= minimum_observations for p in self.map_points)

    def median_depth(self, quantile=2):
        depths = sorted(transform_points(self.pose, p.position)[2] for p in self.map_points if p is not None and not p.bad)
        return depths[(len(depths)-1)//quantile] if depths else -1.

    def set_bad(self, database=None):
        if self.id == self.map.initial_keyframe_id:
            return
        if self.not_erase or self.loop_edges:
            self.to_be_erased = True
            return
        for i in self.covisibility:
            other = self.map.keyframes.get(i)
            if other is not None:
                other.covisibility.pop(self.id, None)
                other.ordered_covisibility = [j for j in other.ordered_covisibility if j != self.id]
        for point in self.map_points.copy():
            if point is not None:
                point.erase_observation(self)
        self.covisibility.clear()
        self.ordered_covisibility.clear()
        candidates = {self.parent_id} if self.parent_id is not None else set()
        while self.children:
            best = None
            best_weight = -1
            for child_id in sorted(self.children):
                child = self.map.keyframes[child_id]
                if child.bad:
                    continue
                for candidate in child.ordered_covisibility:
                    if candidate in candidates and child.covisibility[candidate] > best_weight:
                        best = (child, self.map.keyframes[candidate])
                        best_weight = child.covisibility[candidate]
            if best is None:
                break
            child, parent = best
            child.change_parent(parent)
            candidates.add(child.id)
            self.children.remove(child.id)
        parent = self.map.keyframes.get(self.parent_id)
        for child_id in self.children:
            self.map.keyframes[child_id].change_parent(parent)
        self.children.clear()
        if parent is not None:
            parent.children.discard(self.id)
            self.pose_to_parent = self.pose @ inverse_pose(parent.pose)
        self.bad = True
        self.map.keyframes.pop(self.id, None)
        if database is not None:
            database.erase(self.id)


@dataclass(eq=False)
class MapPoint:
    id: int
    position: np.ndarray
    reference_id: int
    map: object = field(repr=False)
    first_keyframe_id: int = -1
    observations: dict = field(default_factory=dict)
    observation_count: int = 0
    descriptor: np.ndarray | None = None
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    min_distance: float = 0.
    max_distance: float = np.inf
    visible: int = 1
    found: int = 1
    last_seen_frame: int = -1
    track_depth: float = np.inf
    bad: bool = False
    replaced: object | None = field(default=None, repr=False)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float).copy()
        if self.first_keyframe_id < 0:
            self.first_keyframe_id = self.reference_id

    def add_observation(self, keyframe, index):
        left, right = self.observations.get(keyframe.id, (-1, -1))
        is_right = keyframe.n_left >= 0 and index >= keyframe.n_left
        if index in (left, right):
            return
        if is_right:
            right = index
        else:
            left = index
        self.observations[keyframe.id] = left, right
        self.observation_count += 2 if keyframe.right_camera is None and keyframe.right_u[index] >= 0 else 1

    def erase_observation(self, keyframe):
        indices = self.observations.pop(keyframe.id, None)
        if indices is None:
            return
        for index in indices:
            if index >= 0:
                self.observation_count -= 2 if keyframe.right_camera is None and keyframe.right_u[index] >= 0 else 1
        if self.reference_id == keyframe.id:
            self.reference_id = min(self.observations, default=-1)
        if self.observation_count <= 2:
            self.set_bad()

    def set_bad(self):
        self.bad = True
        for keyframe_id, indices in self.observations.items():
            keyframe = self.map.keyframes.get(keyframe_id)
            if keyframe is not None:
                for index in indices:
                    if index >= 0 and keyframe.map_points[index] is self:
                        keyframe.map_points[index] = None
        self.observations.clear()
        self.map.points.pop(self.id, None)

    def replace(self, other):
        if other.id == self.id:
            return
        for keyframe_id, indices in self.observations.copy().items():
            keyframe = self.map.keyframes.get(keyframe_id)
            if keyframe is None:
                continue
            existing = keyframe_id in other.observations
            for index in indices:
                if index < 0:
                    continue
                if not existing:
                    keyframe.map_points[index] = other
                    other.add_observation(keyframe, index)
                else:
                    keyframe.map_points[index] = None
        self.bad, self.replaced = True, other
        self.observations.clear()
        other.found += self.found
        other.visible += self.visible
        other.compute_descriptor()
        self.map.points.pop(self.id, None)

    @property
    def found_ratio(self):
        return self.found / self.visible

    def compute_descriptor(self):
        descriptors = []
        for keyframe_id, indices in sorted(self.observations.items()):
            keyframe = self.map.keyframes.get(keyframe_id)
            if keyframe is not None and not keyframe.bad:
                descriptors.extend(keyframe.features.descriptors[i] for i in indices if i >= 0)
        if descriptors:
            descriptors = np.asarray(descriptors)
            distances = hamming(descriptors[:, None], descriptors[None])
            medians = np.sort(distances, axis=1)[:, (len(descriptors)-1)//2]
            self.descriptor = descriptors[np.argmin(medians)].copy()

    def update_normal_and_depth(self):
        if self.bad or not self.observations:
            return
        directions = []
        for keyframe_id, indices in self.observations.items():
            keyframe = self.map.keyframes[keyframe_id]
            for index in indices:
                if index >= 0:
                    direction = self.position - keyframe.center_for(index)
                    directions.append(direction / np.linalg.norm(direction))
        self.normal = np.mean(directions, axis=0)
        reference = self.map.keyframes[self.reference_id]
        index = next(i for i in self.observations[self.reference_id] if i >= 0)
        distance = np.linalg.norm(self.position - reference.center)
        self.max_distance = distance * reference.scales[reference.features.octaves[index]]
        self.min_distance = self.max_distance / reference.scales[-1]

    def predict_scale(self, distance, scales):
        if distance <= 0:
            return len(scales)-1
        level = np.ceil(np.log(self.max_distance / distance) / np.log(scales[1] / scales[0])) if len(scales) > 1 else 0
        return int(np.clip(level, 0, len(scales)-1))


@dataclass(eq=False)
class Map:
    id: int
    initial_keyframe_id: int
    keyframes: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)
    origins: list = field(default_factory=list)
    inertial: bool = False
    imu_initialized: bool = False
    inertial_ba1: bool = False
    inertial_ba2: bool = False
    bad: bool = False
    change_index: int = 0
    big_change_index: int = 0

    def apply_scaled_rotation(self, pose, scale, scale_velocity=False):
        """Map::ApplyScaledRotation: transform world, retain camera projection."""
        for keyframe in self.keyframes.values():
            world_pose = inverse_pose(keyframe.pose)
            world_pose[:3, :3] = pose[:3, :3] @ world_pose[:3, :3]
            world_pose[:3, 3] = scale * pose[:3, :3] @ world_pose[:3, 3] + pose[:3, 3]
            keyframe.pose = inverse_pose(world_pose)
            if hasattr(keyframe, "velocity"):
                keyframe.velocity = pose[:3, :3] @ keyframe.velocity * (scale if scale_velocity else 1.)
        for point in self.points.values():
            point.position = scale * pose[:3, :3] @ point.position + pose[:3, 3]
            point.update_normal_and_depth()
        self.change_index += 1


class Atlas:
    def __init__(self, inertial=False):
        self.maps = {}
        self.bad_maps = {}
        self.all_keyframes = {}
        self.next_map_id = self.next_keyframe_id = self.next_point_id = 0
        self.inertial = inertial
        self.big_change_index = 0
        self.current = self.create_map()

    def create_map(self):
        current = Map(self.next_map_id, self.next_keyframe_id, inertial=self.inertial)
        self.next_map_id += 1
        self.maps[current.id] = current
        self.current = current
        return current

    def change_map(self, map_id):
        if self.maps[map_id].bad:
            raise ValueError("Cannot activate bad map")
        self.current = self.maps[map_id]

    def discard_current(self):
        old = self.current
        old.bad = True
        self.bad_maps[old.id] = self.maps.pop(old.id)
        return self.create_map()

    def add_keyframe(self, frame):
        # Arrays extracted for a frame are immutable after insertion; pose/match
        # containers are copied because mapping updates keyframes independently.
        fields = Frame.__dataclass_fields__
        values = {name: getattr(frame, name) for name in fields}
        values.update(id=self.next_keyframe_id, pose=frame.pose.copy(),
                      map_points=frame.map_points.copy(), outliers=frame.outliers.copy(),
                      bow=frame.bow.copy(), feature_nodes=frame.feature_nodes.copy())
        values["body_state"] = copy.deepcopy(frame.body_state)
        keyframe = KeyFrame(**values, source_frame_id=frame.id, map=self.current,
                            inertial=self.current.imu_initialized)
        self.next_keyframe_id += 1
        self.current.keyframes[keyframe.id] = keyframe
        self.all_keyframes[keyframe.id] = keyframe
        if len(self.current.keyframes) == 1:
            self.current.origins.append(keyframe.id)
        return keyframe

    def add_point(self, position, reference):
        point = MapPoint(self.next_point_id, position, reference.id, self.current)
        self.next_point_id += 1
        self.current.points[point.id] = point
        return point
