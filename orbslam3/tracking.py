"""Sequential tracking decisions from src/Tracking.cc.

The tracker queues keyframes. The caller runs local mapping then loop closing
at the end of each image, so changes become visible at the next image boundary.
Core methods consume prepared frames, never files, video, UI, or configuration.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from collections import Counter, deque
from dataclasses import dataclass
from enum import IntEnum
import copy
import cv2
import numpy as np
from .camera import KannalaBrandt8
from .geometry import inverse_pose, make_pose
from .initialization import TwoViewReconstruction
from .map import MapPoint
from .matching import (search_for_initialization, search_by_bow_frame,
                       search_by_projection_last_frame, search_by_projection_local,
                       search_by_projection_relocalization, _geometry_candidate)
from .optimization import optimize_pose, bundle_adjustment, ReprojectionObservation
from .imu import Bias, Preintegrated, frame_intervals, predict_body_state
from .inertial_optimization import BodyState, optimize_inertial_tracking


class TrackingState(IntEnum):
    NO_IMAGES_YET = 0
    NOT_INITIALIZED = 1
    OK = 2
    RECENTLY_LOST = 3
    LOST = 4


@dataclass
class TrackingResult:
    timestamp: float
    state: TrackingState
    pose: np.ndarray | None
    map_id: int
    inliers: int
    keyframe_id: int | None
    diagnostics: dict


@dataclass
class TrajectoryEntry:
    timestamp: float
    relative_pose: np.ndarray
    reference: object
    state: TrackingState

    def pose(self, all_keyframes):
        relative, reference = self.relative_pose.copy(), self.reference
        while reference.bad and reference.parent_id is not None:
            relative = relative @ reference.pose_to_parent
            reference = all_keyframes[reference.parent_id]
        return relative @ reference.pose


def frame_observations(frame, pose_id=0):
    indices, observations, points = [], [], {}
    for index, point in enumerate(frame.map_points):
        if point is None or point.bad:
            continue
        pixel = frame.features.points[index]
        measurement = np.r_[pixel, frame.right_u[index]] if frame.right_camera is None and frame.right_u[index] >= 0 else pixel
        right = frame.n_left >= 0 and index >= frame.n_left
        observations.append(ReprojectionObservation(pose_id, point.id, measurement, frame.camera_for(index),
                            float(frame.inverse_variances[frame.features.octaves[index]]), frame.baseline_fx,
                            frame.T_right_left if right else None))
        points[point.id] = point.position
        indices.append(index)
    return indices, points, observations


def synchronize_body_pose(frame):
    if frame.imu_calibration is None or frame.pose is None:
        return
    world_body = inverse_pose(frame.pose) @ frame.imu_calibration.T_camera_body
    previous = frame.body_state
    frame.body_state = BodyState(world_body[:3, :3], world_body[:3, 3],
                                 previous.velocity if previous is not None else np.zeros(3),
                                 previous.bias if previous is not None else Bias())


def synchronize_camera_pose(frame):
    body = frame.body_state
    frame.pose = frame.imu_calibration.T_camera_body @ inverse_pose(make_pose(body.rotation, body.position))


def need_new_keyframe(sensor, *, frame_id, timestamp, last_id, last_timestamp, last_reloc_id,
                      max_frames, map_keyframes, reference_matches, inliers, tracked_close,
                      untracked_close, state, imu_initialized=False, dual_camera=False,
                      mapping_idle=True, mapping_initializing=False, queued=0, only_tracking=False,
                      mapping_stopped=False):
    """The c1a/c1b/c1c/c2/c3/c4 conditions in NeedNewKeyFrame, visibly intact."""
    if sensor.inertial and not imu_initialized:
        return timestamp-last_timestamp >= .25
    if only_tracking or mapping_stopped:
        return False
    if frame_id < last_reloc_id+max_frames and map_keyframes > max_frames:
        return False
    need_close = tracked_close < 100 and untracked_close > 70
    ratio = .4 if map_keyframes < 2 else .75
    if sensor.monocular and not sensor.inertial:
        ratio = .9
    if dual_camera:
        ratio = .75
    if sensor.monocular and sensor.inertial:
        ratio = .75 if inliers > 350 else .9
    c1a = frame_id >= last_id+max_frames
    c1b = frame_id >= last_id and mapping_idle  # mMinFrames = 0
    c1c = not sensor.monocular and not sensor.inertial and (inliers < .25*reference_matches or need_close)
    c2 = (inliers < ratio*reference_matches or need_close) and inliers > 15
    c3 = sensor.inertial and timestamp-last_timestamp >= .5
    c4 = sensor.inertial and sensor.monocular and (15 < inliers < 75 or state == TrackingState.RECENTLY_LOST)
    if not (((c1a or c1b or c1c) and c2) or c3 or c4):
        return False
    return mapping_idle or mapping_initializing or (not sensor.monocular and queued < 3)


class Tracker:
    def __init__(self, atlas, sensor, calibration, vocabulary, database, *, fast_imu_init=False,
                 insert_keyframes_when_lost=True):
        self.atlas, self.sensor, self.calibration = atlas, sensor, calibration
        self.vocabulary, self.database = vocabulary, database
        self.max_frames = int(calibration.frames_per_second)
        self.fast_imu_init = fast_imu_init
        self.insert_keyframes_when_lost = insert_keyframes_when_lost
        self.state = TrackingState.NO_IMAGES_YET
        self.last = self.current = self.initial = None
        self.reference = self.last_keyframe = None
        self.last_keyframe_frame_id = self.last_reloc_id = self.initial_extractor_keyframe_id = 0
        self.previous_positions = None
        self.velocity = None
        self.local_keyframes, self.local_points = [], []
        self.pending_keyframes, self.history = [], []
        self.inliers = 0
        self.lost_timestamp = 0.
        self.imu_queue = deque()
        self.accumulated_imu = Preintegrated(Bias(), calibration.imu) if sensor.inertial else None
        self.last_map_change = -1
        self.diagnostics = {}
        self.only_tracking = False
        self.visual_odometry = False
        self.temporary_points = []
        self.far_points, self.far_threshold = False, np.inf

    def set_localization_mode(self, enabled):
        """Freeze map growth while retaining the source's visual VO fallback.

        The original localization-only branch explicitly excludes inertial
        operation. Its visual stereo/RGB-D path can create transient depth
        points, but these never enter the Atlas or the keyframe database.
        """
        if enabled and self.sensor.inertial:
            raise ValueError("ORB-SLAM3 does not support inertial localization-only mode")
        self.only_tracking = bool(enabled)

    @property
    def initializing_features(self):
        if not self.sensor.monocular:
            return False
        if self.state in (TrackingState.NO_IMAGES_YET, TrackingState.NOT_INITIALIZED):
            return True
        # The source compares last frame ID with initialization KEYFRAME ID,
        # not the frame that created that keyframe. Preserve this mixed-ID gate.
        return not self.sensor.inertial and self.current.id-self.initial_extractor_keyframe_id < self.max_frames

    def _new_map(self, discard=False):
        old_id = self.atlas.current.id
        if discard:
            self.database.clear_map(old_id)
            self.atlas.discard_current()
        else:
            self.atlas.create_map()
        self.state = TrackingState.NO_IMAGES_YET
        self.last = self.initial = self.reference = self.last_keyframe = None
        self.velocity = self.previous_positions = None
        self.local_keyframes, self.local_points = [], []
        self.pending_keyframes.clear()
        self.temporary_points.clear()
        self.visual_odometry = False
        self.last_map_change = -1
        if self.sensor.inertial:
            self.accumulated_imu = Preintegrated(Bias(), self.calibration.imu)

    def _preintegrate(self, frame, measurements):
        frame.imu_calibration = self.calibration.imu
        bias = self.last_keyframe.body_state.bias if self.last_keyframe is not None else Bias()
        frame.body_state = BodyState(np.eye(3), np.zeros(3), np.zeros(3), bias)
        for measurement in measurements:
            if self.imu_queue and measurement.timestamp <= self.imu_queue[-1].timestamp:
                # Retained boundary samples may be repeated in a caller's packet.
                if measurement.timestamp == self.imu_queue[-1].timestamp:
                    continue
                raise ValueError("IMU inputs must arrive in increasing timestamp order")
            self.imu_queue.append(measurement)
        if self.last is None:
            return
        selected = []
        while self.imu_queue:
            measurement = self.imu_queue[0]
            if measurement.timestamp < self.last.timestamp-.001:
                self.imu_queue.popleft()
            elif measurement.timestamp < frame.timestamp-.001:
                selected.append(self.imu_queue.popleft())
            else:
                selected.append(measurement)
                break
        if len(selected) < 2:
            self.diagnostics["imu"] = "insufficient boundary measurements"
            return
        preintegrated = Preintegrated(self.last.body_state.bias, self.calibration.imu)
        for acceleration, gyro, dt in frame_intervals(selected, self.last.timestamp, frame.timestamp):
            preintegrated.integrate(acceleration, gyro, dt)
            self.accumulated_imu.integrate(acceleration, gyro, dt)
        frame.frame_preintegrated, frame.preintegrated = preintegrated, self.accumulated_imu

    def track(self, frame, imu=()):
        self.current, self.diagnostics, self.inliers = frame, {}, 0
        frame.last_keyframe_id = self.last_keyframe.id if self.last_keyframe is not None else None
        active = self.atlas.current
        if self.last is not None and frame.timestamp < self.last.timestamp:
            self.imu_queue.clear()
            self._new_map()
            self.diagnostics["reset"] = "backward timestamp"
            return self._result(frame)
        if self.last is not None and frame.timestamp == self.last.timestamp:
            raise ValueError("Image timestamps must be strictly increasing")
        if self.last is not None and self.sensor.inertial and frame.timestamp > self.last.timestamp+1.:
            self._new_map(discard=not (active.imu_initialized and active.inertial_ba2))
            self.diagnostics["reset"] = "inertial timestamp jump"
            return self._result(frame)
        if self.sensor.inertial:
            self._preintegrate(frame, imu)
        map_updated = active.change_index > self.last_map_change
        self.last_map_change = active.change_index
        if self.state == TrackingState.NO_IMAGES_YET:
            self.state = TrackingState.NOT_INITIALIZED
        if self.state == TrackingState.NOT_INITIALIZED:
            self._initialize_monocular(frame) if self.sensor.monocular else self._initialize_depth(frame)
            self.last = frame
            self._record(frame)
            return self._result(frame)
        if self.state == TrackingState.LOST and not self.only_tracking:
            self._new_map(discard=len(active.keyframes) < 10)
            return self._result(frame)
        success = False
        if self.only_tracking:
            success = self._track_localization(frame, map_updated)
        elif self.state == TrackingState.OK:
            for i, point in enumerate(self.last.map_points):
                if point is not None and point.replaced is not None:
                    self.last.map_points[i] = point.replaced
            if (self.velocity is None and not active.imu_initialized) or frame.id < self.last_reloc_id+2:
                success = self._track_reference(frame)
            else:
                success = self._track_motion(frame, map_updated) or self._track_reference(frame)
            if not success:
                if self.sensor.inertial and frame.id <= self.last_reloc_id+self.max_frames:
                    self.state = TrackingState.LOST
                elif len(active.keyframes) > 10:
                    self.state = TrackingState.RECENTLY_LOST
                    self.lost_timestamp = frame.timestamp
                else:
                    self.state = TrackingState.LOST
        elif self.state == TrackingState.RECENTLY_LOST:
            if self.sensor.inertial:
                success = active.imu_initialized and self._predict_imu(frame, map_updated)
                if frame.timestamp-self.lost_timestamp > 5.:
                    success, self.state = False, TrackingState.LOST
            else:
                success = self._relocalize(frame)
                if frame.timestamp-self.lost_timestamp > 3. and not success:
                    self.state = TrackingState.LOST
        if success and (not self.only_tracking or not self.visual_odometry):
            success = self._track_local_map(frame, map_updated)
        if success:
            self.state = TrackingState.OK
        elif self.state == TrackingState.OK:
            self.state, self.lost_timestamp = TrackingState.RECENTLY_LOST, frame.timestamp
            if self.sensor.inertial and not (active.imu_initialized and active.inertial_ba2):
                self._new_map(discard=True)
                return self._result(frame)
        if self.state in (TrackingState.OK, TrackingState.RECENTLY_LOST):
            if frame.pose is not None and self.last.pose is not None:
                self.velocity = frame.pose @ inverse_pose(self.last.pose)
            # Tracking.cc removes transient depth associations before deciding
            # whether a new keyframe may be inserted, then destroys the points.
            for i, point in enumerate(frame.map_points):
                if point is not None and point.observation_count < 1:
                    frame.map_points[i], frame.outliers[i] = None, False
            self.temporary_points.clear()
            if self._needs_keyframe(frame) and (success or self.sensor.inertial and self.insert_keyframes_when_lost):
                self._create_keyframe(frame)
            for i, outlier in enumerate(frame.outliers):
                if outlier:
                    frame.map_points[i] = None
        if self.state == TrackingState.LOST:
            self._new_map(discard=len(active.keyframes) <= 10 or self.sensor.inertial and not active.imu_initialized)
            return self._result(frame)
        self._record(frame)
        self.last = frame
        return self._result(frame)

    def _result(self, frame):
        self.diagnostics["localization_only"] = self.only_tracking
        self.diagnostics["visual_odometry"] = self.visual_odometry
        return TrackingResult(frame.timestamp, self.state, None if frame.pose is None else frame.pose.copy(),
                              self.atlas.current.id, self.inliers, self.last_keyframe.id if self.last_keyframe else None,
                              self.diagnostics.copy())

    def _track_localization(self, frame, map_updated):
        """Tracking.cc's separate localization/VO/relocalization decision tree."""
        if self.state == TrackingState.LOST:
            return self._relocalize(frame)
        if not self.visual_odometry:
            if self.velocity is not None:
                return self._track_motion(frame, map_updated)
            return self._track_reference(frame)
        motion_success = False
        if self.velocity is not None:
            motion_success = self._track_motion(frame, map_updated)
            motion_pose = None if frame.pose is None else frame.pose.copy()
            motion_points = frame.map_points.copy()
            motion_outliers = frame.outliers.copy()
        relocalized = self._relocalize(frame)
        if motion_success and not relocalized:
            frame.pose, frame.map_points, frame.outliers = motion_pose, motion_points, motion_outliers
            if self.visual_odometry:
                for index, point in enumerate(frame.map_points):
                    if point is not None and not frame.outliers[index]:
                        point.found += 1
        elif relocalized:
            self.visual_odometry = False
        return motion_success or relocalized

    def _record(self, frame):
        if self.reference is not None and frame.pose is not None and self.state in (TrackingState.OK, TrackingState.RECENTLY_LOST):
            frame.reference_id = self.reference.id
            self.history.append(TrajectoryEntry(frame.timestamp, frame.pose @ inverse_pose(self.reference.pose), self.reference, self.state))

    def _seed_points(self, frame, keyframe, all_depths=False):
        count = 0
        n = frame.n_left if frame.n_left >= 0 else len(frame.features)
        indices = sorted((frame.depth[i], i) for i in range(n) if frame.depth[i] > 0)
        for depth, i in indices:
            point = frame.map_points[i]
            if point is None or point.observation_count < 1:
                world_point = frame.unproject_depth(i)
                if world_point is None:
                    continue
                point = self.atlas.add_point(world_point, keyframe)
                point.add_observation(keyframe, i)
                keyframe.map_points[i] = frame.map_points[i] = point
                if frame.n_left >= 0 and frame.stereo_matches[i] >= 0:
                    j = frame.stereo_matches[i]
                    keyframe.map_points[j] = frame.map_points[j] = point
                    point.add_observation(keyframe, j)
                point.compute_descriptor()
                point.update_normal_and_depth()
            count += 1
            if not all_depths and depth > frame.depth_threshold and count > 100:
                break

    def _initialize_depth(self, frame):
        if len(frame.features) <= 500:
            return
        if self.sensor.inertial:
            if self.last is None or frame.preintegrated is None or self.last.preintegrated is None:
                return
            if not self.fast_imu_init and np.linalg.norm(frame.frame_preintegrated.average_acceleration-self.last.frame_preintegrated.average_acceleration) < .5:
                return
            self.accumulated_imu = Preintegrated(Bias(), self.calibration.imu)
            frame.preintegrated = self.accumulated_imu
        frame.pose = np.eye(4)
        synchronize_body_pose(frame)
        keyframe = self.atlas.add_keyframe(frame)
        frame.last_keyframe_id = keyframe.last_keyframe_id = keyframe.id
        self._seed_points(frame, keyframe, all_depths=True)
        self.pending_keyframes.append(keyframe)
        self.reference = self.last_keyframe = keyframe
        self.local_keyframes = [keyframe]
        self.local_points = list(self.atlas.current.points.values())
        self.last_keyframe_frame_id = frame.id
        self.state = TrackingState.OK
        self.inliers = keyframe.tracked_points()

    def _initialize_monocular(self, frame):
        if self.initial is None:
            if len(frame.features) > 100:
                self.initial = frame
                self.previous_positions = frame.features.points.copy()
                if self.sensor.inertial:
                    self.accumulated_imu = Preintegrated(Bias(), self.calibration.imu)
                    frame.preintegrated = self.accumulated_imu
            return
        if len(frame.features) <= 100 or self.sensor.inertial and self.last.timestamp-self.initial.timestamp > 1.:
            self.initial = None
            return
        matches = search_for_initialization(self.initial, frame, self.previous_positions)
        if np.count_nonzero(matches >= 0) < 100:
            self.initial = None
            return
        keys1, keys2 = self.initial.features.points, frame.features.points
        if isinstance(frame.camera, KannalaBrandt8):
            camera = frame.camera
            distortion = np.array([camera.k1, camera.k2, camera.k3, camera.k4])
            keys1 = cv2.fisheye.undistortPoints(keys1[:, None].astype(float), camera.K, distortion, P=camera.K)[:, 0]
            keys2 = cv2.fisheye.undistortPoints(keys2[:, None].astype(float), camera.K, distortion, P=camera.K)[:, 0]
        result = TwoViewReconstruction(frame.camera.K).reconstruct(keys1, keys2, matches)
        if not result.success:
            return
        matches[~result.triangulated] = -1
        self.initial.pose, frame.pose = np.eye(4), result.pose
        synchronize_body_pose(self.initial)
        synchronize_body_pose(frame)
        first, current = self.atlas.add_keyframe(self.initial), self.atlas.add_keyframe(frame)
        first.compute_bow(self.vocabulary)
        current.compute_bow(self.vocabulary)
        for i in np.flatnonzero(matches >= 0):
            j = matches[i]
            point = self.atlas.add_point(result.points[i], current)
            first.map_points[i] = current.map_points[j] = frame.map_points[j] = point
            point.add_observation(first, i)
            point.add_observation(current, j)
            point.compute_descriptor()
            point.update_normal_and_depth()
        first.update_connections()
        current.update_connections()
        self._global_visual_ba(iterations=20)
        median_depth = first.median_depth()
        if median_depth < 0 or current.tracked_points(1) < 50:
            self._new_map(discard=True)
            return
        scale = (4. if self.sensor.inertial else 1.)/median_depth
        current.pose[:3, 3] *= scale
        for point in self.atlas.current.points.values():
            point.position *= scale
            point.update_normal_and_depth()
        frame.pose = current.pose.copy()
        synchronize_body_pose(current)
        synchronize_body_pose(frame)
        if self.sensor.inertial:
            first.next_id, current.previous_id = current.id, first.id
            self.initial.last_keyframe_id = first.last_keyframe_id = first.id
            frame.last_keyframe_id = current.last_keyframe_id = first.id
            first.preintegrated, current.preintegrated = None, self.accumulated_imu
            self.accumulated_imu = Preintegrated(current.body_state.bias, self.calibration.imu)
        self.pending_keyframes.extend([first, current])
        self.reference = self.last_keyframe = current
        self.last_keyframe_frame_id = frame.id
        self.initial_extractor_keyframe_id = current.id
        self.local_keyframes = [current, first]
        self.local_points = list(self.atlas.current.points.values())
        self.velocity = None
        self.state = TrackingState.OK
        self.inliers = current.tracked_points()

    def _global_visual_ba(self, iterations=20):
        from .mapping import reprojection_observations
        active = self.atlas.current
        observations = reprojection_observations(active.keyframes.values())
        result = bundle_adjustment({i: k.pose for i, k in active.keyframes.items()},
                                   {i: p.position for i, p in active.points.items()}, observations,
                                   {active.initial_keyframe_id}, iterations=iterations)
        for i, pose in result.poses.items():
            active.keyframes[i].pose = pose
            synchronize_body_pose(active.keyframes[i])
        for i, position in result.points.items():
            active.points[i].position = position

    def _visual_pose(self, frame):
        indices, points, observations = frame_observations(frame)
        if not observations:
            return 0
        result = optimize_pose(frame.pose, points, observations)
        frame.pose = result.pose
        frame.outliers[:] = False
        frame.outliers[indices] = result.outliers
        synchronize_body_pose(frame)
        self.diagnostics["pose_chi2"] = result.chi2.tolist()
        return result.inliers

    def _discard_outliers(self, frame):
        count = 0
        for i, point in enumerate(frame.map_points):
            if point is not None:
                if frame.outliers[i]:
                    point.last_seen_frame = frame.id
                    frame.map_points[i], frame.outliers[i] = None, False
                elif point.observation_count > 0:
                    count += 1
        return count

    def _track_reference(self, frame):
        frame.compute_bow(self.vocabulary)
        frame.map_points = search_by_bow_frame(self.reference, frame, nnratio=.7)
        if sum(p is not None for p in frame.map_points) < 15:
            return False
        frame.pose = self.last.pose.copy()
        self._visual_pose(frame)
        matches = self._discard_outliers(frame)
        return self.sensor.inertial or matches >= 10

    def _predict_imu(self, frame, map_updated):
        base = self.last_keyframe if map_updated else self.last
        preintegrated = frame.preintegrated if map_updated else frame.frame_preintegrated
        if base is None or base.body_state is None or preintegrated is None:
            return False
        state = base.body_state
        rotation, position, velocity = predict_body_state(state.rotation, state.position, state.velocity, preintegrated, state.bias)
        frame.body_state = BodyState(rotation, position, velocity, state.bias)
        synchronize_camera_pose(frame)
        return True

    def _track_motion(self, frame, map_updated):
        self._update_last_frame()
        if self.atlas.current.imu_initialized and frame.id > self.last_reloc_id+self.max_frames:
            return self._predict_imu(frame, map_updated)
        frame.pose = self.velocity @ self.last.pose
        threshold = 7 if self.sensor.stereo and not self.sensor.inertial else 15
        frame.map_points = [None]*len(frame.features)
        matches = search_by_projection_last_frame(frame, self.last, threshold, self.sensor.monocular)
        if matches < 20:
            frame.map_points = [None]*len(frame.features)
            matches = search_by_projection_last_frame(frame, self.last, 2*threshold, self.sensor.monocular)
        if matches < 20:
            return self.sensor.inertial
        self._visual_pose(frame)
        rejected = sum(point is not None and frame.outliers[index]
                       for index, point in enumerate(frame.map_points))
        map_matches = self._discard_outliers(frame)
        if self.only_tracking:
            self.visual_odometry = map_matches < 10
            self.inliers = matches-rejected
            return matches-rejected > 20
        return self.sensor.inertial or map_matches >= 10

    def _update_last_frame(self):
        """Correct the previous pose and seed source localization depth points."""
        if self.history:
            self.last.pose = self.history[-1].pose(self.atlas.all_keyframes)
        if (self.last.id == self.last_keyframe_frame_id or self.sensor.monocular
                or not self.only_tracking):
            return
        n = self.last.n_left if self.last.n_left >= 0 else len(self.last.features)
        depths = sorted((self.last.depth[i], i) for i in range(n) if self.last.depth[i] > 0)
        for count, (depth, index) in enumerate(depths, start=1):
            existing = self.last.map_points[index]
            if existing is None or existing.observation_count < 1:
                position = self.last.unproject_depth(index)
                if position is None:
                    continue
                direction = position-self.last.center_for(index)
                distance = np.linalg.norm(direction)
                if distance == 0:
                    continue
                maximum_distance = distance*self.last.scales[self.last.features.octaves[index]]
                point = MapPoint(self.atlas.next_point_id, position, -1, self.atlas.current,
                                 descriptor=self.last.features.descriptors[index].copy(),
                                 normal=direction/distance, max_distance=maximum_distance,
                                 min_distance=maximum_distance/self.last.scales[-1])
                # IDs share the source global counter, although these temporary
                # points have no observations and are not inserted into a map.
                self.atlas.next_point_id += 1
                self.last.map_points[index] = point
                self.temporary_points.append(point)
            if depth > self.last.depth_threshold and count > 100:
                break

    def _update_local_map(self, frame):
        active = self.atlas.current
        voting = self.last if active.imu_initialized and frame.id >= self.last_reloc_id+2 else frame
        counts = Counter()
        for i, point in enumerate(voting.map_points):
            if point is None:
                continue
            if point.bad:
                voting.map_points[i] = None
            else:
                counts.update(point.observations.keys())
        local = [active.keyframes[i] for i in sorted(counts) if i in active.keyframes and not active.keyframes[i].bad]
        seen = {k.id for k in local}
        for keyframe in local.copy():
            if len(local) > 80:
                break
            for neighbors in (keyframe.best_covisible(10), [active.keyframes[i] for i in sorted(keyframe.children) if i in active.keyframes]):
                for other in neighbors:
                    if not other.bad and other.id not in seen:
                        local.append(other)
                        seen.add(other.id)
                        break
            if keyframe.parent_id in active.keyframes and keyframe.parent_id not in seen:
                parent = active.keyframes[keyframe.parent_id]
                local.append(parent)
                seen.add(parent.id)
                break
        if self.sensor.inertial and len(local) < 80:
            temporal = self.last_keyframe
            for _ in range(20):
                if temporal is None:
                    break
                if temporal.id not in seen:
                    local.append(temporal)
                    seen.add(temporal.id)
                    temporal = active.keyframes.get(temporal.previous_id)
        if counts:
            valid = [i for i in counts if i in active.keyframes and not active.keyframes[i].bad]
            if valid:
                self.reference = active.keyframes[max(valid, key=lambda i: (counts[i], -i))]
        points, point_ids = [], set()
        for keyframe in reversed(local):
            for point in keyframe.map_points:
                if point is not None and not point.bad and point.id not in point_ids:
                    points.append(point)
                    point_ids.add(point.id)
        self.local_keyframes, self.local_points = local, points

    def _track_local_map(self, frame, map_updated):
        self._update_local_map(frame)
        matched = set()
        for i, point in enumerate(frame.map_points):
            if point is not None:
                if point.bad:
                    frame.map_points[i] = None
                else:
                    point.visible += 1
                    point.last_seen_frame = frame.id
                    matched.add(point.id)
        candidates = []
        for point in self.local_points:
            if point.id in matched or point.last_seen_frame == frame.id:
                continue
            visible = _geometry_candidate(frame, point) is not None
            if frame.n_left >= 0:
                visible = visible or _geometry_candidate(frame, point, right=True) is not None
            if visible:
                point.visible += 1
                candidates.append(point)
        threshold = 3 if "rgbd" in self.sensor.value else 1
        active = self.atlas.current
        if active.imu_initialized:
            threshold = 2 if active.inertial_ba2 else 6
        elif self.sensor.inertial:
            threshold = 10
        if frame.id < self.last_reloc_id+2:
            threshold = 5
        if self.state in (TrackingState.LOST, TrackingState.RECENTLY_LOST):
            threshold = 15
        search_by_projection_local(frame, candidates, threshold,
                                   far_points=self.far_points, far_threshold=self.far_threshold)
        if not active.imu_initialized or frame.id <= self.last_reloc_id+self.max_frames:
            self._visual_pose(frame)
        else:
            previous = self.last_keyframe if map_updated else self.last
            preintegrated = frame.preintegrated if map_updated else frame.frame_preintegrated
            prior = None if map_updated else previous.state_prior
            indices, points, observations = frame_observations(frame, pose_id=1)
            result = optimize_inertial_tracking(previous.body_state, frame.body_state, preintegrated,
                                                points, observations, self.calibration.imu,
                                                previous_prior=prior,
                                                track_depths={point.id: getattr(point, "track_depth", np.inf)
                                                              for point in frame.map_points if point is not None})
            frame.body_state, frame.state_prior = result.states[1], result.prior
            frame.outliers[:] = False
            frame.outliers[indices] = result.outliers
            synchronize_camera_pose(frame)
        self.inliers = 0
        for i, point in enumerate(frame.map_points):
            if point is None:
                continue
            if not frame.outliers[i]:
                point.found += 1
                self.inliers += self.only_tracking or point.observation_count > 0
            elif self.sensor.stereo and not self.sensor.inertial:
                frame.map_points[i] = None
        if frame.id < self.last_reloc_id+self.max_frames and self.inliers < 50:
            return False
        if self.inliers > 10 and self.state == TrackingState.RECENTLY_LOST:
            return True
        threshold = (15 if active.imu_initialized else 50) if self.sensor.monocular and self.sensor.inertial else 15 if self.sensor.inertial else 30
        return self.inliers >= threshold

    def _needs_keyframe(self, frame):
        if self.last_keyframe is None or self.reference is None:
            return False
        n = frame.n_left if frame.n_left >= 0 else len(frame.features)
        close = (frame.depth[:n] > 0) & (frame.depth[:n] < frame.depth_threshold)
        tracked = np.array([point is not None and not frame.outliers[i] for i, point in enumerate(frame.map_points[:n])], dtype=bool)
        return need_new_keyframe(self.sensor, frame_id=frame.id, timestamp=frame.timestamp,
                                last_id=self.last_keyframe_frame_id, last_timestamp=self.last_keyframe.timestamp,
                                last_reloc_id=self.last_reloc_id, max_frames=self.max_frames,
                                map_keyframes=len(self.atlas.current.keyframes),
                                reference_matches=self.reference.tracked_points(2 if len(self.atlas.current.keyframes) <= 2 else 3),
                                inliers=self.inliers, tracked_close=int((close & tracked).sum()), untracked_close=int((close & ~tracked).sum()),
                                state=self.state, imu_initialized=self.atlas.current.imu_initialized,
                                dual_camera=frame.right_camera is not None, only_tracking=self.only_tracking)

    def _create_keyframe(self, frame):
        keyframe = self.atlas.add_keyframe(frame)
        if self.last_keyframe is not None:
            keyframe.previous_id, self.last_keyframe.next_id = self.last_keyframe.id, keyframe.id
        self.reference = self.last_keyframe = keyframe
        if self.sensor.inertial:
            self.accumulated_imu = Preintegrated(keyframe.body_state.bias, self.calibration.imu)
        if not self.sensor.monocular:
            self._seed_points(frame, keyframe)
        self.pending_keyframes.append(keyframe)
        self.last_keyframe_frame_id = frame.id
        return keyframe

    def _relocalize(self, frame):
        from .pnp import MLPnPSolver
        frame.compute_bow(self.vocabulary)
        candidates = self.database.relocalization_candidates(frame.bow, self.atlas.current.id)
        solvers = []
        for identifier in candidates:
            keyframe = self.atlas.current.keyframes.get(identifier)
            if keyframe is None or keyframe.bad:
                continue
            matches = search_by_bow_frame(keyframe, frame, nnratio=.75)
            if sum(p is not None for p in matches) >= 15:
                solver = MLPnPSolver(frame, matches)
                solver.set_ransac_parameters(.99, 10, 300, 6, .5, 5.991)
                solvers.append((keyframe, matches, solver))
        while solvers:
            for entry in solvers.copy():
                keyframe, matches, solver = entry
                solution = solver.iterate(5)
                if solution.exhausted:
                    solvers.remove(entry)
                if solution.pose is None:
                    continue
                frame.pose = solution.pose
                frame.map_points = [point if inside else None for point, inside in zip(matches, solution.inliers)]
                found = {point for point in frame.map_points if point is not None}
                good = self._visual_pose(frame)
                if good < 10:
                    continue
                for i in np.flatnonzero(frame.outliers):
                    frame.map_points[i] = None
                if good < 50:
                    additional = search_by_projection_relocalization(frame, keyframe, found, 10, 100)
                    if additional+good >= 50:
                        good = self._visual_pose(frame)
                        if 30 < good < 50:
                            found = {point for point in frame.map_points if point is not None}
                            additional = search_by_projection_relocalization(frame, keyframe, found, 3, 64)
                            if good+additional >= 50:
                                good = self._visual_pose(frame)
                                for i in np.flatnonzero(frame.outliers):
                                    frame.map_points[i] = None
                if good >= 50:
                    self.last_reloc_id = frame.id
                    return True
        return False

    def _live_reference(self, reference):
        while reference.bad and reference.parent_id is not None:
            reference = self.atlas.all_keyframes[reference.parent_id]
        return reference

    def _update_frame_imu(self, bias, keyframe_id):
        """UpdateFrameIMU propagation from EACH frame's preintegration anchor.

        A newly inserted keyframe changes the tracker's latest keyframe, but
        an older frame's accumulated delta still starts at its own last KF.
        C++ stores that distinction in Frame::mpLastKeyFrame.
        """
        self.last_keyframe = self.atlas.all_keyframes[keyframe_id]
        visited = set()
        for frame in (self.last, self.current):
            if frame is None or id(frame) in visited:
                continue
            visited.add(id(frame))
            if frame.body_state is not None:
                frame.body_state.bias = copy.deepcopy(bias)
            if frame.preintegrated is not None:
                frame.preintegrated.set_new_bias(bias)
            reference = self.atlas.all_keyframes.get(frame.last_keyframe_id)
            if reference is None or reference.body_state is None:
                # The first frame has no preceding integration interval.
                continue
            base = reference.body_state
            if frame is self.last and frame.id == reference.source_frame_id:
                frame.body_state = BodyState(base.rotation, base.position, base.velocity, bias)
            elif frame.preintegrated is not None:
                rotation, position, velocity = predict_body_state(
                    base.rotation, base.position, base.velocity, frame.preintegrated, bias)
                frame.body_state = BodyState(rotation, position, velocity, bias)
            else:
                continue
            synchronize_camera_pose(frame)

    def apply_inertial_corrections(self, corrections):
        """Apply (map_id, scale, bias, current_keyframe_id) mapper updates.

        The coordinator calls this before loop processing/map transfer, as
        LocalMapping::InitializeIMU calls Tracking::UpdateFrameIMU in C++.
        """
        for map_id, scale, bias, keyframe_id in corrections:
            for entry in self.history:
                if self._live_reference(entry.reference).map.id == map_id:
                    entry.relative_pose[:3, 3] *= scale
            if bias is None:
                bias = self.atlas.all_keyframes[keyframe_id].body_state.bias
            self._update_frame_imu(bias, keyframe_id)

    def apply_merge_correction(self, merge, *, history_keyframe_ids):
        """MergeLocal2's UpdateFrameIMU for the pre-merge current map only.

        Map transfer has already completed at the sequential loop boundary.
        Membership captured before that transfer prevents scaling history
        from the donor map. Keyframes are already corrected; non-keyframe
        poses must first follow the world Sim3 and then their own IMU anchor.
        """
        scale, transform = merge["scale"], merge["world_transform"]
        for entry in self.history:
            if self._live_reference(entry.reference).id in history_keyframe_ids:
                entry.relative_pose[:3, 3] *= scale
        visited = set()
        for frame in (self.last, self.current):
            if frame is None or frame.pose is None or id(frame) in visited:
                continue
            visited.add(id(frame))
            anchor = self.atlas.all_keyframes.get(frame.last_keyframe_id)
            visual_reference = self.atlas.all_keyframes.get(frame.reference_id)
            reference = anchor if anchor is not None else visual_reference
            if reference is not None and self._live_reference(reference).id not in history_keyframe_ids:
                continue
            world_camera = inverse_pose(frame.pose)
            world_camera[:3, :3] = transform.rotation @ world_camera[:3, :3]
            world_camera[:3, 3] = transform.map(world_camera[:3, 3])
            frame.pose = inverse_pose(world_camera)
            if frame.body_state is not None:
                velocity = scale * transform.rotation @ frame.body_state.velocity
                synchronize_body_pose(frame)
                frame.body_state.velocity = velocity
        if self.last_keyframe is not None:
            bias = merge.get("bias")
            if bias is None:
                bias = self.last_keyframe.body_state.bias
            self._update_frame_imu(bias, self.last_keyframe.id)
