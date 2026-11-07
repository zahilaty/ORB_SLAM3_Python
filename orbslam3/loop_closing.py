"""Sequential place-recognition, loop correction, and map-merge coordination.

Translated from LoopClosing.cc's current NewDetectCommonRegions pipeline.
BoW retrieval proposes candidates; Horn RANSAC, bilateral Sim3 optimization,
and projection across covisible keyframes verify them. Loop correction fuses
landmarks, constructs the essential graph, and runs the source's global BA.

Global BA runs synchronously at the loop boundary in this implementation. The
objective and variable selection are preserved; updates consequently become
available at a deterministic boundary rather than after a background worker.

Derived from ORB-SLAM3, University of Zaragoza contributors, 2014-2021.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from dataclasses import dataclass, field
import copy

import numpy as np

from .geometry import Sim3, inverse_pose, make_pose, so3_exp, so3_log, transform_points
from .input import Sensor
from .matching import search_by_bow_keyframes, search_by_projection_sim3, fuse_sim3
from .loop_geometry import (Sim3ReprojectionMatch, optimize_sim3, essential_graph_edges,
                            optimize_essential_graph, correct_map_points)
from .sim3_solver import Sim3Solver
from .optimization import bundle_adjustment


def _similarity(pose):
    return Sim3(pose[:3, :3].copy(), pose[:3, 3].copy(), 1.)


def _valid(point):
    return point is not None and not point.bad


def _set_pose(keyframe, pose):
    keyframe.pose = pose.copy()
    if keyframe.body_state is not None and keyframe.imu_calibration is not None:
        world_body = inverse_pose(pose) @ keyframe.imu_calibration.T_camera_body
        keyframe.body_state.rotation = world_body[:3, :3]
        keyframe.body_state.position = world_body[:3, 3]


def _point_cloud(keyframes):
    points, owners, seen = [], [], set()
    for keyframe in keyframes:
        for point in keyframe.map_points:
            if _valid(point) and point.id not in seen:
                points.append(point)
                owners.append(keyframe)
                seen.add(point.id)
    return points, owners


def _optimization_matches(current, matched, matched_points, all_points):
    correspondences, indices = [], []
    for i, point2 in enumerate(matched_points):
        point1 = current.map_points[i]
        if not _valid(point1) or not _valid(point2):
            continue
        if current.n_left >= 0 and i >= current.n_left:
            continue
        index2 = point2.observations.get(matched.id, (-1, -1))[0]
        if index2 < 0 and not all_points:
            continue
        point1_camera = transform_points(current.pose, point1.position)
        point2_camera = transform_points(matched.pose, point2.position)
        if point2_camera[2] < 0:
            continue
        if index2 >= 0:
            pixel2 = matched.features.points[index2]
            variance2 = matched.inverse_variances[matched.features.octaves[index2]]
        else:
            # OptimizeSim3's bAllPoints branch synthesizes NORMALIZED coordinates
            # and leaves cv::KeyPoint::octave at zero; do not silently turn
            # these into camera-model pixels.
            pixel2 = point2_camera[:2] / point2_camera[2]
            variance2 = matched.inverse_variances[0]
        correspondences.append(Sim3ReprojectionMatch(
            point1_camera, point2_camera, current.features.points[i], pixel2,
            current.camera, matched.camera,
            current.inverse_variances[current.features.octaves[i]], variance2,
            observed2=index2 >= 0
        ))
        indices.append(i)
    return correspondences, indices


@dataclass
class RegionHypothesis:
    matched: object | None = None
    last_current: object | None = None
    similarity: Sim3 | None = None
    coincidences: int = 0
    misses: int = 0
    points: list = field(default_factory=list)
    matches: list = field(default_factory=list)
    detected: bool = False


class LoopCloser:
    def __init__(self, atlas, database, vocabulary, sensor, *, seed=0, active=True):
        self.atlas, self.database, self.vocabulary = atlas, database, vocabulary
        self.sensor = Sensor(sensor) if not isinstance(sensor, Sensor) else sensor
        # System.cc only frees scale for purely visual monocular input. The
        # source selectively relaxes it again for pre-BA2 monocular-inertial.
        self.fix_scale = self.sensor is not Sensor.MONOCULAR
        self.seed, self.active = seed, active
        self.loop = RegionHypothesis()
        self.merge = RegionHypothesis()
        self.last_current = None
        self.last_loop_id = 0
        self.corrections = 0

    def _release(self, keyframe):
        if keyframe is None:
            return
        if not keyframe.loop_edges:
            keyframe.not_erase = False
        if keyframe.to_be_erased and not keyframe.not_erase:
            keyframe.set_bad(self.database if keyframe.id in self.database.entries else None)

    def _clear(self, hypothesis):
        self._release(hypothesis.last_current)
        self._release(hypothesis.matched)
        hypothesis.coincidences = hypothesis.misses = 0
        hypothesis.points = []
        hypothesis.matches = []
        hypothesis.detected = False

    def reset_active_map(self, map_id):
        # Source removes queued KFs only. This sequential pipeline has no queue.
        # Keep valid archived maps/candidates available for later merging. A
        # discarded map cannot continue a geometric hypothesis safely.
        for hypothesis in (self.loop, self.merge):
            keyframe = hypothesis.last_current
            if keyframe is not None and keyframe.map.id == map_id and keyframe.map.bad:
                self._clear(hypothesis)
        self.last_loop_id = self.atlas.current.initial_keyframe_id

    def _sync_database(self):
        for identifier, entry in self.database.entries.items():
            keyframe = self.atlas.all_keyframes.get(identifier)
            if keyframe is not None:
                entry.map_id = keyframe.map.id
                entry.bad = keyframe.bad
                entry.neighbors = list(keyframe.ordered_covisibility)
        self.database.bad_maps = {identifier for identifier, map_ in self.atlas.maps.items() if map_.bad}
        self.database.bad_maps.update(self.atlas.bad_maps)

    def _index(self, keyframe):
        if keyframe.id not in self.database.entries:
            self.database.add(keyframe.id, keyframe.bow, keyframe.map.id,
                               keyframe.ordered_covisibility)

    def _fixed_scale(self, current):
        return self.fix_scale and not (self.sensor is Sensor.MONOCULAR_IMU and not current.map.inertial_ba2)

    def _find_projection_matches(self, current, matched, similarity):
        neighborhood = matched.best_covisible(10)
        initial_count = len(neighborhood)
        neighborhood.append(matched)
        checked = {keyframe.id for keyframe in neighborhood}
        current_connected = set(current.covisibility)
        if initial_count < 10:
            for i in range(initial_count):
                neighbors = neighborhood[i].best_covisible(10)
                inserted = 0
                for neighbor in neighbors:
                    if inserted >= 10:
                        break
                    if neighbor.id not in checked and neighbor.id not in current_connected:
                        checked.add(neighbor.id)
                        inserted += 1
                # The source inserts the WHOLE neighbor vector despite checking
                # the set above; repeated/unfiltered KFs are intentional.
                neighborhood.extend(neighbors)
        points, _ = _point_cloud(neighborhood)
        matches = [None] * len(current.features)
        count = search_by_projection_sim3(current, similarity, points, matches, 3, 1.5)
        return count, points, matches

    def _optimize_matches(self, current, matched, matches, initial, *, fix_scale, all_points=True):
        correspondences, indices = _optimization_matches(current, matched, matches, all_points)
        result = optimize_sim3(initial, correspondences, 10, fix_scale=fix_scale, all_points=all_points)
        for index, inlier in zip(indices, result.inliers):
            if not inlier:
                matches[index] = None
        return result

    def _refine_previous_region(self, current, matched, similarity):
        count, points, matches = self._find_projection_matches(current, matched, similarity)
        if count < 30:
            return False, similarity, points, matches
        camera_similarity = similarity @ _similarity(inverse_pose(matched.pose))
        optimized = self._optimize_matches(current, matched, matches, camera_similarity,
                                           fix_scale=self._fixed_scale(current))
        if optimized.count <= 50:
            return False, similarity, points, matches
        # The original computes an optimized Scm above but never uses it here.
        # It repeats projection with the propagated ORIGINAL transform, scale=1.
        estimation = Sim3(similarity.rotation, similarity.translation, 1.)
        count, points, matches = self._find_projection_matches(current, matched, estimation)
        return count >= 100, estimation, points, matches

    def _continue_hypothesis(self, current, hypothesis, *, merge=False):
        if hypothesis.coincidences <= 0:
            return False
        relative = _similarity(current.pose @ inverse_pose(hypothesis.last_current.pose))
        predicted = relative @ hypothesis.similarity
        valid, similarity, points, matches = self._refine_previous_region(current, hypothesis.matched, predicted)
        if valid:
            hypothesis.coincidences += 1
            self._release(hypothesis.last_current)
            hypothesis.last_current = current
            hypothesis.similarity = similarity
            hypothesis.points, hypothesis.matches = points, matches
            hypothesis.detected = hypothesis.coincidences >= 3
            # Merge's success branch does not reset mnMergeNumNotFound.
            if not merge:
                hypothesis.misses = 0
            return True
        if merge:
            hypothesis.detected = False
        hypothesis.misses += 1
        if hypothesis.misses >= 2:
            self._clear(hypothesis)
        return False

    def _ransac_candidate(self, current, candidate, matches):
        points1, points2, variances1, variances2, indices = [], [], [], [], []
        for i, point2 in enumerate(matches):
            point1 = current.map_points[i]
            if not _valid(point1) or not _valid(point2):
                continue
            index1 = point1.observations.get(current.id, (-1, -1))[0]
            index2 = point2.observations.get(candidate.id, (-1, -1))[0]
            # Sim3Solver's nonempty provenance-vector path still uses the base
            # candidate, not each supplied observation keyframe, in this source.
            if index1 < 0 or index2 < 0:
                continue
            points1.append(transform_points(current.pose, point1.position))
            points2.append(transform_points(candidate.pose, point2.position))
            variances1.append(1 / current.inverse_variances[current.features.octaves[index1]])
            variances2.append(1 / candidate.inverse_variances[candidate.features.octaves[index2]])
            indices.append(i)
        if len(indices) < 15:
            return None
        solver = Sim3Solver(np.asarray(points1), np.asarray(points2), current.camera, candidate.camera,
                             np.asarray(variances1), np.asarray(variances2),
                             fix_scale=self._fixed_scale(current), original_indices=indices,
                             output_size=len(matches), seed=self.seed + current.id + candidate.id)
        solver.set_ransac_parameters(.99, 15, 300)
        result = solver.iterate(20)
        while not result.converged and not result.exhausted:
            result = solver.iterate(20)
        return result.transform if result.converged else None

    def _detect_bow_region(self, current, candidates, hypothesis):
        best = None
        best_matches = 0
        connected = set(current.covisibility)
        for candidate in candidates:
            if candidate is None or candidate.bad:
                continue
            neighbors = candidate.best_covisible(10)
            if neighbors:
                neighbors.append(neighbors[0])
                neighbors[0] = candidate
            else:
                neighbors = [candidate]
            if any(neighbor.id in connected for neighbor in neighbors):
                continue
            bow_matches = [search_by_bow_keyframes(current, neighbor, .9, True)
                           if neighbor is not None and not neighbor.bad else [] for neighbor in neighbors]
            matched_points = [None] * len(current.features)
            unique = set()
            bow_count = 0
            for matches in bow_matches:
                for index, point in enumerate(matches):
                    if _valid(point) and point.id not in unique:
                        unique.add(point.id)
                        bow_count += 1
                        matched_points[index] = point
            if bow_count < 20:
                continue
            # pMostBoWMatchesKF stays equal to candidate in this source; its
            # computed best-neighbor index is never assigned to that pointer.
            camera_similarity = self._ransac_candidate(current, candidate, matched_points)
            if camera_similarity is None:
                continue
            points, owners = _point_cloud(candidate.best_covisible(10) + [candidate])
            world_similarity = camera_similarity @ _similarity(candidate.pose)
            matches = [None] * len(current.features)
            matched_owners = [None] * len(current.features)
            projected = search_by_projection_sim3(current, world_similarity, points, matches, 8, 1.5,
                                                   point_keyframes=owners, matched_keyframes=matched_owners)
            if projected < 50:
                continue
            # Source computes relaxed bFixedScale here but passes mbFixScale.
            optimized = self._optimize_matches(current, candidate, matches, camera_similarity,
                                               fix_scale=self.fix_scale)
            if optimized.count < 20:
                continue
            world_similarity = optimized.transform @ _similarity(candidate.pose)
            matches = [None] * len(current.features)
            projected = search_by_projection_sim3(current, world_similarity, points, matches, 5, 1.)
            if projected < 80:
                continue
            coincidences = 0
            for neighbor in current.best_covisible(10):
                relative = _similarity(neighbor.pose @ inverse_pose(current.pose))
                count, _, _ = self._find_projection_matches(neighbor, candidate, relative @ world_similarity)
                if count >= 30:
                    coincidences += 1
                if coincidences >= 3:
                    break
            if projected > best_matches:
                best_matches = projected
                best = (candidate, world_similarity, coincidences, points, matches)
        if best is not None:
            matched, similarity, coincidences, points, matches = best
            hypothesis.matched = matched
            hypothesis.last_current = current
            hypothesis.similarity = similarity
            hypothesis.coincidences = coincidences
            hypothesis.points, hypothesis.matches = points, matches
            hypothesis.detected = coincidences >= 3
            matched.not_erase = True
        return hypothesis.detected

    def detect_common_regions(self, current):
        current.not_erase = True
        current.place_recognition = True
        current.compute_bow(self.vocabulary)
        map_ = current.map
        self._sync_database()
        if (map_.inertial and not map_.inertial_ba2) or len(map_.keyframes) < 12:
            self._index(current)
            self._release(current)
            return False
        loop_in_current = self._continue_hypothesis(current, self.loop)
        merge_in_current = self._continue_hypothesis(current, self.merge, merge=True)
        if self.loop.detected or self.merge.detected:
            self._index(current)
            return True
        if not merge_in_current or not loop_in_current:
            loops, merges = self.database.detect_n_best(current.bow, map_.id,
                                                        connected=set(current.covisibility), count=3)
            if not loop_in_current and loops:
                self._detect_bow_region(current, [self.atlas.all_keyframes.get(i) for i in loops], self.loop)
            if not merge_in_current and merges:
                self._detect_bow_region(current, [self.atlas.all_keyframes.get(i) for i in merges], self.merge)
        self._index(current)
        if self.loop.detected or self.merge.detected:
            return True
        self._release(current)
        current.place_recognition = False
        return False

    def _fuse(self, corrected, points, map_):
        fused = 0
        for identifier, similarity in corrected.items():
            keyframe = map_.keyframes[identifier]
            count, replacements = fuse_sim3(keyframe, similarity, points, 4)
            fused += count
            for replacement, point in zip(replacements, points):
                if replacement is not None:
                    replacement.replace(point)
        return fused

    def correct_loop(self, current, matched, similarity, matches, loop_points):
        map_ = current.map
        current.update_connections()
        connected = current.best_covisible(len(current.ordered_covisibility)) + [current]
        original_current_inverse = inverse_pose(current.pose)
        noncorrected = {keyframe.id: _similarity(keyframe.pose) for keyframe in connected}
        corrected = {keyframe.id: (_similarity(keyframe.pose @ original_current_inverse) @ similarity)
                     for keyframe in connected}
        corrected[current.id] = similarity
        for keyframe in connected:
            _set_pose(keyframe, corrected[keyframe.id].camera_pose())
        corrected_references = {}
        for identifier in sorted(corrected):
            keyframe = map_.keyframes[identifier]
            old, new = noncorrected[identifier], corrected[identifier]
            for point in keyframe.map_points:
                if not _valid(point) or point.id in corrected_references:
                    continue
                point.position = new.inverse().map(old.map(point.position))
                corrected_references[point.id] = identifier
                point.update_normal_and_depth()
            if map_.imu_initialized and keyframe.body_state is not None:
                keyframe.body_state.velocity = new.rotation.T @ old.rotation @ keyframe.body_state.velocity
            keyframe.update_connections()
        map_.change_index += 1
        for index, loop_point in enumerate(matches):
            if loop_point is None:
                continue
            current_point = current.map_points[index]
            if current_point is not None:
                current_point.replace(loop_point)
            else:
                current.map_points[index] = loop_point
                loop_point.add_observation(current, index)
                loop_point.compute_descriptor()
        fused = self._fuse(corrected, loop_points, map_)
        loop_connections = {}
        connected_ids = {keyframe.id for keyframe in connected}
        for keyframe in connected:
            prior_neighbors = set(keyframe.ordered_covisibility)
            keyframe.update_connections()
            loop_connections[keyframe.id] = set(keyframe.covisibility) - prior_neighbors - connected_ids
        keyframes = {i: keyframe for i, keyframe in map_.keyframes.items() if not keyframe.bad}
        initial = {i: corrected.get(i, _similarity(keyframe.pose)) for i, keyframe in keyframes.items()}
        four_dof = map_.inertial and map_.imu_initialized
        edges = essential_graph_edges(initial, noncorrected=noncorrected,
                 parents={i: keyframe.parent_id for i, keyframe in keyframes.items()},
                 historical_loops={i: keyframe.loop_edges for i, keyframe in keyframes.items()},
                 covisibility={i: keyframe.covisibility for i, keyframe in keyframes.items()},
                 previous={i: keyframe.previous_id for i, keyframe in keyframes.items()
                           if four_dof or keyframe.body_state is not None},
                 loop_connections=loop_connections, current_id=current.id,
                 loop_id=matched.id, four_dof=four_dof)
        fixed = {matched.id} if four_dof else {map_.initial_keyframe_id}
        extrinsics = {i: keyframe.imu_calibration.T_camera_body for i, keyframe in keyframes.items()
                      if keyframe.imu_calibration is not None}
        graph = optimize_essential_graph(initial, edges, fixed, fix_scale=self._fixed_scale(current),
                                          four_dof=four_dof, camera_from_body=extrinsics)
        for identifier, pose in graph.poses.items():
            _set_pose(keyframes[identifier], pose)
        references = {i: (point.reference_id if four_dof else corrected_references.get(i, point.reference_id))
                      for i, point in map_.points.items() if not point.bad}
        point_updates = correct_map_points({i: map_.points[i].position for i in references},
                                            references, initial, graph.similarities)
        for identifier, position in point_updates.items():
            point = map_.points[identifier]
            point.position = position
            point.update_normal_and_depth()
        map_.change_index += 1
        self.atlas.big_change_index = getattr(self.atlas, "big_change_index", 0) + 1
        current.loop_edges.add(matched.id)
        matched.loop_edges.add(current.id)
        current.not_erase = matched.not_erase = True
        global_result = None
        if not map_.imu_initialized or (len(map_.keyframes) < 200 and len(self.atlas.maps) == 1):
            global_result = self.global_bundle_adjustment(map_)
        self.last_loop_id = current.id
        self.corrections += 1
        return {"fused": fused, "graph_edges": len(edges), "graph_initial_cost": graph.initial_cost,
                "graph_final_cost": graph.final_cost, "four_dof": four_dof, "global_ba": global_result}

    def global_bundle_adjustment(self, map_):
        from .mapping import reprojection_observations
        keyframes = {i: keyframe for i, keyframe in map_.keyframes.items() if not keyframe.bad}
        points = {i: point.position for i, point in map_.points.items() if not point.bad}
        observations = reprojection_observations(keyframes.values())
        poses_before = {i: keyframe.pose.copy() for i, keyframe in keyframes.items()}
        optimized_states = {}
        if map_.imu_initialized:
            from .inertial_optimization import InertialLink, full_inertial_bundle_adjustment
            states = {i: keyframe.body_state for i, keyframe in keyframes.items() if keyframe.body_state is not None}
            calibrations = {i: keyframe.imu_calibration for i, keyframe in keyframes.items() if keyframe.body_state is not None}
            links = [InertialLink(keyframe.previous_id, i, keyframe.preintegrated)
                     for i, keyframe in keyframes.items() if i in states and keyframe.previous_id in states
                     and keyframe.preintegrated is not None]
            result = full_inertial_bundle_adjustment(states, points, observations, links, calibrations, iterations=7)
            optimized_states = result.states
            optimized_poses = {i: calibrations[i].T_camera_body @ inverse_pose(make_pose(state.rotation, state.position))
                               for i, state in optimized_states.items()}
        else:
            result = bundle_adjustment(poses_before, points, observations,
                                        {map_.initial_keyframe_id}, iterations=10, robust=False)
            optimized_poses = result.poses
        self.apply_global_ba(map_, optimized_poses, result.points, poses_before, optimized_states)
        return {"initial_cost": result.initial_cost, "final_cost": result.final_cost,
                "iterations": result.iterations, "inertial": map_.imu_initialized}

    def apply_global_ba(self, map_, optimized_poses, optimized_points, poses_before, optimized_states=None):
        """Commit BA and propagate through new descendants/reference landmarks.

        Sequential BA creates no concurrent descendants; retaining this source
        propagation also makes the map update correct if an external scheduler
        solves a snapshot and commits after inserting keyframes.
        """
        optimized_states = {} if optimized_states is None else optimized_states
        pending = list(map_.origins)
        updates = dict(optimized_poses)
        visited = set()
        while pending:
            identifier = pending.pop(0)
            if identifier in visited or identifier not in map_.keyframes:
                continue
            visited.add(identifier)
            keyframe = map_.keyframes[identifier]
            if keyframe.bad:
                continue
            old_pose = poses_before.get(identifier, keyframe.pose.copy())
            poses_before.setdefault(identifier, old_pose)
            for child_id in sorted(keyframe.children):
                child = map_.keyframes.get(child_id)
                if child is None or child.bad:
                    continue
                if child_id not in updates and identifier in updates:
                    updates[child_id] = child.pose @ inverse_pose(old_pose) @ updates[identifier]
                    poses_before.setdefault(child_id, child.pose.copy())
                pending.append(child_id)
        for identifier, pose in updates.items():
            keyframe = map_.keyframes.get(identifier)
            if keyframe is None or keyframe.bad:
                continue
            if identifier in optimized_states:
                keyframe.body_state = copy.deepcopy(optimized_states[identifier])
            elif keyframe.body_state is not None:
                keyframe.body_state.velocity = pose[:3, :3].T @ poses_before[identifier][:3, :3] @ keyframe.body_state.velocity
            _set_pose(keyframe, pose)
            if keyframe.body_state is not None and keyframe.preintegrated is not None:
                keyframe.preintegrated.set_new_bias(keyframe.body_state.bias)
        for identifier, point in map_.points.items():
            if point.bad:
                continue
            if identifier in optimized_points:
                point.position = optimized_points[identifier].copy()
            elif point.reference_id in updates and point.reference_id in poses_before:
                reference = point.reference_id
                point.position = transform_points(inverse_pose(updates[reference]),
                                                   transform_points(poses_before[reference], point.position))
        map_.big_change_index = getattr(map_, "big_change_index", 0) + 1
        map_.change_index += 1

    def process(self, current):
        result = {"loop": False, "merge": False, "pose_corrections": {}, "diagnostics": {}}
        if not self.active or current.id == 0 or current.bad:
            return result
        before = {i: keyframe.pose.copy() for i, keyframe in self.atlas.all_keyframes.items() if not keyframe.bad}
        detected = self.detect_common_regions(current)
        result["diagnostics"].update(loop_coincidences=self.loop.coincidences,
                                     merge_coincidences=self.merge.coincidences,
                                     loop_misses=self.loop.misses, merge_misses=self.merge.misses)
        if detected and self.merge.detected:
            matched = self.merge.matched
            world_transform = self.merge.similarity.inverse() @ _similarity(current.pose)
            abort_reason = None
            if self.sensor.inertial and not current.map.imu_initialized:
                abort_reason = "imu_not_initialized"
            elif current.map.inertial and matched.map.inertial:
                if not .90 <= world_transform.scale <= 1.10:
                    abort_reason = "inertial_merge_scale"
                elif self.sensor.inertial and current.map.inertial_ba1:
                    angle = so3_log(world_transform.rotation)
                    world_transform = Sim3(so3_exp([0., 0., angle[2]]), world_transform.translation, 1.)
            if abort_reason is None:
                from .merging import merge_maps
                merge_result = merge_maps(self.atlas, current, matched, self.merge.similarity,
                                           self.sensor, self.database, world_transform=world_transform)
                result["merge"] = merge_result.get("merged", True)
                result["diagnostics"]["merge"] = merge_result
            else:
                result["diagnostics"]["merge_aborted"] = abort_reason
            self._clear(self.merge)
            # Source's bad-scale continue skips the loop/reset handling this KF.
            if abort_reason == "inertial_merge_scale":
                return result
            if abort_reason != "inertial_merge_scale" and self.loop.detected:
                self._clear(self.loop)
        if detected and self.loop.detected:
            similarity = self.loop.similarity
            good_loop = True
            if current.map.inertial:
                correction = _similarity(inverse_pose(current.pose)) @ similarity
                angles = so3_log(correction.rotation)
                good_loop = abs(angles[0]) < .008 and abs(angles[1]) < .008 and abs(angles[2]) < .349
                if good_loop and self.sensor.inertial and current.map.inertial_ba2:
                    correction = Sim3(so3_exp([0., 0., angles[2]]), correction.translation, 1.)
                    similarity = _similarity(current.pose) @ correction
            if good_loop:
                result["diagnostics"]["loop"] = self.correct_loop(current, self.loop.matched, similarity,
                                                                    self.loop.matches, self.loop.points)
                result["loop"] = True
            else:
                result["diagnostics"]["loop_aborted"] = "inertial_rotation"
            self._clear(self.loop)
        self.last_current = current
        self._sync_database()
        for identifier, old_pose in before.items():
            keyframe = self.atlas.all_keyframes.get(identifier)
            if keyframe is not None and not keyframe.bad and not np.array_equal(old_pose, keyframe.pose):
                result["pose_corrections"][identifier] = keyframe.pose @ inverse_pose(old_pose)
        return result
