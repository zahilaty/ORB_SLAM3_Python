"""ORB-SLAM3 visual and visual-inertial map welding.

Source: LoopClosing::{MergeLocal,MergeLocal2,SearchAndFuse}, the merging
OptimizeEssentialGraph overload, and Optimizer::MergeInertialBA. All geometry
and graph construction is in Python; background-worker synchronization becomes
explicit sequential boundaries. See docs/merging_audit.md for source quirks.

Derived from ORB-SLAM3, Copyright (C) 2017-2021 University of Zaragoza and
Copyright (C) 2014-2016 the ORB-SLAM2 authors. GPL-3.0-or-later.
"""

from collections import Counter
import copy
import numpy as np

from .features import hamming
from .geometry import Sim3, inverse_pose, transform_points
from .imu import rotation_exp, rotation_log
from .input import Sensor
from .matching import fuse_sim3
from .optimization import merge_bundle_adjustment
from .loop_geometry import EssentialGraphEdge, optimize_essential_graph
from .inertial_mapping import body_state_from_camera, camera_pose_from_body
from .inertial_optimization import (
    InertialLink, refine_inertial_biases, merge_inertial_bundle_adjustment,
)


def _sim3(pose):
    return Sim3(pose[:3, :3], pose[:3, 3], 1.)


def _valid_keyframes(keyframes):
    return [keyframe for keyframe in keyframes if keyframe is not None and not keyframe.bad]


def _points_observed(keyframes):
    return {point.id: point for keyframe in keyframes for point in keyframe.map_points
            if point is not None and not point.bad}


def visual_welding_window(center, count=25):
    """MergeLocal's initial neighborhood and five covisibility expansions."""
    window = {center.id: center}
    window.update((keyframe.id, keyframe) for keyframe in center.best_covisible(count))
    for _ in range(5):
        if len(window) >= count:
            break
        additions = {}
        for keyframe in tuple(window.values()):
            additions.update((neighbor.id, neighbor) for neighbor in keyframe.best_covisible(count // 2)
                             if not neighbor.bad and neighbor.id not in window)
        window.update(additions)
    return sorted(window.values(), key=lambda keyframe: keyframe.id)


def _reverse_parent_chain(child, new_parent, all_keyframes):
    """Attach the welding KF, reversing its old parent path to retain a tree."""
    previous_parent = child.parent_id
    if previous_parent in all_keyframes:
        all_keyframes[previous_parent].children.discard(child.id)
    child.parent_id = new_parent.id
    new_parent.children.add(child.id)
    parent, new_child, seen = previous_parent, child, {child.id, new_parent.id}
    while parent is not None:
        if parent in seen or parent not in all_keyframes:
            raise ValueError("invalid/cyclic spanning tree while merging maps")
        seen.add(parent)
        old_parent = all_keyframes[parent]
        next_parent = old_parent.parent_id
        old_parent.children.discard(new_child.id)
        if next_parent in all_keyframes:
            all_keyframes[next_parent].children.discard(old_parent.id)
        old_parent.parent_id = new_child.id
        new_child.children.add(old_parent.id)
        new_child, parent = old_parent, next_parent


def _global_descriptor(point, all_keyframes):
    descriptors = []
    for identifier, indices in sorted(point.observations.items()):
        keyframe = all_keyframes.get(identifier)
        if keyframe is not None and not keyframe.bad:
            descriptors.extend(keyframe.features.descriptors[index] for index in indices if index >= 0)
    if descriptors:
        descriptors = np.asarray(descriptors)
        distances = hamming(descriptors[:, None], descriptors[None])
        medians = np.sort(distances, axis=1)[:, (len(descriptors) - 1) // 2]
        point.descriptor = descriptors[np.argmin(medians)].copy()


def _global_normal_depth(point, all_keyframes):
    """MapPoint::UpdateNormalAndDepth with cross-map observer references.

    The original stores pointers, so an observation remains accessible while
    its KF is outside the transferred local window. This avoids refreshing a
    wider set of connections/descriptors than the source at a merge boundary.
    """
    if point.bad or not point.observations:
        return
    directions = []
    for identifier, indices in sorted(point.observations.items()):
        keyframe = all_keyframes[identifier]
        for index in indices:
            if index >= 0:
                direction = point.position - keyframe.center_for(index)
                directions.append(direction / np.linalg.norm(direction))
    point.normal = np.mean(directions, axis=0)
    reference = all_keyframes[point.reference_id]
    index = next(index for index in point.observations[point.reference_id] if index >= 0)
    distance = np.linalg.norm(point.position - reference.center)
    point.max_distance = distance * reference.scales[reference.features.octaves[index]]
    point.min_distance = point.max_distance / reference.scales[-1]


def _replace_across_maps(previous, retained, all_keyframes):
    """MapPoint::Replace while some observing KFs still belong to the donor."""
    if previous.id == retained.id:
        return
    for identifier, indices in previous.observations.copy().items():
        keyframe = all_keyframes.get(identifier)
        if keyframe is None:
            raise ValueError("map-point observation references an absent keyframe")
        already_observed = identifier in retained.observations
        for index in indices:
            if index < 0:
                continue
            if already_observed:
                keyframe.map_points[index] = None
            else:
                keyframe.map_points[index] = retained
                retained.add_observation(keyframe, index)
    previous.bad, previous.replaced = True, retained
    previous.observations.clear()
    retained.found += previous.found
    retained.visible += previous.visible
    previous.map.points.pop(previous.id, None)
    _global_descriptor(retained, all_keyframes)


def _search_and_fuse(corrected, candidates, all_keyframes):
    replacements_count = 0
    for keyframe, similarity in corrected:
        _, replacements = fuse_sim3(keyframe, similarity, candidates, threshold=4.)
        for old, retained in zip(replacements, candidates):
            if old is not None:
                _replace_across_maps(old, retained, all_keyframes)
                replacements_count += 1
    return replacements_count


def _transfer_keyframes(source, destination, keyframes):
    for keyframe in _valid_keyframes(keyframes):
        if keyframe.map is source:
            if not hasattr(keyframe, "inertial"):
                keyframe.inertial = source.imu_initialized
            source.keyframes.pop(keyframe.id)
            destination.keyframes[keyframe.id] = keyframe
            keyframe.map = destination


def _transfer_points(source, destination, points):
    for point in points:
        if not point.bad and point.map is source:
            source.points.pop(point.id, None)
            destination.points[point.id] = point
            point.map = destination


def _set_pose(keyframe, pose, velocity=None):
    keyframe.pose = np.asarray(pose, float).copy()
    if getattr(keyframe, "body_state", None) is not None:
        previous = keyframe.body_state
        keyframe.body_state = body_state_from_camera(keyframe.pose, keyframe.imu_calibration,
                                                      previous.velocity if velocity is None else velocity,
                                                      previous.bias)


def _synchronize_database(map_, database=None):
    # Source updates only the welding neighborhoods' covisibility and the
    # points actually optimized/corrected. A whole-map metadata refresh here
    # would change when new connections/descriptors become available.
    if database is not None:
        for identifier, keyframe in map_.keyframes.items():
            if identifier in database.entries:
                database.entries[identifier].map_id = map_.id
                database.entries[identifier].neighbors = keyframe.ordered_covisibility.copy()
        database.bad_maps.discard(map_.id)


def _visual_welding_ba(current, local_window, merge_window, all_keyframes):
    from .mapping import reprojection_observations, _erase_outlier_observations
    keyframes = {keyframe.id: keyframe for keyframe in _valid_keyframes(merge_window + local_window)
                 if keyframe.map is current.map}
    points = {identifier: point for identifier, point in _points_observed(keyframes.values()).items()
              if point.map is current.map}
    # The original visual welding overload builds left observations only.
    observations = [observation for observation in reprojection_observations(keyframes.values(), set(points))
                    if observation.camera_from_reference is None]
    result = merge_bundle_adjustment(
        {identifier: keyframe.pose for identifier, keyframe in keyframes.items()},
        {identifier: point.position for identifier, point in points.items()}, observations,
        {keyframe.id for keyframe in merge_window if keyframe.id in keyframes})
    erased = _erase_outlier_observations(current.map, observations, result.outliers)
    for identifier, pose in result.poses.items():
        _set_pose(keyframes[identifier], pose)
    for identifier, position in result.points.items():
        if identifier in current.map.points:
            current.map.points[identifier].position = position.copy()
            _global_normal_depth(current.map.points[identifier], all_keyframes)
    return {"initial_cost": result.initial_cost, "final_cost": result.final_cost,
            "iterations": result.iterations, "erased_observations": erased}


def merging_essential_graph_edges(fixed, corrected, remaining, before_merge):
    """Literal good/bad-pose selection of the merging graph overload.

    The source uses the old inverse pose for its first endpoint even when
    both endpoints have corrected poses. For fixed target-map frames there is
    no old inverse pose and the default identity is used. This inherited
    behavior is intentionally visible rather than silently repaired.
    """
    frames = {keyframe.id: keyframe for keyframe in fixed + corrected + remaining if not keyframe.bad}
    good = {keyframe.id for keyframe in fixed + corrected if not keyframe.bad}
    bad = {keyframe.id for keyframe in corrected + remaining if not keyframe.bad}
    current = {identifier: _sim3(keyframe.pose) for identifier, keyframe in frames.items()}
    old = {identifier: _sim3(before_merge.get(identifier, keyframe.pose))
           for identifier, keyframe in frames.items() if identifier in bad}
    edges = []
    for identifier, keyframe in sorted(frames.items()):
        inverse_first = old[identifier].inverse() if identifier in bad else Sim3(np.eye(3), np.zeros(3))

        def add(second, kind):
            if second not in frames:
                return
            if identifier in good and second in good:
                other = current[second]
            elif identifier in bad and second in bad:
                other = old[second]
            else:
                return
            edges.append(EssentialGraphEdge(identifier, second, other @ inverse_first, np.eye(7), kind))

        if keyframe.parent_id in frames:
            add(keyframe.parent_id, "spanning_tree")
        for second in sorted(keyframe.loop_edges):
            if second < identifier:
                add(second, "historical_loop")
        for second, weight in sorted(keyframe.covisibility.items()):
            if (weight >= 100 and second < identifier and second != keyframe.parent_id
                    and second not in keyframe.children and second not in keyframe.loop_edges):
                add(second, "covisibility")
    return current, edges


def _propagate_visual_graph(current, merge_window, local_window, remaining, remaining_points, before_merge, all_keyframes):
    initial, edges = merging_essential_graph_edges(merge_window, local_window, remaining, before_merge)
    fixed = {keyframe.id for keyframe in merge_window + local_window if keyframe.id in initial}
    result = optimize_essential_graph(initial, edges, fixed, iterations=20)
    old_remaining = {keyframe.id: keyframe.pose.copy() for keyframe in remaining}
    for keyframe in remaining:
        _set_pose(keyframe, result.poses[keyframe.id])
    for point in remaining_points:
        reference = point.reference_id
        if reference in old_remaining:
            point.position = transform_points(inverse_pose(result.poses[reference]),
                                                transform_points(old_remaining[reference], point.position))
        elif reference in before_merge and reference in result.poses:
            point.position = transform_points(inverse_pose(result.poses[reference]),
                                                transform_points(before_merge[reference], point.position))
        else:
            raise ValueError("noncritical merge point has no valid old-map reference keyframe")
        _global_normal_depth(point, all_keyframes)
    return {"initial_cost": result.initial_cost, "final_cost": result.final_cost, "iterations": result.iterations}


def _merge_visual(atlas, current, matched, similarity, sensor, database):
    source, target = current.map, matched.map
    source_id, target_id = source.id, target.id
    current.update_connections()
    local_window, merge_window = visual_welding_window(current), visual_welding_window(matched)
    local_points = _points_observed(local_window)
    target_points = list(_points_observed(merge_window).values())
    old_current = current.pose.copy()
    before_merge = {keyframe.id: keyframe.pose.copy() for keyframe in source.keyframes.values()}
    corrected = {keyframe.id: _sim3(keyframe.pose @ inverse_pose(old_current)) @ similarity for keyframe in local_window}
    original = {keyframe.id: _sim3(keyframe.pose) for keyframe in local_window}
    for keyframe in local_window:
        _set_pose(keyframe, corrected[keyframe.id].camera_pose())
    transferable_points = []
    for point in local_points.values():
        if point.reference_id in corrected:
            point.position = corrected[point.reference_id].inverse().map(original[point.reference_id].map(point.position))
            point.normal = corrected[point.reference_id].rotation.T @ original[point.reference_id].rotation @ point.normal
            transferable_points.append(point)
    _transfer_keyframes(source, target, local_window)
    _transfer_points(source, target, transferable_points)
    if source.initial_keyframe_id in atlas.all_keyframes:
        atlas.all_keyframes[source.initial_keyframe_id].first_connection = False
    _reverse_parent_chain(current, matched, atlas.all_keyframes)
    matched.update_connections()
    fused = _search_and_fuse([(keyframe, corrected[keyframe.id]) for keyframe in local_window],
                             target_points, atlas.all_keyframes)
    for keyframe in local_window + merge_window:
        keyframe.update_connections()
    welding = _visual_welding_ba(current, local_window, merge_window, atlas.all_keyframes)
    remaining = _valid_keyframes(list(source.keyframes.values()))
    remaining_points = [point for point in source.points.values() if not point.bad]
    graph = None
    if remaining:
        if sensor.monocular:
            for keyframe in remaining:
                original[keyframe.id] = _sim3(keyframe.pose)
                corrected[keyframe.id] = _sim3(keyframe.pose @ inverse_pose(old_current)) @ similarity
                _set_pose(keyframe, corrected[keyframe.id].camera_pose())
            for point in remaining_points:
                reference = point.reference_id
                point.position = corrected[reference].inverse().map(original[reference].map(point.position))
                _global_normal_depth(point, atlas.all_keyframes)
        else:
            graph = _propagate_visual_graph(current, merge_window, local_window, remaining,
                                             remaining_points, before_merge, atlas.all_keyframes)
        _transfer_keyframes(source, target, remaining)
        _transfer_points(source, target, remaining_points)
    # Source changes the surviving Map's ID to that of the formerly active map.
    atlas.maps.pop(source_id, None)
    atlas.maps.pop(target_id, None)
    source.bad = True
    target.id = source_id
    atlas.maps[target.id], atlas.current = target, target
    # Source RemoveBadMaps clears its bad-map set after completing this branch.
    atlas.bad_maps.clear()
    current.merge_edges.add(matched.id)
    matched.merge_edges.add(current.id)
    current.not_erase = matched.not_erase = True
    source.change_index += 1
    target.change_index += 2
    _synchronize_database(target, database)
    return {"merged": True, "branch": "visual", "active_map_id": target.id,
            "fused_points": fused, "welding_ba": welding, "essential_graph": graph,
            "scale": 1. / similarity.scale, "world_transform": similarity.inverse() @ _sim3(old_current),
            "updated_keyframes": tuple(target.keyframes), "retained_empty_map": False}


def inertial_welding_window(current, matched, all_keyframes):
    """MergeInertialBA: six current states, older-map boundary, up to 12 states."""
    temporal = [current]
    for _ in range(1, 6):
        previous = all_keyframes.get(temporal[-1].previous_id)
        if previous is None:
            break
        temporal.append(previous)
    previous = all_keyframes.get(temporal[-1].previous_id)
    covisible = [previous] if previous is not None else [temporal.pop()]
    temporal.append(matched)
    for _ in range(1, 3):
        previous = all_keyframes.get(temporal[-1].previous_id)
        if previous is None:
            break
        temporal.append(previous)
    previous = all_keyframes.get(temporal[-1].previous_id)
    fixed = [previous] if previous is not None else [temporal.pop()]
    following = all_keyframes.get(matched.next_id)
    if following is not None:
        temporal.append(following)
    while len(temporal) < 12:
        following = all_keyframes.get(temporal[-1].next_id)
        if following is None:
            break
        # The original may revisit preceding merge-window entries here when
        # matched.next_id is absent; its bounded vector then duplicates edges.
        # Retain those repeated temporal entries rather than changing weights.
        temporal.append(following)
    counts, points = Counter(), {}
    for keyframe in temporal:
        for point in keyframe.map_points:
            if point is not None and not point.bad:
                counts[point.id] += 1
                points[point.id] = point
    selected = {keyframe.id for keyframe in temporal + covisible + fixed}
    # sortByVal in Optimizer.cc:47 surprisingly orders increasing counts.
    for identifier in sorted(points, key=lambda identifier: (counts[identifier], identifier))[:30]:
        for observer in sorted(points[identifier].observations):
            if observer not in selected:
                selected.add(observer)
                keyframe = all_keyframes.get(observer)
                if keyframe is not None and not keyframe.bad:
                    covisible.append(keyframe)
                    break
    return temporal, covisible, fixed, points


def _inertial_welding_ba(current, matched, atlas):
    from .mapping import reprojection_observations, _erase_outlier_observations
    temporal, covisible, fixed, points = inertial_welding_window(current, matched, atlas.all_keyframes)
    selected = {keyframe.id: keyframe for keyframe in temporal + covisible + fixed}
    if any(keyframe.body_state is None or keyframe.imu_calibration is None for keyframe in selected.values()):
        raise ValueError("inertial welding requires body states and calibration on its complete window")
    observations = [observation for observation in reprojection_observations(selected.values(), set(points))
                    if observation.pose_id <= current.id and observation.camera_from_reference is None]
    links = [InertialLink(keyframe.previous_id, keyframe.id, keyframe.preintegrated)
             for keyframe in temporal if keyframe.previous_id in selected and keyframe.preintegrated is not None]
    result = merge_inertial_bundle_adjustment(
        {identifier: keyframe.body_state for identifier, keyframe in selected.items()},
        {identifier: point.position for identifier, point in points.items()}, observations, links,
        {identifier: keyframe.imu_calibration for identifier, keyframe in selected.items()},
        {keyframe.id for keyframe in temporal}, {keyframe.id for keyframe in covisible}, {keyframe.id for keyframe in fixed},
        inertial_ids={identifier for identifier, keyframe in selected.items()
                      if getattr(keyframe, "inertial", keyframe.map.imu_initialized)})
    erased = _erase_outlier_observations(current.map, observations, result.outliers)
    for identifier, state in result.states.items():
        keyframe = selected[identifier]
        keyframe.body_state = state
        keyframe.pose = camera_pose_from_body(state, keyframe.imu_calibration)
        if keyframe.preintegrated is not None:
            keyframe.preintegrated.set_new_bias(state.bias)
    for identifier, position in result.points.items():
        if identifier in current.map.points:
            current.map.points[identifier].position = position.copy()
            current.map.points[identifier].update_normal_and_depth()
    current.map.change_index += 1
    return {"initial_cost": result.initial_cost, "final_cost": result.final_cost,
            "iterations": result.iterations, "erased_observations": erased,
            "temporal_ids": tuple(keyframe.id for keyframe in temporal),
            "covisible_ids": tuple(keyframe.id for keyframe in covisible),
            "fixed_ids": tuple(keyframe.id for keyframe in fixed)}


def _transform_inertial_map(map_, similarity):
    # Map::ApplyScaledRotation rotates world, scales camera centers and velocities.
    states = {identifier: copy.deepcopy(keyframe.body_state) for identifier, keyframe in map_.keyframes.items()
              if keyframe.body_state is not None}
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = similarity.rotation, similarity.translation
    map_.apply_scaled_rotation(transform, similarity.scale, scale_velocity=similarity.scale != 1.)
    for identifier, state in states.items():
        keyframe = map_.keyframes[identifier]
        keyframe.body_state = body_state_from_camera(keyframe.pose, keyframe.imu_calibration,
                                                      similarity.scale * similarity.rotation @ state.velocity, state.bias)


def _merge_inertial(atlas, current, matched, world_transform, database, tracking_keyframe):
    target, donor = current.map, matched.map
    original_count = len(target.keyframes)
    _transform_inertial_map(target, world_transform)
    if not target.inertial_ba2:
        keyframes = list(target.keyframes.values())
        states = {keyframe.id: keyframe.body_state for keyframe in keyframes}
        if any(state is None for state in states.values()):
            raise ValueError("inertial merging requires initialized body states")
        links = [InertialLink(keyframe.previous_id, keyframe.id, keyframe.preintegrated)
                 for keyframe in keyframes if keyframe.previous_id in states and keyframe.preintegrated is not None]
        estimate = refine_inertial_biases(states, links)
        for identifier, state in estimate.states.items():
            keyframe = target.keyframes[identifier]
            keyframe.body_state = state
            if keyframe.preintegrated is not None:
                keyframe.preintegrated.set_new_bias(state.bias)
                if identifier in estimate.reintegrate_ids:
                    keyframe.preintegrated.reintegrate()
        target.imu_initialized = target.inertial_ba1 = target.inertial_ba2 = True
    origin = donor.keyframes.get(donor.initial_keyframe_id)
    _transfer_keyframes(donor, target, list(donor.keyframes.values()))
    _transfer_points(donor, target, list(donor.points.values()))
    if origin is not None:
        origin.first_connection = False
    _reverse_parent_chain(matched, current, atlas.all_keyframes)
    merge_connected = _valid_keyframes([matched] + matched.best_covisible(len(matched.ordered_covisibility)))[:6]
    current.update_connections()
    current_connected = _valid_keyframes([current] + current.best_covisible(len(current.ordered_covisibility)))[:6]
    candidates = {}
    for keyframe in merge_connected:
        candidates.update(_points_observed([keyframe]))
        if len(candidates) > 1000:
            break
    fused = _search_and_fuse([(keyframe, _sim3(keyframe.pose)) for keyframe in current_connected],
                             list(candidates.values()), atlas.all_keyframes)
    for keyframe in current_connected + merge_connected:
        keyframe.update_connections()
    welding = None
    if original_count >= 10:
        welding = _inertial_welding_ba(tracking_keyframe or current, matched, atlas)
    # This exact MergeLocal2 implementation retains an empty donor Atlas map.
    atlas.current = target
    _synchronize_database(target, database)
    return {"merged": True, "branch": "inertial", "active_map_id": target.id,
            "fused_points": fused, "welding_ba": welding, "essential_graph": None,
            "scale": world_transform.scale, "world_transform": world_transform,
            "bias": copy.deepcopy(current.body_state.bias), "updated_keyframes": tuple(target.keyframes),
            "retained_empty_map": True}


def merge_maps(atlas, current, matched, similarity, sensor, database=None,
                tracking_keyframe=None, world_transform=None):
    """Weld two Atlas maps using a verified current-camera<-matched-world Sim3.

    Optional world_transform is the source's already gravity-restricted
    ``mSold_new`` (matched-world <- current-world). This function also enforces
    the source inertial initialization/scale gates for direct callers.
    """
    sensor = Sensor(sensor)
    if current.bad or matched.bad or current.map is matched.map:
        return {"merged": False, "reason": "invalid_or_same_map"}
    if sensor.inertial and not current.map.imu_initialized:
        return {"merged": False, "reason": "imu_not_initialized"}
    transform = similarity.inverse() @ _sim3(current.pose) if world_transform is None else world_transform
    if current.map.inertial and matched.map.inertial:
        if transform.scale < .9 or transform.scale > 1.1:
            return {"merged": False, "reason": "inertial_scale_gate"}
        if sensor.inertial and current.map.inertial_ba1:
            yaw = rotation_log(transform.rotation)[2]
            transform = Sim3(rotation_exp([0., 0., yaw]), transform.translation, 1.)
    if sensor.inertial:
        return _merge_inertial(atlas, current, matched, transform, database, tracking_keyframe)
    return _merge_visual(atlas, current, matched, similarity, sensor, database)
