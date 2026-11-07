"""Loop coordinator state decisions and real geometric loop correction."""

import numpy as np
from types import SimpleNamespace

from orbslam3.camera import Pinhole
from orbslam3.features import FeatureSet
from orbslam3.geometry import Sim3, inverse_pose, make_pose, so3_exp
from orbslam3.input import Sensor
from orbslam3.loop_closing import LoopCloser, RegionHypothesis
from orbslam3.map import Atlas, Frame
from orbslam3.vocabulary import InvertedDatabase


class SyntheticVocabulary:
    def transform(self, descriptors, levelsup=4):
        return {0: 1.}, {0: list(range(len(descriptors)))}

    def score(self, first, second):
        return sum(min(first.get(word, 0), second.get(word, 0)) for word in first.keys() & second.keys())


def make_keyframe(atlas, points=None, descriptors=None, pose=None, drift=None):
    camera = Pinhole(500, 500, 320, 240)
    points = np.empty((0, 3)) if points is None else points
    descriptors = np.empty((0, 32), np.uint8) if descriptors is None else descriptors
    n = len(points)
    pose = np.eye(4) if pose is None else pose
    local = points @ pose[:3, :3].T + pose[:3, 3]
    pixels = camera.project(local) if n else np.empty((0, 2))
    features = FeatureSet(pixels.astype(np.float32), descriptors, np.zeros(n, np.int32),
                           np.zeros(n, np.float32), np.ones(n, np.float32), np.full(n, 31., np.float32), n)
    actual_pose = pose.copy() if drift is None else pose @ inverse_pose(drift)
    scales = 1.2 ** np.arange(8)
    frame = Frame(atlas.next_keyframe_id, atlas.next_keyframe_id * .1, features, camera,
                   640, 480, scales, 1 / scales**2, pose=actual_pose, baseline_fx=40.,
                   right_u=pixels[:, 0] - 40 / local[:, 2] if n else np.empty(0))
    keyframe = atlas.add_keyframe(frame)
    keyframe.compute_bow(SyntheticVocabulary())
    return keyframe


def coordinator(atlas, sensor=Sensor.RGBD):
    vocabulary = SyntheticVocabulary()
    return LoopCloser(atlas, InvertedDatabase(vocabulary), vocabulary, sensor)


def test_small_map_and_inertial_ba2_gates_still_index_keyframes():
    atlas = Atlas()
    keys = [make_keyframe(atlas) for _ in range(4)]
    closer = coordinator(atlas)
    assert not closer.process(keys[-1])["loop"]
    assert keys[-1].id in closer.database.entries
    assert not keys[-1].not_erase
    for _ in range(8):
        current = make_keyframe(atlas)
    atlas.current.inertial = True
    assert not closer.process(current)["loop"]
    assert current.id in closer.database.entries


def test_temporal_three_coincidences_and_two_failure_reset(monkeypatch):
    atlas = Atlas()
    keys = [make_keyframe(atlas) for _ in range(5)]
    closer = coordinator(atlas)
    hypothesis = RegionHypothesis(keys[0], keys[1], Sim3(np.eye(3), np.zeros(3)), 1)
    keys[0].not_erase = keys[1].not_erase = True
    monkeypatch.setattr(closer, "_refine_previous_region",
                        lambda current, matched, prediction: (True, prediction, [], []))
    assert closer._continue_hypothesis(keys[2], hypothesis)
    assert hypothesis.coincidences == 2 and not hypothesis.detected
    assert not keys[1].not_erase
    assert closer._continue_hypothesis(keys[3], hypothesis)
    assert hypothesis.coincidences == 3 and hypothesis.detected
    monkeypatch.setattr(closer, "_refine_previous_region",
                        lambda current, matched, prediction: (False, prediction, [], []))
    closer._continue_hypothesis(keys[4], hypothesis)
    assert hypothesis.coincidences == 3 and hypothesis.misses == 1
    closer._continue_hypothesis(keys[4], hypothesis)
    assert hypothesis.coincidences == 0 and hypothesis.misses == 0
    assert not keys[0].not_erase


def test_source_merge_success_does_not_clear_previous_failure(monkeypatch):
    atlas = Atlas()
    keys = [make_keyframe(atlas) for _ in range(3)]
    closer = coordinator(atlas)
    hypothesis = RegionHypothesis(keys[0], keys[1], Sim3(np.eye(3), np.zeros(3)),
                                   coincidences=1, misses=1)
    monkeypatch.setattr(closer, "_refine_previous_region",
                        lambda current, matched, prediction: (True, prediction, [], []))
    closer._continue_hypothesis(keys[2], hypothesis, merge=True)
    assert hypothesis.coincidences == 2 and hypothesis.misses == 1


def test_reset_preserves_archived_map_database_but_releases_discarded_hypothesis():
    atlas = Atlas()
    first, current = make_keyframe(atlas), make_keyframe(atlas)
    closer = coordinator(atlas)
    closer._index(current)
    closer.loop = RegionHypothesis(first, current, Sim3(np.eye(3), np.zeros(3)), 1)
    old_map = atlas.current
    atlas.create_map()
    closer.reset_active_map(old_map.id)
    assert closer.loop.coincidences == 1
    assert current.id in closer.database.entries
    old_map.bad = True
    closer.reset_active_map(old_map.id)
    assert closer.loop.coincidences == 0
    assert current.id in closer.database.entries


def test_inertial_loop_rotation_gate_rejects_roll(monkeypatch):
    atlas = Atlas(inertial=True)
    matched, current = make_keyframe(atlas), make_keyframe(atlas)
    atlas.current.imu_initialized = atlas.current.inertial_ba2 = True
    closer = coordinator(atlas, Sensor.RGBD_IMU)
    closer.loop = RegionHypothesis(matched, current, Sim3(so3_exp([.01, 0, 0]), np.zeros(3)),
                                    coincidences=3, detected=True)
    monkeypatch.setattr(closer, "detect_common_regions", lambda keyframe: True)
    result = closer.process(current)
    assert not result["loop"]
    assert result["diagnostics"]["loop_aborted"] == "inertial_rotation"
    assert closer.loop.coincidences == 0


def test_bad_inertial_merge_scale_skips_a_simultaneous_loop(monkeypatch):
    atlas = Atlas(inertial=True)
    first, current = make_keyframe(atlas), make_keyframe(atlas)
    atlas.current.imu_initialized = atlas.current.inertial_ba2 = True
    atlas.create_map()
    matched = make_keyframe(atlas)
    closer = coordinator(atlas, Sensor.RGBD_IMU)
    closer.merge = RegionHypothesis(matched, current, Sim3(np.eye(3), np.zeros(3), 2.),
                                     coincidences=3, detected=True)
    closer.loop = RegionHypothesis(first, current, Sim3(np.eye(3), np.zeros(3)),
                                    coincidences=3, detected=True)
    monkeypatch.setattr(closer, "detect_common_regions", lambda keyframe: True)
    def unexpected_loop(*arguments):
        raise AssertionError("Source continues before handling the simultaneous loop")
    monkeypatch.setattr(closer, "correct_loop", unexpected_loop)
    result = closer.process(current)
    assert result["diagnostics"]["merge_aborted"] == "inertial_merge_scale"
    assert not result["loop"] and not result["merge"]
    assert closer.merge.coincidences == 0
    assert closer.loop.detected and closer.loop.coincidences == 3


def test_temporal_refinement_source_discards_optimized_similarity(monkeypatch):
    atlas = Atlas()
    matched, current = make_keyframe(atlas), make_keyframe(atlas)
    closer = coordinator(atlas)
    propagated = Sim3(so3_exp([.01, .02, .03]), np.array([.1, .2, .3]), 1.2)
    optimized = Sim3(np.eye(3), np.array([8., 9., 10.]), 2.)
    calls = []
    def projection(frame, candidate, similarity):
        calls.append(similarity)
        return (30 if len(calls) == 1 else 100), [], []
    monkeypatch.setattr(closer, "_find_projection_matches", projection)
    monkeypatch.setattr(closer, "_optimize_matches", lambda *args, **kwargs:
                        SimpleNamespace(count=51, transform=optimized))
    valid, returned, _, _ = closer._refine_previous_region(current, matched, propagated)
    assert valid and returned.scale == 1
    np.testing.assert_array_equal(returned.translation, propagated.translation)
    np.testing.assert_array_equal(returned.rotation, propagated.rotation)


def test_global_ba_commit_propagates_to_later_spanning_tree_children_and_points():
    atlas = Atlas()
    parent = make_keyframe(atlas)
    child = make_keyframe(atlas, pose=make_pose(translation=[-.5, 0, 0]))
    child.change_parent(parent)
    point = atlas.add_point([0., 0., 5.], child)
    closer = coordinator(atlas)
    optimized_parent = make_pose(translation=[0., -.2, 0.])
    closer.apply_global_ba(atlas.current, {parent.id: optimized_parent}, {}, {parent.id: np.eye(4)})
    np.testing.assert_allclose(child.pose, make_pose(translation=[-.5, -.2, 0.]))
    np.testing.assert_allclose(point.position, [0., .2, 5.])
    assert atlas.current.big_change_index == 1


def real_loop_scene():
    atlas = Atlas()
    rng = np.random.default_rng(6)
    world = rng.uniform([-.9, -.65, 4.], [.9, .65, 6.], (120, 3))
    descriptors = rng.integers(0, 256, (len(world), 32), dtype=np.uint8)
    drift = make_pose(translation=[.3, -.04, .02])
    keys = []
    for region in range(2):
        for local in range(6):
            keys.append(make_keyframe(atlas, world, descriptors,
                                       pose=make_pose(translation=[-.02 * local, 0, 0]),
                                       drift=drift if region else None))
    for region in range(2):
        reference = keys[region * 6]
        for i, position in enumerate(world):
            if region:
                position = position + drift[:3, 3]
            point = atlas.add_point(position, reference)
            for keyframe in keys[region*6:(region+1)*6]:
                point.add_observation(keyframe, i)
                keyframe.map_points[i] = point
            point.compute_descriptor()
            point.update_normal_and_depth()
    for i, keyframe in enumerate(keys):
        keyframe.update_connections()
        if i:
            if keyframe.parent_id is not None:
                atlas.current.keyframes[keyframe.parent_id].children.discard(keyframe.id)
            keyframe.change_parent(keys[i-1])
        keyframe.first_connection = False
    return atlas, keys, world


def test_real_bow_horn_projection_sim3_graph_fusion_and_global_ba_loop():
    atlas, keys, world = real_loop_scene()
    closer = coordinator(atlas)
    for keyframe in keys[1:-1]:
        closer._index(keyframe)
    current = keys[-1]
    initial_pose = current.pose.copy()
    result = closer.process(current)
    assert result["loop"] and not result["merge"]
    assert len(current.loop_edges) == 1
    matched_id = next(iter(current.loop_edges))
    assert current.id in atlas.current.keyframes[matched_id].loop_edges
    assert current.not_erase
    assert current.id in result["pose_corrections"]
    assert np.linalg.norm(current.pose[:3, 3] - initial_pose[:3, 3]) > .1
    assert len(atlas.current.points) == 120
    assert all(current.map_points[i] is keys[0].map_points[i] for i in range(120))
    diagnostics = result["diagnostics"]["loop"]
    assert diagnostics["graph_edges"] > 0
    assert diagnostics["global_ba"]["final_cost"] <= diagnostics["global_ba"]["initial_cost"]
    assert closer.loop.coincidences == 0


def test_real_cross_map_bow_geometry_and_visual_welding_merge():
    atlas = Atlas()
    rng = np.random.default_rng(9)
    world = rng.uniform([-.8, -.6, 4.], [.8, .6, 6.], (100, 3))
    descriptors = rng.integers(0, 256, (len(world), 32), dtype=np.uint8)
    all_keys = []
    regions = []
    drift = make_pose(translation=[.25, -.02, .01])
    for region, count in enumerate((6, 12)):
        if region:
            atlas.create_map()
        keys = [make_keyframe(atlas, world, descriptors,
                              pose=make_pose(translation=[-.01 * local, 0, 0]),
                              drift=drift if region else None) for local in range(count)]
        for i, position in enumerate(world):
            point = atlas.add_point(position + (drift[:3, 3] if region else 0), keys[0])
            for keyframe in keys:
                point.add_observation(keyframe, i)
                keyframe.map_points[i] = point
            point.compute_descriptor()
            point.update_normal_and_depth()
        for local, keyframe in enumerate(keys):
            keyframe.update_connections()
            if local:
                if keyframe.parent_id is not None:
                    keyframe.map.keyframes[keyframe.parent_id].children.discard(keyframe.id)
                keyframe.change_parent(keys[local-1])
            keyframe.first_connection = False
        regions.append(keys)
        all_keys.extend(keys)
    closer = coordinator(atlas)
    for keyframe in all_keys[1:-1]:
        closer._index(keyframe)
    result = closer.process(regions[1][-1])
    assert result["merge"] and not result["loop"]
    assert result["diagnostics"]["merge"]["branch"] == "visual"
    assert regions[0][0].map is regions[1][0].map
    assert len(atlas.current.points) == 100
    assert all(entry.map_id == atlas.current.id for entry in closer.database.entries.values())
