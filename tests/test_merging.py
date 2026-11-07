"""Synthetic two-map welding, transfer, tree and factor-window checks."""

import unittest
import numpy as np
from numpy.testing import assert_allclose

from orbslam3.camera import Pinhole
from orbslam3.features import FeatureSet
from orbslam3.geometry import Sim3, inverse_pose, transform_points
from orbslam3.imu import Bias, ImuCalibration, Preintegrated, rotation_exp
from orbslam3.inertial_mapping import body_state_from_camera, camera_pose_from_body
from orbslam3.input import Sensor
from orbslam3.map import Atlas, Frame
from orbslam3.merging import (
    merge_maps, merging_essential_graph_edges, visual_welding_window,
    inertial_welding_window, _replace_across_maps,
)
from orbslam3.vocabulary import InvertedDatabase


def two_maps(count=3, point_count=15, scale=1.3, inertial=False, stereo=False):
    atlas = Atlas(inertial=inertial)
    camera = Pinhole(400., 400., 320., 240.)
    points = np.array([[x, y, 5. + .1 * (index % 3)]
                       for index, (x, y) in enumerate((x, y) for x in np.linspace(-.8, .8, 5)
                                                    for y in np.linspace(-.5, .5, 3))])[:point_count]
    point_count = len(points)
    descriptors = np.random.default_rng(99).integers(0, 256, (point_count, 32), dtype=np.uint8)
    transform = Sim3(rotation_exp([0., 0., .08]), np.array([.3, -.1, 0.]), scale)
    calibration = ImuCalibration(np.eye(4), .01, .1, .001, .005)
    groups = []
    for group in range(2):
        if group:
            atlas.create_map()
        map_ = atlas.current
        keyframes = []
        for index in range(count):
            pose = np.eye(4)
            pose[0, 3] = -.1 * index
            camera_points = transform_points(pose, points)
            pixels = camera.project(camera_points)
            features = FeatureSet(pixels, descriptors.copy(), np.zeros(point_count, int),
                                  np.zeros(point_count), np.ones(point_count), np.full(point_count, 31.), point_count)
            frame = Frame(atlas.next_keyframe_id, index * .2, features, camera, 640, 480,
                          np.array([1., 1.2, 1.44]), np.array([1., 1 / 1.44, 1 / 2.0736]),
                          pose=pose, baseline_fx=40., depth_threshold=10.)
            if stereo:
                frame.right_u = pixels[:, 0] - 40. / camera_points[:, 2]
            if inertial:
                frame.imu_calibration = calibration
                frame.body_state = body_state_from_camera(pose, calibration, np.array([.5, 0., 0.]), Bias())
            keyframe = atlas.add_keyframe(frame)
            keyframe.first_connection = False
            keyframe.bow, keyframe.feature_nodes = {0: 1.}, {0: list(range(point_count))}
            if keyframes:
                keyframe.previous_id, keyframes[-1].next_id = keyframes[-1].id, keyframe.id
                keyframe.parent_id = keyframes[-1].id
                keyframes[-1].children.add(keyframe.id)
            if inertial:
                keyframe.preintegrated = None
                if index:
                    keyframe.preintegrated = Preintegrated(Bias(), calibration)
                    for _ in range(20):
                        keyframe.preintegrated.integrate([0, 0, 9.81], [0, 0, 0], .01)
            keyframes.append(keyframe)
        for index, position in enumerate(points):
            point = atlas.add_point(position, keyframes[0])
            for keyframe in keyframes:
                point.add_observation(keyframe, index)
                keyframe.map_points[index] = point
            point.compute_descriptor()
            point.update_normal_and_depth()
        for keyframe in keyframes:
            keyframe.update_connections()
        if group == 0:
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3], transform_matrix[:3, 3] = transform.rotation, transform.translation
            old_states = {keyframe.id: keyframe.body_state for keyframe in keyframes} if inertial else {}
            map_.apply_scaled_rotation(transform_matrix, transform.scale, scale_velocity=True)
            for keyframe in keyframes:
                if inertial:
                    keyframe.body_state = body_state_from_camera(keyframe.pose, calibration,
                                                                 transform.scale * transform.rotation @ old_states[keyframe.id].velocity,
                                                                 old_states[keyframe.id].bias)
        if inertial:
            map_.imu_initialized = map_.inertial_ba1 = map_.inertial_ba2 = True
            for keyframe in keyframes:
                keyframe.inertial = True
        groups.append((map_, keyframes))
    target, source = groups[0], groups[1]
    matched, current = target[1][-1], source[1][-1]
    similarity = Sim3(current.pose[:3, :3], current.pose[:3, 3]) @ transform.inverse()
    return atlas, current, matched, similarity, transform, points


def assert_integrity(test, atlas, map_):
    for identifier, keyframe in map_.keyframes.items():
        test.assertIs(keyframe.map, map_)
        test.assertIs(atlas.all_keyframes[identifier], keyframe)
        seen, current = set(), keyframe
        while current.parent_id is not None:
            test.assertNotIn(current.id, seen)
            seen.add(current.id)
            current = atlas.all_keyframes[current.parent_id]
        for index, point in enumerate(keyframe.map_points):
            if point is not None:
                test.assertFalse(point.bad)
                test.assertIs(point.map, map_)
                test.assertIn(index, point.observations[keyframe.id])
    for point in map_.points.values():
        for identifier, indices in point.observations.items():
            for index in indices:
                if index >= 0:
                    test.assertIs(atlas.all_keyframes[identifier].map_points[index], point)


class VisualMergingTests(unittest.TestCase):
    def test_monocular_welding_transforms_fuses_and_rebuilds_tree_database(self):
        atlas, current, matched, similarity, transform, points = two_maps()
        source, target = current.map, matched.map
        source_id = source.id
        database = InvertedDatabase(None)
        for keyframe in atlas.all_keyframes.values():
            database.add(keyframe.id, keyframe.bow, keyframe.map.id)
        result = merge_maps(atlas, current, matched, similarity, Sensor.MONOCULAR, database)
        self.assertTrue(result["merged"])
        self.assertEqual(result["branch"], "visual")
        self.assertIs(atlas.current, target)
        self.assertTrue(source.bad)
        self.assertEqual(target.id, source_id)
        self.assertEqual(set(atlas.maps), {source_id})
        self.assertEqual(len(target.keyframes), 6)
        self.assertEqual(len(target.points), 15)
        self.assertEqual(result["fused_points"], 15)
        self.assertEqual(current.parent_id, matched.id)
        self.assertIn(matched.id, current.merge_edges)
        self.assertIn(current.id, matched.merge_edges)
        self.assertTrue(current.not_erase and matched.not_erase)
        self.assertEqual({entry.map_id for entry in database.entries.values()}, {source_id})
        assert_allclose(result["world_transform"].rotation, transform.rotation, atol=1e-15)
        self.assertAlmostEqual(result["scale"], transform.scale)
        for keyframe in target.keyframes.values():
            for index, point in enumerate(keyframe.map_points):
                if point is not None:
                    assert_allclose(keyframe.camera.project(transform_points(keyframe.pose, point.position)),
                                    keyframe.features.points[index], atol=1e-7)
        assert_integrity(self, atlas, target)

    def test_depth_supported_welding_preserves_metric_stereo_factors(self):
        atlas, current, matched, similarity, transform, points = two_maps(scale=1., stereo=True)
        result = merge_maps(atlas, current, matched, similarity, Sensor.RGBD)
        self.assertTrue(result["merged"])
        self.assertLess(result["welding_ba"]["final_cost"], 1e-12)
        self.assertEqual(len(atlas.current.points), 15)
        assert_integrity(self, atlas, atlas.current)

    def test_cross_map_point_replacement_keeps_observers_outside_local_transfer(self):
        atlas, current, matched, similarity, transform, points = two_maps()
        old, retained = current.map_points[0], matched.map_points[0]
        old_observers, target_observers = set(old.observations), set(retained.observations)
        _replace_across_maps(old, retained, atlas.all_keyframes)
        self.assertTrue(old.bad)
        self.assertEqual(set(retained.observations), old_observers | target_observers)
        for identifier in old_observers:
            self.assertIs(atlas.all_keyframes[identifier].map_points[0], retained)
        self.assertEqual(retained.observation_count, 6)

    def test_merging_essential_graph_literal_good_bad_pose_measurement(self):
        atlas, current, matched, similarity, transform, points = two_maps(count=2)
        # Include one target fixed pose and the corrected current fixed pose,
        # plus one uncorrected old-map pose. This isolates source pose masks.
        remaining = current.map.keyframes[current.previous_id]
        before = {current.id: current.pose.copy(), remaining.id: remaining.pose.copy()}
        current.pose = similarity.camera_pose()
        current.parent_id = matched.id
        remaining.parent_id = current.id
        initial, edges = merging_essential_graph_edges([matched], [current], [remaining], before)
        tree = {(edge.first, edge.second): edge for edge in edges if edge.kind == "spanning_tree"}
        expected = Sim3(matched.pose[:3, :3], matched.pose[:3, 3]) @ Sim3(before[current.id][:3, :3], before[current.id][:3, 3]).inverse()
        edge = tree[current.id, matched.id]
        assert_allclose(edge.measurement.translation, expected.translation)
        self.assertIn((remaining.id, current.id), tree)

    def test_noncritical_visual_region_uses_mono_transform_or_depth_pose_graph(self):
        for sensor in (Sensor.MONOCULAR, Sensor.RGBD):
            atlas, current, matched, similarity, transform, points = two_maps(scale=1.3 if sensor.monocular else 1.)
            source = current.map
            original_points = list(source.points.values())
            # Distinct landmarks per old-map keyframe leave only current in the
            # welding neighborhood while its spanning-tree predecessors still
            # require the source's noncritical-region correction procedure.
            for keyframe in list(source.keyframes.values()):
                for index, old in enumerate(original_points):
                    point = atlas.add_point(old.position, keyframe)
                    point.add_observation(keyframe, index)
                    point.descriptor = old.descriptor.copy()
                    keyframe.map_points[index] = point
                    point.update_normal_and_depth()
                keyframe.covisibility.clear()
                keyframe.ordered_covisibility.clear()
            for old in original_points:
                source.points.pop(old.id)
                old.observations.clear()
                old.bad = True
            result = merge_maps(atlas, current, matched, similarity, sensor)
            self.assertTrue(result["merged"])
            self.assertEqual(len(atlas.current.keyframes), 6)
            self.assertEqual(len(atlas.current.points), 45)
            self.assertEqual(result["essential_graph"] is None, sensor.monocular)
            assert_integrity(self, atlas, atlas.current)
            for keyframe in atlas.current.keyframes.values():
                for index, point in enumerate(keyframe.map_points):
                    if point is not None:
                        assert_allclose(keyframe.camera.project(transform_points(keyframe.pose, point.position)),
                                        keyframe.features.points[index], atol=1e-5)


class InertialMergingTests(unittest.TestCase):
    def test_inertial_gates_precede_map_mutation(self):
        atlas, current, matched, similarity, transform, points = two_maps(scale=1., inertial=True)
        current.map.imu_initialized = False
        result = merge_maps(atlas, current, matched, similarity, Sensor.STEREO_IMU)
        self.assertFalse(result["merged"])
        self.assertEqual(result["reason"], "imu_not_initialized")
        current.map.imu_initialized = True
        result = merge_maps(atlas, current, matched, similarity, Sensor.STEREO_IMU,
                            world_transform=Sim3(np.eye(3), np.zeros(3), 1.2))
        self.assertFalse(result["merged"])
        self.assertEqual(result["reason"], "inertial_scale_gate")
        self.assertEqual(len(atlas.maps), 2)

    def test_small_inertial_map_retains_empty_donor_and_skips_welding_ba(self):
        atlas, current, matched, similarity, transform, points = two_maps(scale=1., inertial=True, stereo=True)
        active, donor = current.map, matched.map
        result = merge_maps(atlas, current, matched, similarity, Sensor.STEREO_IMU)
        self.assertTrue(result["merged"])
        self.assertEqual(result["branch"], "inertial")
        self.assertIs(atlas.current, active)
        self.assertEqual(len(active.keyframes), 6)
        self.assertEqual(len(active.points), 15)
        self.assertEqual(len(donor.keyframes), 0)
        self.assertIn(donor.id, atlas.maps)
        self.assertTrue(result["retained_empty_map"])
        self.assertIsNone(result["welding_ba"])
        self.assertFalse(current.merge_edges)
        self.assertEqual(matched.parent_id, current.id)
        for keyframe in active.keyframes.values():
            assert_allclose(camera_pose_from_body(keyframe.body_state, keyframe.imu_calibration), keyframe.pose, atol=1e-12)
        assert_integrity(self, atlas, active)

    def test_unfinished_inertial_map_runs_bias_refinement_and_sets_ba_flags(self):
        atlas, current, matched, similarity, transform, points = two_maps(scale=1., inertial=True)
        current.map.inertial_ba2 = current.map.inertial_ba1 = False
        result = merge_maps(atlas, current, matched, similarity, Sensor.MONOCULAR_IMU)
        self.assertTrue(result["merged"])
        self.assertTrue(atlas.current.imu_initialized and atlas.current.inertial_ba1 and atlas.current.inertial_ba2)
        self.assertTrue(np.isfinite(result["bias"].vector()).all())

    def test_inertial_merge_welding_window_and_eight_iteration_ba(self):
        atlas, current, matched, similarity, transform, points = two_maps(count=12, scale=1., inertial=True, stereo=True)
        result = merge_maps(atlas, current, matched, similarity, Sensor.STEREO_IMU)
        self.assertTrue(result["merged"])
        self.assertIsNotNone(result["welding_ba"])
        self.assertLessEqual(result["welding_ba"]["iterations"], 8)
        self.assertLess(result["welding_ba"]["final_cost"], 1e-10)
        self.assertEqual(result["welding_ba"]["fixed_ids"], (8,))
        self.assertIn(17, result["welding_ba"]["covisible_ids"])
        self.assertEqual(len(atlas.current.points), 15)
        assert_integrity(self, atlas, atlas.current)

    def test_uninitialized_matched_keyframes_keep_visual_only_inertial_state(self):
        atlas, current, matched, similarity, transform, points = two_maps(count=12, scale=1., inertial=True, stereo=True)
        donor = matched.map
        donor.imu_initialized = donor.inertial_ba1 = donor.inertial_ba2 = False
        for keyframe in donor.keyframes.values():
            keyframe.inertial = False
        old_states = {identifier: (keyframe.body_state.velocity.copy(), keyframe.body_state.bias.vector().copy())
                      for identifier, keyframe in donor.keyframes.items()}
        result = merge_maps(atlas, current, matched, similarity, Sensor.STEREO_IMU)
        self.assertTrue(result["merged"])
        for identifier, (velocity, bias) in old_states.items():
            keyframe = atlas.all_keyframes[identifier]
            self.assertFalse(keyframe.inertial)
            assert_allclose(keyframe.body_state.velocity, velocity)
            assert_allclose(keyframe.body_state.bias.vector(), bias)


if __name__ == "__main__":
    unittest.main()
