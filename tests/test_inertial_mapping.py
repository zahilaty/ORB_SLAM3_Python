"""Inertial initialization scheduling and map-coordinate correction checks."""

import copy
from unittest import TestCase, main
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose

from orbslam3.camera import Pinhole
from orbslam3.features import FeatureSet
from orbslam3.geometry import inverse_pose
from orbslam3.imu import Bias, ImuCalibration, rotation_exp
from orbslam3.inertial_optimization import BodyState
from orbslam3.inertial_mapping import (
    InertialMapper, InertialMappingUpdate, apply_inertial_scaled_rotation,
    camera_pose_from_body, body_state_from_camera,
)
from orbslam3.map import Map, KeyFrame, MapPoint

from test_inertial_optimization import visual_scene


def keyframe_map(count=12, point_count=12):
    states, links, calibration, points, observations = visual_scene(count, point_count)
    map_ = Map(1, 0, inertial=True)
    camera = Pinhole(400, 402, 320, 240)
    for identifier, state in states.items():
        frame_observations = [observation for observation in observations if observation.pose_id == identifier]
        pixels = np.array([observation.measurement[:2] for observation in frame_observations])
        features = FeatureSet(pixels, np.zeros((point_count, 32), np.uint8), np.zeros(point_count, int),
                              np.zeros(point_count), np.ones(point_count), np.ones(point_count) * 31, point_count)
        keyframe = KeyFrame(identifier, identifier * .2, features, camera, 640, 480,
                            np.array([1., 1.2]), np.array([1., 1. / 1.44]), baseline_fx=40.,
                            pose=camera_pose_from_body(state, calibration), map=map_,
                            right_u=np.array([observation.measurement[2] for observation in frame_observations]))
        keyframe.body_state, keyframe.imu_calibration = copy.deepcopy(state), calibration
        keyframe.preintegrated = links[identifier - 1].preintegrated.copy() if identifier else None
        keyframe.previous_id = identifier - 1 if identifier else None
        keyframe.next_id = identifier + 1 if identifier + 1 < count else None
        if identifier:
            keyframe.parent_id = identifier - 1
        map_.keyframes[identifier] = keyframe
    map_.origins = [0]
    for identifier, point in points.items():
        landmark = MapPoint(identifier, point, 0, map_)
        map_.points[identifier] = landmark
        for keyframe in map_.keyframes.values():
            keyframe.map_points[identifier] = landmark
            landmark.add_observation(keyframe, identifier)
        landmark.update_normal_and_depth()
    return map_


class InertialMapCorrectionTests(TestCase):
    def test_camera_body_pose_roundtrip_and_nonzero_lever_arm_scaling(self):
        map_ = keyframe_map(3, 5)
        transform = np.eye(4)
        transform[:3, :3] = rotation_exp([.03, -.02, .01])
        transform[:3, 3] = [.2, -.1, .05]
        calibration = ImuCalibration(transform, .01, .1, .001, .01)
        for keyframe in map_.keyframes.values():
            keyframe.imu_calibration = calibration
            keyframe.body_state = body_state_from_camera(keyframe.pose, calibration,
                                                          keyframe.body_state.velocity, keyframe.body_state.bias)
            assert_allclose(camera_pose_from_body(keyframe.body_state, calibration), keyframe.pose, atol=1e-15)
        old = {identifier: (keyframe.center.copy(), copy.deepcopy(keyframe.body_state))
               for identifier, keyframe in map_.keyframes.items()}
        rotation, scale = rotation_exp([.1, -.2, .05]), 1.5
        old_points = {identifier: point.position.copy() for identifier, point in map_.points.items()}
        apply_inertial_scaled_rotation(map_, rotation, scale)
        for identifier, keyframe in map_.keyframes.items():
            center, state = old[identifier]
            assert_allclose(keyframe.center, scale * rotation @ center, atol=1e-14)
            assert_allclose(keyframe.body_state.velocity, scale * rotation @ state.velocity)
            assert_allclose(camera_pose_from_body(keyframe.body_state, calibration), keyframe.pose, atol=1e-14)
            # A lever arm is rigid and must not be incorrectly scaled with world.
            self.assertGreater(np.linalg.norm(keyframe.body_state.position - scale * rotation @ state.position), .05)
        for identifier, point in map_.points.items():
            assert_allclose(point.position, scale * rotation @ old_points[identifier])

    def test_real_initialization_updates_map_and_full_inertial_ba(self):
        map_ = keyframe_map(12, 15)
        mapper = InertialMapper(monocular=False)
        result = mapper.initialize(map_.keyframes[11])
        self.assertTrue(result.initialized)
        self.assertTrue(result.changed)
        self.assertTrue(map_.imu_initialized)
        self.assertEqual(mapper.initialization_count, 1)
        self.assertAlmostEqual(mapper.motion_time, 2.2)
        self.assertEqual(len(result.updated_keyframes), 12)
        for keyframe in map_.keyframes.values():
            self.assertTrue(np.isfinite(keyframe.pose).all())
            assert_allclose(camera_pose_from_body(keyframe.body_state, keyframe.imu_calibration),
                            keyframe.pose, atol=1e-12)
        for point in map_.points.values():
            self.assertTrue(np.isfinite(point.position).all())
        self.assertFalse(mapper.initialize(map_.keyframes[11]).changed)


class InertialSchedulingTests(TestCase):
    def test_initialization_gate_motion_accumulation_and_low_motion_reset(self):
        map_ = keyframe_map(9, 2)
        mapper = InertialMapper(monocular=True)
        self.assertFalse(mapper.initialize(map_.keyframes[8]).initialized)
        map_.imu_initialized = True
        mapper._ensure_map(map_.keyframes[8])
        mapper.motion_time = 3.
        previous_time = mapper.motion_time
        result = mapper.before_local_ba(map_.keyframes[8])
        self.assertFalse(result.reset_required)
        self.assertGreater(mapper.motion_time, previous_time)
        # The same keyframe cannot count its motion twice through this API.
        mapper.before_local_ba(map_.keyframes[8])
        self.assertAlmostEqual(mapper.motion_time, previous_time + .2)
        for keyframe in map_.keyframes.values():
            keyframe.pose = np.eye(4)
        self.assertTrue(mapper.before_local_ba(map_.keyframes[7]).reset_required)

    def test_refinement_thresholds_priors_and_outer_scale_window_guard(self):
        map_ = keyframe_map(12, 2)
        map_.imu_initialized = True
        current = map_.keyframes[11]
        mapper = InertialMapper(monocular=True)
        mapper._ensure_map(current)
        with patch.object(mapper, "_initialize_with_priors", return_value=InertialMappingUpdate(initialized=True)) as initialize:
            mapper.motion_time = 5.
            mapper.refine(current)
            initialize.assert_not_called()
            mapper.motion_time = 5.1
            result = mapper.refine(current)
            initialize.assert_called_once_with(current, 1., 1e5)
            self.assertTrue(map_.inertial_ba1)
            self.assertEqual(result.event, "inertial_ba1")
            mapper.motion_time = 15.1
            mapper.refine(current)
            self.assertTrue(map_.inertial_ba2)
            self.assertEqual(initialize.call_args.args, (current, 0., 0.))
        with patch("orbslam3.inertial_mapping.refine_gravity_scale") as refinement:
            mapper.motion_time = 55.1
            mapper.refine(current)
            refinement.assert_not_called()
            mapper.motion_time = 25.1
            refinement.return_value.scale = 1.
            mapper.refine(current, tracking_ok=False)
            refinement.assert_not_called()
            mapper.refine(current)
            refinement.assert_called_once()


if __name__ == "__main__":
    main()
