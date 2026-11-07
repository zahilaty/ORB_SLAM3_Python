"""Synthetic checks of ORB-SLAM3 inertial graphs and analytic factors."""

import copy
import unittest
import numpy as np
from numpy.testing import assert_allclose

from orbslam3.camera import Pinhole, KannalaBrandt8
from orbslam3.imu import Bias, ImuCalibration, Preintegrated, predict_body_state, rotation_exp, rotation_log
from orbslam3.optimization import ReprojectionObservation
from orbslam3.inertial_optimization import (
    BodyState, InertialLink, StatePrior, body_projection_factor, state_prior_factor,
    optimize_inertial_initialization, refine_inertial_biases, refine_gravity_scale,
    initial_gravity_velocities, inertial_initialization_ready,
    full_inertial_bundle_adjustment, local_inertial_bundle_adjustment,
    select_inertial_local_window, optimize_inertial_tracking,
    _state_parameters, _graph_layout, _normal_equations, _solve_schur,
    _visual_factor, _retract,
)


def synthetic_trajectory(count=12):
    calibration = ImuCalibration(np.eye(4), .01, .08, .0005, .002)
    bias = Bias([.01, -.02, .015], [.003, -.002, .004])
    states = {0: BodyState(np.eye(3), np.zeros(3), np.array([.3, -.1, .05]), bias)}
    links = []
    for frame in range(1, count):
        preintegrated = Preintegrated(bias, calibration)
        for sample in range(20):
            time = (frame - 1) * .2 + sample * .01
            acceleration = np.array([1.5 * np.sin(2. * time), .8 * np.cos(1.7 * time), 9.81 + .6 * np.sin(1.1 * time)])
            gyro = np.array([.2 * np.sin(.8 * time), .25 * np.cos(.7 * time), .1 * np.sin(1.2 * time)])
            preintegrated.integrate(acceleration + bias.accelerometer, gyro + bias.gyroscope, .01)
        previous = states[frame - 1]
        rotation, position, velocity = predict_body_state(previous.rotation, previous.position,
                                                          previous.velocity, preintegrated)
        states[frame] = BodyState(rotation, position, velocity, bias)
        links.append(InertialLink(frame - 1, frame, preintegrated))
    return states, links, calibration


def visual_scene(count=4, points_count=30):
    states, links, calibration = synthetic_trajectory(count)
    camera = Pinhole(400., 402., 320., 240.)
    rng = np.random.default_rng(32)
    points = {index: np.array([rng.uniform(-1., 1.5), rng.uniform(-1., 1.), rng.uniform(4., 7.)])
              for index in range(points_count)}
    observations = []
    for identifier, state in states.items():
        for point_id, point in points.items():
            point_camera = state.rotation.T @ (point - state.position)
            left = camera.project(point_camera)
            measurement = np.r_[left, left[0] - 40. / point_camera[2]]
            observations.append(ReprojectionObservation(identifier, point_id, measurement, camera, baseline_fx=40.))
    return states, links, calibration, points, observations


class InertialFactorJacobianTests(unittest.TestCase):
    def test_body_projection_jacobians_all_camera_factor_types(self):
        state = BodyState(rotation_exp([.2, -.1, .05]), [.1, .2, -.1], [.1, .2, .3])
        transform = np.eye(4)
        transform[:3, :3] = rotation_exp([.02, .01, -.03])
        transform[:3, 3] = [.03, -.02, .01]
        calibration = ImuCalibration(transform, .01, .1, .001, .01)
        right = np.eye(4)
        right[0, 3] = -.1
        point = np.array([.5, .3, 4.])
        cameras = [Pinhole(400, 401, 320, 240), KannalaBrandt8(400, 401, 320, 240, .01, -.001)]
        for camera in cameras:
            for dimensions, extrinsic in ((2, None), (3, None), (2, right)):
                observation = ReprojectionObservation(1, 2, np.array([350, 250, 340])[:dimensions], camera,
                                                       baseline_fx=40, camera_from_reference=extrinsic)
                residual, pose_j, point_j, depth = body_projection_factor(state, point, observation, calibration)
                epsilon = 1e-6
                pose_numerical, point_numerical = np.zeros_like(pose_j), np.zeros_like(point_j)
                for axis in range(6):
                    delta = np.eye(6)[axis] * epsilon
                    positive = BodyState(state.rotation @ rotation_exp(delta[:3]),
                                         state.position + state.rotation @ delta[3:], state.velocity)
                    negative = BodyState(state.rotation @ rotation_exp(-delta[:3]),
                                         state.position - state.rotation @ delta[3:], state.velocity)
                    pose_numerical[:, axis] = (body_projection_factor(positive, point, observation, calibration)[0]
                                               - body_projection_factor(negative, point, observation, calibration)[0]) / (2 * epsilon)
                for axis in range(3):
                    delta = np.eye(3)[axis] * epsilon
                    point_numerical[:, axis] = (body_projection_factor(state, point + delta, observation, calibration)[0]
                                                - body_projection_factor(state, point - delta, observation, calibration)[0]) / (2 * epsilon)
                assert_allclose(pose_numerical, pose_j, atol=4e-8)
                assert_allclose(point_numerical, point_j, atol=4e-8)

    def test_state_prior_jacobians(self):
        reference = BodyState(rotation_exp([.1, -.2, .3]), [1, 2, 3], [.1, .2, .3], Bias([.01, .02, .03], [.001, .002, .003]))
        state = BodyState(reference.rotation @ rotation_exp([.02, -.03, .01]), [1.1, 2.1, 3.1], [.2, .3, .4], Bias([.02, .03, .04], [.002, .003, .004]))
        prior = StatePrior(reference, np.eye(15))
        error, jacobians = state_prior_factor(state, prior)
        parameters = _state_parameters({0: state})
        order, offsets, _, dimension = _graph_layout(parameters, set())
        analytical = np.column_stack(jacobians)
        epsilon = 1e-6
        for axis in range(15):
            delta = np.eye(15)[axis] * epsilon
            def evaluate(sign):
                p = _retract(parameters, order, offsets, sign * delta)
                rotation, position = p["pose", 0]
                updated = BodyState(rotation, position, p["velocity", 0], Bias(p["accel", 0], p["gyro", 0]))
                return state_prior_factor(updated, prior)[0]
            assert_allclose((evaluate(1) - evaluate(-1)) / (2 * epsilon), analytical[:, axis], atol=1e-9)

    def test_sparse_landmark_schur_equals_direct_normal_solve(self):
        states, links, calibration, points, observations = visual_scene(2, 10)
        parameters = _state_parameters(states, points)
        fixed = {key for key in parameters if key[0] in ("velocity", "gyro", "accel")}
        factors = [_visual_factor(observation, calibration) for observation in observations]
        _, offsets, state_dimension, dimension = _graph_layout(parameters, fixed)
        normal = _normal_equations(parameters, factors, offsets, dimension)
        increment, rhs = _solve_schur(normal, .1, state_dimension)
        direct = np.linalg.solve(normal[0].toarray() + np.eye(dimension) * .1, rhs)
        assert_allclose(increment, direct, atol=1e-10)


class InertialInitializationTests(unittest.TestCase):
    def test_gravity_scale_initialization_and_fixed_variable_selection(self):
        states, links, calibration = synthetic_trajectory(14)
        expected_gravity = rotation_exp([.08, -.05, 0.])
        expected_scale = 1.35
        visual = {identifier: BodyState(expected_gravity @ state.rotation,
                                        expected_gravity @ state.position / expected_scale,
                                        expected_gravity @ state.velocity / expected_scale, state.bias)
                  for identifier, state in states.items()}
        result = optimize_inertial_initialization(visual, links, fixed_velocity=True)
        self.assertLess(result.final_cost, result.initial_cost * 1e-6)
        self.assertAlmostEqual(result.scale, expected_scale, places=4)
        assert_allclose(result.gravity_rotation @ [0, 0, -1], expected_gravity @ [0, 0, -1], atol=1e-5)
        for identifier, before in visual.items():
            assert_allclose(result.states[identifier].position, before.position)
            assert_allclose(result.states[identifier].velocity, before.velocity)
            assert_allclose(result.states[identifier].bias.vector(), before.bias.vector())
        stereo = optimize_inertial_initialization(states, links, monocular=False, fixed_velocity=True)
        self.assertEqual(stereo.scale, 1.)
        self.assertLess(stereo.final_cost, 1e-15)

    def test_joint_velocities_bias_and_scale_optimization(self):
        states, links, calibration = synthetic_trajectory(18)
        expected_gravity = rotation_exp([.04, -.06, 0.])
        expected_scale = 1.2
        visual = {identifier: BodyState(expected_gravity @ state.rotation,
                                        expected_gravity @ state.position / expected_scale,
                                        expected_gravity @ state.velocity / expected_scale + [.02, -.01, .03], Bias())
                  for identifier, state in states.items()}
        result = optimize_inertial_initialization(visual, links)
        self.assertLess(result.final_cost, result.initial_cost * 1e-7)
        self.assertAlmostEqual(result.scale, expected_scale, places=3)
        assert_allclose(result.states[0].bias.vector(), states[0].bias.vector(), atol=2e-4)
        assert_allclose(result.gravity_rotation @ [0, 0, -1], expected_gravity @ [0, 0, -1], atol=2e-4)

    def test_bias_and_gravity_scale_refinement(self):
        states, links, calibration = synthetic_trajectory(15)
        perturbed = copy.deepcopy(states)
        for state in perturbed.values():
            state.velocity += [.02, -.01, .03]
            state.bias = Bias()
        result = refine_inertial_biases(perturbed, links)
        self.assertLess(result.final_cost, result.initial_cost * 1e-6)
        assert_allclose(result.states[0].bias.vector(), states[0].bias.vector(), atol=2e-4)
        gravity = rotation_exp([.03, -.02, 0.])
        scale = 1.15
        visual = {identifier: BodyState(gravity @ state.rotation, gravity @ state.position / scale,
                                        gravity @ state.velocity / scale, state.bias)
                  for identifier, state in states.items()}
        refined = refine_gravity_scale(visual, links)
        self.assertLess(refined.final_cost, refined.initial_cost * 1e-8)
        self.assertAlmostEqual(refined.scale, scale, places=5)

    def test_gravity_velocity_seed_and_initialization_gate(self):
        states, links, calibration = synthetic_trajectory(12)
        updated, gravity = initial_gravity_velocities(states, links)
        self.assertTrue(np.isfinite(gravity).all())
        for link in links:
            # Previous velocity may be overwritten by the following link, as in C++.
            if link.current == len(states) - 1:
                expected = (states[link.current].position - states[link.previous].position) / link.preintegrated.dt
                assert_allclose(updated[link.current].velocity, expected)
                assert_allclose(updated[link.previous].velocity, expected)
        timestamps = {identifier: identifier * .2 for identifier in states}
        self.assertTrue(inertial_initialization_ready(states, timestamps, True))
        self.assertFalse(inertial_initialization_ready(dict(list(states.items())[:9]), timestamps, True))


class VisualInertialGraphTests(unittest.TestCase):
    def test_full_vi_bundle_adjustment_and_local_marker_early_return(self):
        states, links, calibration, points, observations = visual_scene(4, 25)
        rng = np.random.default_rng(20)
        perturbed = copy.deepcopy(states)
        for identifier, state in perturbed.items():
            if identifier:
                state.rotation = state.rotation @ rotation_exp(rng.normal(0, .003, 3))
                state.position += rng.normal(0, .02, 3)
                state.velocity += rng.normal(0, .02, 3)
        noisy_points = {identifier: point + rng.normal(0, .03, 3) for identifier, point in points.items()}
        result = full_inertial_bundle_adjustment(perturbed, noisy_points, observations, links, calibration,
                                                 fixed_states={0}, iterations=20)
        self.assertLess(result.final_cost, result.initial_cost * 1e-6)
        assert_allclose(result.states[3].position, states[3].position, atol=1e-4)
        assert_allclose(result.states[0].position, states[0].position)
        for identifier, point in points.items():
            assert_allclose(result.points[identifier], point, atol=1e-4)
        self.assertEqual(result.inliers, len(observations))
        declined = full_inertial_bundle_adjustment(states, points, observations, links, calibration,
                                                    fix_local=True, local_marks={0: 3, 1: 3})
        self.assertFalse(declined.accepted)

    def test_shared_bias_full_ba_and_local_ba(self):
        states, links, calibration, points, observations = visual_scene(4, 20)
        points = {identifier: point + [.005, -.003, .002] for identifier, point in points.items()}
        result = full_inertial_bundle_adjustment(states, points, observations, links, calibration,
                                                 fixed_states={0}, initialize_bias=True, iterations=10)
        self.assertLess(result.final_cost, result.initial_cost * 1e-6)
        assert_allclose(result.states[0].bias.vector(), result.states[3].bias.vector())
        local = local_inertial_bundle_adjustment(states, points, observations, links[::-1], calibration,
                                                  optimizable_ids={1, 2, 3}, fixed_ids={0})
        self.assertTrue(local.accepted)
        self.assertLess(local.final_cost, local.initial_cost * 1e-6)
        self.assertEqual(local.inliers, len(observations))

    def test_local_temporal_and_fixed_window_selection(self):
        predecessors = {identifier: identifier - 1 if identifier else None for identifier in range(15)}
        observations = {0: [14, 13, 2], 1: [10, 9, 1], 2: [0, 1], 3: [14, 3]}
        temporal, fixed, points = select_inertial_local_window(14, predecessors, observations)
        self.assertEqual(temporal, tuple(range(14, 4, -1)))
        self.assertEqual(fixed, (4, 2, 1, 3))
        self.assertEqual(points, (0, 1, 3))

    def test_last_keyframe_and_last_frame_tracking_generate_priors(self):
        states, links, calibration, points, observations = visual_scene(3, 35)
        current = copy.deepcopy(states[1])
        current.position += [.03, -.02, .01]
        current.velocity += [.02, .01, -.01]
        current.rotation = current.rotation @ rotation_exp([.003, -.002, .001])
        frame1 = [observation for observation in observations if observation.pose_id == 1]
        result1 = optimize_inertial_tracking(states[0], current, links[0].preintegrated, points, frame1, calibration)
        assert_allclose(result1.states[1].position, states[1].position, atol=1e-7)
        self.assertEqual(result1.inliers, len(frame1))
        self.assertEqual(result1.prior.information.shape, (15, 15))
        self.assertGreaterEqual(np.linalg.eigvalsh(result1.prior.information).min(), -1e-5)
        frame2 = [observation for observation in observations if observation.pose_id == 2]
        next_state = copy.deepcopy(states[2])
        next_state.position += [-.02, .01, -.02]
        result2 = optimize_inertial_tracking(result1.states[1], next_state, links[1].preintegrated,
                                             points, frame2, calibration, result1.prior, previous_id=1, current_id=2)
        assert_allclose(result2.states[2].position, states[2].position, atol=1e-6)
        self.assertEqual(result2.inliers, len(frame2))
        assert_allclose(result2.prior.information, result2.prior.information.T, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
