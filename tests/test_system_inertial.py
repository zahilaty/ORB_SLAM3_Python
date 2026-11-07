"""Image-to-map visual-inertial integration on generated, physical scenes.

No feature/matcher/mapper/optimizer methods are mocked. The normal trained
ORB vocabulary is used. Ground truth comes from analytic renderer motion,
not from another SLAM result or an implementation-derived pose fixture.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from orbslam3.imu import ImuCalibration
from orbslam3.input import Calibration, Sensor
from orbslam3.system import SLAM
from orbslam3.tracking import TrackingState
from orbslam3.vocabulary import Vocabulary

from synthetic import TexturedPlanes


@pytest.fixture(scope="module")
def trained_vocabulary():
    path = Path(__file__).resolve().parents[1] / "assets" / "ORBvoc.txt"
    return Vocabulary.from_text(path)


def test_renderer_imu_matches_analytic_camera_acceleration_and_stereo_geometry():
    scene = TexturedPlanes()
    epsilon, time = 1e-4, .8
    numerical_velocity = (scene.position(time + epsilon) - scene.position(time - epsilon)) / (2 * epsilon)
    numerical_acceleration = (scene.position(time + epsilon) - 2 * scene.position(time)
                              + scene.position(time - epsilon)) / epsilon**2
    np.testing.assert_allclose(numerical_velocity, scene.velocity(time), atol=3e-9)
    np.testing.assert_allclose(numerical_acceleration, scene.acceleration(time), atol=1e-8)
    imu = scene.imu(time, time)[0]
    np.testing.assert_allclose(imu.acceleration - [0, 0, 9.81], numerical_acceleration, atol=1e-8)
    np.testing.assert_array_equal(imu.angular_velocity, [0, 0, 0])
    left, depth = scene.render(time)
    right, _ = scene.render(time, .1)
    # Identical world surface points project with the calibrated disparity.
    # Compare texture values at subpixel right coordinates away from edges;
    # image resampling permits a few grey levels, not a changed geometry.
    yy, xx = np.indices(left.shape)
    right_x = xx - scene.camera.fx * .1 / depth
    import cv2
    sampled = cv2.remap(right, right_x.astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR)
    interior = (xx > 25) & (xx < scene.width - 25)
    interior &= np.abs(depth - np.roll(depth, 15, axis=1)) < 1e-12
    interior &= np.abs(depth - np.roll(depth, -15, axis=1)) < 1e-12
    assert np.median(np.abs(left[interior].astype(float) - sampled[interior].astype(float))) < 10


@pytest.mark.parametrize("sensor", [Sensor.MONOCULAR_IMU, Sensor.STEREO_IMU, Sensor.RGBD_IMU])
def test_images_track_initialize_imu_and_continue_with_metric_state(sensor, trained_vocabulary):
    scene = TexturedPlanes()
    imu = ImuCalibration.from_noise_density(np.eye(4), .0001, .001, .000001, .00001, 200.)
    calibration = Calibration(scene.camera, baseline_fx=scene.camera.fx * .1,
                              depth_threshold=4., frames_per_second=10 / 3., imu=imu)
    # This is the original sensor setting that bypasses the initial stereo
    # acceleration-difference gate. Ten-KF/time gates and all inertial graphs
    # still execute with their normal iteration counts and original priors.
    slam = SLAM(sensor, calibration, vocabulary=trained_vocabulary, nfeatures=750,
                nlevels=4, fast_imu_init=True)
    initialized_at = None
    initialization_events = []
    results = []
    for index in range(15):
        timestamp = .3 * index
        result = slam.process(scene.frame(sensor, timestamp, None if index == 0 else .3 * (index - 1)))
        results.append(result)
        assert len(slam.atlas.maps) == 1, (sensor, index, result.diagnostics)
        if result.pose is not None:
            assert np.isfinite(result.pose).all()
            np.testing.assert_allclose(result.pose[:3, :3].T @ result.pose[:3, :3], np.eye(3), atol=1e-8)
        for keyframe in slam.atlas.current.keyframes.values():
            if keyframe.body_state is not None:
                # Detect stale cached body poses after visual local BA, before
                # they corrupt gravity/scale initialization at ten keyframes.
                body = keyframe.body_state
                world_body = np.eye(4)
                world_body[:3, :3], world_body[:3, 3] = body.rotation, body.position
                world_camera = world_body @ keyframe.imu_calibration.T_body_camera
                np.testing.assert_allclose(keyframe.pose @ world_camera, np.eye(4), atol=1e-8)
        for diagnostics in result.diagnostics.get("mapping", ()):
            event = diagnostics.get("inertial_initialization")
            if event is not None and event.changed:
                initialization_events.append(event)
        if slam.atlas.current.imu_initialized and initialized_at is None:
            initialized_at = index
    assert initialized_at is not None and initialized_at < len(results) - 2
    assert initialization_events and all(event.initialized for event in initialization_events)
    assert slam.mapper.inertial.initialization_count >= 1
    assert results[-1].state == TrackingState.OK and results[-1].inliers >= 15
    assert len(slam.atlas.current.keyframes) >= 10
    assert any(keyframe.state_prior is not None for keyframe in [slam.frame, slam.tracker.last])
    assert slam.frame.frame_preintegrated.dt == pytest.approx(.3, abs=1e-12)
    assert np.isfinite(slam.frame.body_state.bias.vector()).all()
    assert np.isfinite(slam.frame.body_state.velocity).all()
    trajectory = slam.trajectory()
    assert len(trajectory) >= 10
    times = np.array([timestamp for timestamp, *_ in trajectory])
    assert np.all(np.diff(times) > 0)
    poses = np.array([pose for _, pose, *_ in trajectory])
    centers = -np.einsum("nji,nj->ni", poses[:, :3, :3], poses[:, :3, 3])
    # Align only the first camera frame; no scale/trajectory fitting is allowed
    # after IMU initialization because all three modes must recover metres.
    estimated = (centers - centers[0]) @ poses[0, :3, :3].T
    expected = np.array([scene.position(timestamp) - scene.position(times[0]) for timestamp in times])
    translation_error = np.linalg.norm(estimated - expected, axis=1)
    assert np.max(translation_error) < .08, (sensor, translation_error, estimated[-1], expected[-1])
    relative_rotations = poses[0, :3, :3] @ np.swapaxes(poses[:, :3, :3], 1, 2)
    angle_error = np.linalg.norm(Rotation.from_matrix(relative_rotations).as_rotvec(), axis=1)
    assert np.max(angle_error) < .03
