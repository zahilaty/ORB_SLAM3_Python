"""Real-image extraction, rigid stereo triangulation and KB8 tracking."""

from pathlib import Path
import cv2
import numpy as np

from orbslam3.geometry import inverse_pose
from orbslam3.input import Calibration, Sensor
from orbslam3.system import SLAM
from orbslam3.tracking import TrackingState
from orbslam3.vocabulary import Vocabulary

from synthetic import FisheyePlanes


def test_kb8_renderer_bearings_agree_with_independent_opencv_fisheye_model():
    scene = FisheyePlanes()
    camera = scene.camera
    pixels = np.array([[20., 25.], [80., 180.], [160., 120.], [290., 210.]])
    distortion = np.array([camera.k1, camera.k2, camera.k3, camera.k4])
    independent = cv2.fisheye.undistortPoints(pixels[:, None], camera.K, distortion)[:, 0]
    rays = camera.unproject(pixels)
    np.testing.assert_allclose(rays[:, :2], independent, atol=1e-9)
    reprojected, _ = cv2.fisheye.projectPoints(rays[None], np.zeros(3), np.zeros(3), camera.K, distortion)
    np.testing.assert_allclose(camera.project(rays), reprojected[0], atol=1e-9)
    # The right axis is intentionally rotated; the input is not a rectified
    # stereo pair passed through a differently named calibration structure.
    assert abs(scene.T_right_left[0, 2]) > .02
    assert not np.allclose(scene.bearings[25, 20, :2], (pixels[0] - [camera.cx, camera.cy]) / [camera.fx, camera.fy])


def test_unrectified_kb8_images_triangulate_initialize_and_track_metric_motion():
    scene = FisheyePlanes()
    camera = scene.camera
    calibration = Calibration(camera, baseline_fx=camera.fx * .1, depth_threshold=4.,
                              right_camera=camera, T_right_left=scene.T_right_left,
                              overlap_left=(0, scene.width - 1), overlap_right=(0, scene.width - 1))
    vocabulary = Vocabulary.from_text(Path(__file__).resolve().parents[1] / "assets" / "ORBvoc.txt")
    slam = SLAM(Sensor.STEREO, calibration, vocabulary=vocabulary, nfeatures=900, nlevels=4)
    times = np.arange(5) * .08
    results = []
    for time in times:
        result = slam.process(scene.frame(Sensor.STEREO, float(time)))
        results.append(result)
        frame = slam.frame
        assert 0 < frame.n_left < len(frame.features)
        assert frame.features.mono_count == 0
        assert frame.stereo_points.shape == (frame.n_left, 3)
        valid = np.flatnonzero(frame.stereo_matches[:frame.n_left] >= 0)
        assert len(valid) > 100
        for index in valid:
            right_index = frame.stereo_matches[index]
            assert right_index >= frame.n_left
            # ComputeStereoFishEyeMatches performs one-way knn, allowing
            # several left features to select the same right feature. Its
            # reverse slot retains the last accepted left index.
            assert frame.stereo_matches[right_index] == valid[frame.stereo_matches[valid] == right_index][-1]
            point = frame.stereo_points[index]
            assert point[2] > 0.
            np.testing.assert_allclose(frame.depth[index], point[2], atol=1e-12)
            # Geometric rejection permits pyramid-weighted reprojection error;
            # verify the retained 3D points satisfy both rigid camera views.
            right_point = scene.T_right_left[:3, :3] @ point + scene.T_right_left[:3, 3]
            error_left = np.sum((camera.project(point) - frame.features.points[index])**2)
            error_right = np.sum((camera.project(right_point) - frame.features.points[right_index])**2)
            assert error_left <= 5.991 / frame.inverse_variances[frame.features.octaves[index]] + 1e-8
            assert error_right <= 5.991 / frame.inverse_variances[frame.features.octaves[right_index]] + 1e-8
        assert result.state == TrackingState.OK
        assert result.inliers > 80
        assert slam.atlas.current.id == 0
    assert len(slam.atlas.current.points) > 100
    trajectory = slam.trajectory()
    assert len(trajectory) == len(times)
    centers = np.array([inverse_pose(pose)[:3, 3] for _, pose, *_ in trajectory])
    truth = np.array([scene.position(time) for time in times])
    assert np.max(np.linalg.norm(centers - truth, axis=1)) < .025
    # The map must keep independent observations from both optical centers.
    assert any(left >= 0 and right >= 0 for point in slam.atlas.current.points.values()
               for left, right in point.observations.values())
