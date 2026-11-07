"""Actual image-to-map integration on own generated multi-depth scene."""

from pathlib import Path
import numpy as np
import pytest
from synthetic import TexturedPlanes
from orbslam3 import SLAM, Sensor, Calibration
from orbslam3.geometry import inverse_pose
from orbslam3.tracking import TrackingState
from orbslam3.export import save_trajectory, save_points
from orbslam3.visualization import draw_features, annotate_tracking, plot_map
from orbslam3.vocabulary import Vocabulary


@pytest.fixture(scope="module")
def vocabulary():
    return Vocabulary.from_text(Path(__file__).resolve().parents[1]/"assets/ORBvoc.txt")


@pytest.mark.parametrize("sensor", [Sensor.MONOCULAR, Sensor.STEREO, Sensor.RGBD])
def test_real_images_initialization_tracking_trajectory_and_map(sensor, vocabulary):
    scene = TexturedPlanes()
    calibration = Calibration(scene.camera, baseline_fx=scene.camera.fx*.1, depth_threshold=4.)
    slam = SLAM(sensor, calibration, vocabulary=vocabulary, nfeatures=1000, nlevels=4)
    times = np.arange(6)*.12
    results = [slam.process(scene.frame(sensor, float(time))) for time in times]
    initialized = [result for result in results if result.state == TrackingState.OK]
    assert len(initialized) >= 4
    if sensor.monocular:
        # At .12s the nearest plane has only .75 degrees of parallax; the
        # source correctly waits for its one-degree initialization threshold.
        assert results[1].state == TrackingState.NOT_INITIALIZED
        assert results[2].state == TrackingState.OK
    trajectory = slam.trajectory()
    assert len(trajectory) >= 4
    positions = np.array([inverse_pose(pose)[:3, 3] for _, pose, _, _ in trajectory])
    truth = np.array([scene.position(timestamp) for timestamp, _, _, _ in trajectory])
    # Monocular scale is a gauge. Depth modes must recover metric translation.
    scale = np.sum(positions*truth)/np.sum(positions*positions) if sensor.monocular else 1.
    errors = np.linalg.norm(scale*positions-truth, axis=1)
    assert errors.max() < .025
    assert len(slam.atlas.current.points) > 80
    assert all(np.isfinite(point.position).all() and point.observation_count > 0 for point in slam.atlas.current.points.values())
    assert all(np.linalg.det(pose[:3, :3]) > .99999 for _, pose, _, _ in trajectory)


def test_headless_visualization_and_generic_exports_are_optional(tmp_path, vocabulary):
    scene = TexturedPlanes()
    slam = SLAM(Sensor.RGBD, Calibration(scene.camera, baseline_fx=26, depth_threshold=4),
                vocabulary=vocabulary, nfeatures=750, nlevels=4)
    results = [slam.process(scene.frame(Sensor.RGBD, float(t))) for t in (0., .1)]
    frame = slam.frame
    feature_image = draw_features(scene.render(.1)[0], frame)
    assert feature_image.shape == (240, 320, 3)
    assert annotate_tracking(feature_image, results[-1]).shape == feature_image.shape
    save_trajectory(tmp_path/"trajectory.csv", slam.trajectory())
    save_points(tmp_path/"points.csv", slam.atlas)
    assert np.loadtxt(tmp_path/"trajectory.csv", delimiter=",", skiprows=1).shape == (2, 10)
    import matplotlib
    matplotlib.use("Agg")
    figure, _ = plot_map(slam)
    figure.savefig(tmp_path/"map.png")
    assert (tmp_path/"map.png").stat().st_size > 1000
    import matplotlib.pyplot as plt
    plt.close(figure)
