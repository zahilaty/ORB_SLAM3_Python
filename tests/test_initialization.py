import numpy as np
import pytest
from orbslam3.camera import Pinhole
from orbslam3.geometry import make_pose, so3_exp, transform_points
from orbslam3.initialization import TwoViewReconstruction, normalize_points, decompose_homography, check_motion


def test_source_hypothesis_selection_and_structure():
    rng = np.random.default_rng(18)
    camera = Pinhole(500, 500, 320, 240)
    points = rng.uniform([-2., -1.5, 4.], [2., 1.5, 9.], (180, 3))
    expected = make_pose(so3_exp([.02, -.04, .01]), [.6, .03, .01])
    keys1 = camera.project(points)
    keys2 = camera.project(transform_points(expected, points))
    result = TwoViewReconstruction(camera.K).reconstruct(keys1, keys2, np.arange(len(points)))
    assert result.success
    assert result.triangulated.sum() == len(points)
    assert result.model == "fundamental"
    np.testing.assert_allclose(result.pose[:3, :3], expected[:3, :3], atol=1e-10)
    np.testing.assert_allclose(result.pose[:3, 3], expected[:3, 3]/np.linalg.norm(expected[:3, 3]), atol=1e-10)
    np.testing.assert_allclose(result.points*np.linalg.norm(expected[:3, 3]), points, atol=1e-8)


def test_homography_eight_hypotheses_and_source_ambiguity_rejection():
    rng = np.random.default_rng(18)
    camera = Pinhole(500, 500, 320, 240)
    points = rng.uniform([-2., -1.5, 6.], [2., 1.5, 6.], (180, 3))
    pose = make_pose(so3_exp([.02, -.04, .01]), [.6, .03, .01])
    keys1, keys2 = camera.project(points), camera.project(transform_points(pose, points))
    h = camera.K @ (pose[:3, :3] + np.outer(pose[:3, 3], [0., 0., 1.])/6.) @ np.linalg.inv(camera.K)
    hypotheses = decompose_homography(h, camera.K)
    assert len(hypotheses) == 8
    checked = [check_motion(r, t, keys1, keys2, np.c_[np.arange(180), np.arange(180)],
                            np.ones(180, bool), camera.K, 4.) for r, t in hypotheses]
    best = int(np.argmax([entry[0] for entry in checked]))
    assert checked[best][0] == 180
    np.testing.assert_allclose(hypotheses[best][0], pose[:3, :3], atol=1e-12)
    np.testing.assert_allclose(checked[best][1]*np.linalg.norm(pose[:3, 3]), points, atol=1e-10)
    # Source deliberately rejects if a second valid solution explains >=75%.
    # A frontoparallel plane at this baseline has exactly that ambiguity.
    counts = sorted(entry[0] for entry in checked)
    assert counts[-2] >= .75*counts[-1]


def test_pure_rotation_and_small_parallax_rejected():
    rng = np.random.default_rng(2)
    camera = Pinhole(500, 500, 320, 240)
    points = rng.uniform([-2., -1.5, 4.], [2., 1.5, 9.], (150, 3))
    keys1 = camera.project(points)
    for baseline in (0., .001):
        pose = make_pose(so3_exp([.02, -.04, .01]), [baseline, 0., 0.])
        result = TwoViewReconstruction(camera.K).reconstruct(keys1, camera.project(transform_points(pose, points)), np.arange(len(points)))
        assert not result.success


def test_outlier_scores_and_points_not_ransac_library_defaults():
    rng = np.random.default_rng(6)
    camera = Pinhole(500, 500, 320, 240)
    points = rng.uniform([-2., -1.5, 4.], [2., 1.5, 9.], (150, 3))
    pose = make_pose(so3_exp([.01, -.03, .005]), [.6, 0., .03])
    keys1 = camera.project(points)
    keys2 = camera.project(transform_points(pose, points)) + rng.normal(0, .1, (150, 2))
    keys2[::5] = rng.uniform([0., 0.], [640., 480.], (30, 2))
    result = TwoViewReconstruction(camera.K).reconstruct(keys1, keys2, np.arange(len(points)))
    assert result.success
    assert result.triangulated.sum() >= 118
    assert not result.triangulated[::5].any()
    assert np.linalg.norm(result.pose[:3, :3]-pose[:3, :3]) < .01


def test_normalization_is_anisotropic_absolute_deviation():
    points = np.array([[0., 0.], [3., 10.], [6., 8.]])
    normalized, transform = normalize_points(points)
    np.testing.assert_allclose(np.abs(normalized).mean(axis=0), 1.)
    assert transform[0, 0] != transform[1, 1]
