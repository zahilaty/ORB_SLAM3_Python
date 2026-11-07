"""MLPnP geometric derivatives, linear cases, and RANSAC source decisions."""

from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pytest

from orbslam3.camera import Pinhole
from orbslam3.geometry import make_pose, so3_exp
from orbslam3.pnp import MLPnPSolver, bearing_nullspaces, compute_pose, mlpnp_residuals_and_jacobians


CAMERA = Pinhole(500, 510, 320, 240)


def pose_case(count=40, planar=False):
    rng = np.random.default_rng(41)
    points = rng.uniform([-2, -1.5, 3], [2, 1.5, 7], (count, 3))
    if planar:
        # Source detects rank of uncentered scatter, so use a plane through the
        # world origin and translate the camera to keep every landmark visible.
        points[:, 2] = 0
    pose = make_pose(so3_exp([.1, -.15, .05]), [.2, -.1, 5 if planar else .3])
    camera_points = points @ pose[:3, :3].T + pose[:3, 3]
    bearings = camera_points / camera_points[:, 2:3]
    return points, pose, bearings


def test_bearing_nullspaces_are_orthonormal_and_perpendicular():
    _, _, bearings = pose_case()
    bases = bearing_nullspaces(bearings)
    np.testing.assert_allclose(np.einsum("nji,njk->nik", bases, bases),
                               np.broadcast_to(np.eye(2), (len(bases), 2, 2)), atol=1e-14)
    np.testing.assert_allclose(np.einsum("ni,nij->nj", bearings, bases), 0, atol=1e-14)


@pytest.mark.parametrize("rotation", [[.1, -.15, .05], [0, 0, 0], [1e-8, -2e-8, 1e-8]])
def test_mlpnp_additive_rodrigues_jacobian(rotation):
    points, _, bearings = pose_case(8)
    bases = bearing_nullspaces(bearings)
    state = np.r_[rotation, [.2, -.1, .3]]
    _, analytical = mlpnp_residuals_and_jacobians(state, points, bases)
    epsilon = 1e-7
    numerical = np.column_stack([
        (mlpnp_residuals_and_jacobians(state + np.eye(6)[i] * epsilon, points, bases)[0]
         - mlpnp_residuals_and_jacobians(state - np.eye(6)[i] * epsilon, points, bases)[0]) / (2 * epsilon)
        for i in range(6)
    ])
    np.testing.assert_allclose(analytical, numerical, atol=3e-9, rtol=5e-7)


@pytest.mark.parametrize("count", [6, 40])
def test_mlpnp_general_pose_recovery(count):
    points, pose, bearings = pose_case(count)
    estimate = compute_pose(bearings, points)
    np.testing.assert_allclose(estimate, pose, atol=2e-7)


def test_original_cpp_golden_general_noisy_and_planar_branches():
    fixture = json.loads((Path(__file__).parent / "fixtures/mlpnp_reference.json").read_text())
    for case in fixture["cases"]:
        estimate = compute_pose(np.array(case["bearings"]), np.array(case["points"]))
        np.testing.assert_allclose(estimate, case["expected_pose"], atol=2e-8, rtol=2e-8)
        if case["planar"]:
            # The baseline planar scale is wrong for these inputs. This is a
            # verified original-C++ limitation, deliberately preserved here.
            assert np.linalg.norm(estimate[:3, 3] - np.array(case["input_pose"])[:3, 3]) > 1


def test_mlpnp_optional_projected_bearing_covariances():
    points, pose, bearings = pose_case()
    covariances = np.broadcast_to(np.diag([.001, .002, .003]), (len(points), 3, 3))
    estimate = compute_pose(bearings, points, covariances)
    np.testing.assert_allclose(estimate, pose, atol=1e-7)


def make_frame_and_matches(count=60):
    points, pose, bearings = pose_case(count)
    pixels = np.asarray([CAMERA.project(ray) for ray in bearings])
    frame = SimpleNamespace(camera=CAMERA, n_left=-1,
                            features=SimpleNamespace(points=pixels, octaves=np.zeros(count, int)),
                            inverse_variances=np.array([1.0]))
    # The concrete Frame uses len(FeatureSet); a small independent record avoids
    # coupling this test to SLAM state construction.
    class Features:
        points = pixels
        octaves = np.zeros(count, int)
        def __len__(self):
            return count
    frame.features = Features()
    matches = [SimpleNamespace(position=point, bad=False) for point in points]
    return frame, matches, pose


def test_mlpnp_ransac_filters_missing_and_bad_points_and_wrong_correspondences():
    frame, matches, true_pose = make_frame_and_matches()
    frame.features.points[:12] += [70, -40]
    matches[13] = None
    matches[14].bad = True
    solver = MLPnPSolver(frame, matches, seed=5)
    solver.set_ransac_parameters(.99, 10, 300, 6, .5, 5.991)
    result = solver.iterate(5)
    assert result.converged
    assert result.count == 46
    assert not result.inliers[:12].any()
    assert not result.inliers[13:15].any()
    np.testing.assert_allclose(result.pose, true_pose, atol=1e-6)


def test_source_ransac_or_loop_and_refinement_discard(monkeypatch):
    frame, matches, pose = make_frame_and_matches(12)
    solver = MLPnPSolver(frame, matches)
    solver.set_ransac_parameters(.99, 10, 30, 6, .5, 5.991)
    # No support forces the OR loop to consume the entire global budget in one
    # nominally one-trial call, rather than silently changing source scheduling.
    monkeypatch.setattr(solver, "check_inliers", lambda value: np.zeros(12, bool))
    result = solver.iterate(1)
    assert result.iterations == solver.maximum_iterations
    assert result.exhausted
    assert not result.converged


def test_insufficient_correspondences_fail_without_a_pose():
    frame, matches, _ = make_frame_and_matches(5)
    result = MLPnPSolver(frame, matches).iterate(5)
    assert result.exhausted
    assert not result.converged
    assert result.pose is None
    assert result.inliers.shape == (5,)
