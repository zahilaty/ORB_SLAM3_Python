"""Source-specific similarity estimation and loop-correction geometry tests."""

import numpy as np
import pytest
import json
from pathlib import Path

from orbslam3.geometry import Sim3, inverse_pose, make_pose, so3_exp
from orbslam3.loop_geometry import (
    EssentialGraphEdge,
    Sim3ReprojectionMatch,
    _four_dof_retract,
    correct_map_points,
    essential_graph_edges,
    four_dof_graph_factor,
    optimize_essential_graph,
    optimize_sim3,
    sim3_graph_factor,
    sim3_projection_factors,
)
from orbslam3.sim3_solver import Sim3Solver, horn_sim3


class Camera:
    def project(self, point):
        x, y, z = point
        return np.array([500 * x / z + 320, 510 * y / z + 240])

    def project_jacobian(self, point):
        x, y, z = point
        return np.array([[500 / z, 0, -500 * x / z**2],
                         [0, 510 / z, -510 * y / z**2]])


CAMERA = Camera()


def similarity_case(count=60, scale=1.3):
    rng = np.random.default_rng(52)
    points2 = rng.uniform([-1, -.8, 4], [1, .8, 6], (count, 3))
    transform = Sim3(so3_exp([.04, -.06, .02]), [.3, -.1, .2], scale)
    return transform, transform.map(points2), points2


@pytest.mark.parametrize("fix_scale", [False, True])
def test_horn_absolute_orientation_and_scale(fix_scale):
    transform, points1, points2 = similarity_case(scale=1 if fix_scale else 1.3)
    estimate = horn_sim3(points1[:3], points2[:3], fix_scale)
    np.testing.assert_allclose(estimate.rotation, transform.rotation, atol=1e-13)
    np.testing.assert_allclose(estimate.translation, transform.translation, atol=1e-13)
    assert estimate.scale == pytest.approx(transform.scale, abs=1e-13)


def test_horn_identity_rotation_has_no_divide_by_zero():
    _, _, points = similarity_case()
    estimate = horn_sim3(points * 2 + [1, 2, 3], points)
    np.testing.assert_allclose(estimate.rotation, np.eye(3), atol=1e-13)
    np.testing.assert_allclose(estimate.translation, [1, 2, 3], atol=1e-13)
    assert estimate.scale == pytest.approx(2)
    with pytest.raises(ValueError, match="Degenerate"):
        horn_sim3(np.zeros((3, 3)), np.zeros((3, 3)))


def test_sim3_ransac_wrong_matches_and_original_index_remap():
    transform, points1, points2 = similarity_case()
    points1[:12] = points1[12:24]
    solver = Sim3Solver(points1, points2, CAMERA, CAMERA, fix_scale=False,
                        original_indices=np.arange(60) * 2, output_size=120, seed=8)
    solver.set_ransac_parameters(.99, 20, 300)
    result = solver.iterate(1)
    while not result.converged and not result.exhausted:
        result = solver.iterate(1)
    assert result.converged
    assert result.count == 48
    np.testing.assert_array_equal(np.flatnonzero(result.inliers), np.arange(12, 60) * 2)
    np.testing.assert_allclose(result.transform.rotation, transform.rotation, atol=1e-12)
    np.testing.assert_allclose(result.transform.translation, transform.translation, atol=1e-12)
    assert result.transform.scale == pytest.approx(transform.scale)


def test_source_ransac_strict_minimum_and_truncated_thresholds():
    _, points1, points2 = similarity_case(6, scale=1)
    solver = Sim3Solver(points1, points2, CAMERA, CAMERA, variances1=np.full(6, 1.44), seed=0)
    np.testing.assert_array_equal(solver.thresholds1, np.full(6, 13))
    assert solver.maximum_iterations == 1
    result = solver.find()
    assert result.exhausted
    assert not result.converged  # Source requires strictly MORE than six.
    assert result.count == 0
    assert solver.best_count == 6


def make_matches(count=40, scale=1.3):
    transform, points1, points2 = similarity_case(count, scale)
    matches = [Sim3ReprojectionMatch(p1, p2, CAMERA.project(p1), CAMERA.project(p2),
                                     CAMERA, CAMERA, .7, .8)
               for p1, p2 in zip(points1, points2)]
    return transform, matches


def test_bilateral_sim3_projection_jacobians():
    transform, matches = make_matches()
    transform = Sim3.exp(np.array([.01, -.02, .01, .05, -.01, .03, .04])) @ transform
    match = matches[0]
    _, _, forward_jac, inverse_jac = sim3_projection_factors(transform, match)
    epsilon = 1e-7
    numerical_forward, numerical_inverse = [], []
    for axis in range(7):
        delta = np.eye(7)[axis] * epsilon
        plus = sim3_projection_factors(Sim3.exp(delta) @ transform, match)
        minus = sim3_projection_factors(Sim3.exp(-delta) @ transform, match)
        numerical_forward.append((plus[0] - minus[0]) / (2 * epsilon))
        numerical_inverse.append((plus[1] - minus[1]) / (2 * epsilon))
    np.testing.assert_allclose(forward_jac, np.array(numerical_forward).T, atol=6e-7, rtol=1e-7)
    np.testing.assert_allclose(inverse_jac, np.array(numerical_inverse).T, atol=6e-7, rtol=1e-7)


@pytest.mark.parametrize("fix_scale", [False, True])
def test_sim3_optimization_rejects_bilateral_outliers(fix_scale):
    transform, matches = make_matches(scale=1 if fix_scale else 1.3)
    matches[0].pixel1 = matches[0].pixel1 + [60, -40]
    matches[1].pixel2 = matches[1].pixel2 + [-30, 50]
    start = Sim3.exp(np.array([.005, -.01, .005, .01, -.01, .02, 0 if fix_scale else .015])) @ transform
    result = optimize_sim3(start, matches, fix_scale=fix_scale)
    assert result.count == 38
    np.testing.assert_array_equal(result.inliers, np.arange(40) >= 2)
    np.testing.assert_allclose(result.transform.rotation, transform.rotation, atol=1e-8)
    np.testing.assert_allclose(result.transform.translation, transform.translation, atol=1e-7)
    assert result.transform.scale == pytest.approx(transform.scale, abs=1e-8)
    np.testing.assert_array_equal(result.hessian, np.zeros((7, 7)))


def test_sim3_optimization_does_not_commit_when_under_ten_survive():
    transform, matches = make_matches(9)
    start = Sim3.exp(np.array([.005, -.01, .005, .01, -.01, .02, .015])) @ transform
    result = optimize_sim3(start, matches)
    assert result.count == 0
    assert result.transform is start
    assert result.inliers.sum() == 9


def test_original_cpp_golden_sim3_reprojection_optimization():
    fixture = json.loads((Path(__file__).parent / "fixtures/sim3_optimization_reference.json").read_text())
    for case in fixture["cases"]:
        start = case["start"]
        initial = Sim3(np.array(start["rotation"]), np.array(start["translation"]), start["scale"])
        matches = [Sim3ReprojectionMatch(np.array(match["point1"]), np.array(match["point2"]),
                                         np.array(match["pixel1"]), np.array(match["pixel2"]),
                                         CAMERA, CAMERA, match["inverse_variance1"],
                                         match["inverse_variance2"])
                   for match in case["matches"]]
        result = optimize_sim3(initial, matches, fix_scale=case["fix_scale"])
        packed = np.r_[result.transform.rotation.ravel(), result.transform.translation, result.transform.scale]
        np.testing.assert_allclose(packed, case["expected"], atol=3e-8, rtol=3e-8)
        np.testing.assert_array_equal(result.inliers, case["expected_inliers"])


def test_essential_graph_measurements_and_reference_point_correction():
    poses = {
        0: Sim3(np.eye(3), np.zeros(3)),
        1: Sim3(so3_exp([.01, -.03, .02]), np.array([-.4, .1, .02]), 1.1),
        2: Sim3(so3_exp([.02, -.05, .04]), np.array([-.8, .15, .03]), 1.2),
    }
    edges = [EssentialGraphEdge(i, j, poses[j] @ poses[i].inverse())
             for i, j in [(0, 1), (1, 2), (2, 0)]]
    start = poses.copy()
    perturbation = np.array([.01, -.01, .005, .04, -.02, .01, .03])
    for key in (1, 2):
        start[key] = Sim3.exp(perturbation * key) @ poses[key]
    result = optimize_essential_graph(start, edges, {0})
    assert result.similarities[0] is start[0]
    assert result.final_cost < result.initial_cost * 1e-8
    for key in poses:
        np.testing.assert_allclose(result.similarities[key].rotation, poses[key].rotation, atol=2e-7)
        np.testing.assert_allclose(result.similarities[key].translation, poses[key].translation, atol=2e-7)
        assert result.similarities[key].scale == pytest.approx(poses[key].scale, abs=2e-7)
    points = {1: np.array([.4, .2, 5.])}
    corrected = correct_map_points(points, {1: 2}, start, result.similarities)
    np.testing.assert_allclose(result.similarities[2].map(corrected[1]), start[2].map(points[1]), atol=1e-12)


def test_source_graph_edge_selection_preserves_duplicate_historical_loop_edges():
    poses = {i: Sim3(np.eye(3), np.array([-.3 * i, 0, 0])) for i in range(4)}
    edges = essential_graph_edges(
        poses, parents={1: 0, 2: 1, 3: 2}, historical_loops={3: {0}},
        covisibility={3: {0: 150, 1: 100, 2: 110}},
        previous={3: 2}, loop_connections={3: {0, 1}}, current_id=3, loop_id=0,
    )
    kinds = [(e.first, e.second, e.kind) for e in edges]
    assert (3, 0, "new_loop") in kinds
    assert (3, 0, "historical_loop") in kinds
    assert (3, 1, "new_loop") in kinds
    assert (3, 1, "covisibility") not in kinds
    assert (3, 2, "spanning_tree") in kinds
    assert (3, 2, "inertial_temporal") in kinds
    for edge in edges:
        np.testing.assert_allclose(sim3_graph_factor(poses[edge.first], poses[edge.second], edge.measurement), 0, atol=1e-12)


def test_inertial_four_dof_retraction_preserves_body_gravity_and_extrinsic():
    camera_from_body = make_pose(so3_exp([.15, -.1, .05]), [.1, .04, -.02])
    world_from_body = make_pose(so3_exp([.1, -.2, .3]), [1., 2., .5])
    camera_pose = camera_from_body @ inverse_pose(world_from_body)
    initial = Sim3(camera_pose[:3, :3], camera_pose[:3, 3])
    increment = np.array([.2, .4, -.3, .1])
    updated = _four_dof_retract(initial, increment, camera_from_body)
    updated_body = inverse_pose(updated.camera_pose()) @ camera_from_body
    np.testing.assert_allclose(updated_body[:3, :3][2], world_from_body[:3, :3][2], atol=1e-14)
    np.testing.assert_allclose(updated_body[:3, 3], world_from_body[:3, 3] + increment[1:], atol=1e-14)


def test_four_dof_graph_and_source_information_and_no_spanning_tree():
    poses = {i: Sim3(so3_exp([0., 0., .03 * i]), np.array([-.4 * i, .01 * i, 0.]))
             for i in range(4)}
    edges = essential_graph_edges(poses, parents={1: 0, 2: 1, 3: 2},
                                   previous={1: 0, 2: 1, 3: 2},
                                   covisibility={3: {0: 150, 1: 110, 2: 120}}, four_dof=True)
    assert all(e.kind != "spanning_tree" for e in edges)
    assert not any(e.kind == "covisibility" and e.second == 2 for e in edges)
    np.testing.assert_array_equal(edges[0].information.diagonal(), [1000, 1000, 1, 1, 1, 1])
    for edge in edges:
        np.testing.assert_allclose(four_dof_graph_factor(poses[edge.first], poses[edge.second], edge.measurement), 0, atol=1e-14)
    start = {key: (_four_dof_retract(pose, np.array([.01, .03, -.02, .01]) * key,
                                   np.eye(4)) if key else pose) for key, pose in poses.items()}
    result = optimize_essential_graph(start, edges, {0}, four_dof=True)
    assert result.final_cost < result.initial_cost * 1e-8
    for key in poses:
        np.testing.assert_allclose(result.poses[key], poses[key].camera_pose(), atol=1e-7)
