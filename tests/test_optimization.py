"""Geometric, optimization, and optional compiled-baseline differential tests."""

import os
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from orbslam3.optimization import (
    ReprojectionObservation,
    _se3_exp,
    bundle_adjustment,
    huber,
    local_bundle_adjustment,
    merge_bundle_adjustment,
    optimize_pose,
    projection_factor,
)


class PinholeFixture:
    """An independent analytic test camera; core code consumes a camera protocol."""

    def project(self, point):
        x, y, z = point
        return np.array([500 * x / z + 320, 510 * y / z + 240])

    def project_jacobian(self, point):
        x, y, z = point
        return np.array([[500 / z, 0, -500 * x / z**2],
                         [0, 510 / z, -510 * y / z**2]])


CAMERA = PinholeFixture()


def observation(point, point_id, pose_id=0, pose=None, stereo=False, extrinsic=None):
    if pose is None:
        pose = np.eye(4)
    transformed = pose[:3, :3] @ point + pose[:3, 3]
    if extrinsic is not None:
        transformed = extrinsic[:3, :3] @ transformed + extrinsic[:3, 3]
    pixel = CAMERA.project(transformed)
    if stereo:
        pixel = np.r_[pixel, pixel[0] - 40 / transformed[2]]
    return ReprojectionObservation(pose_id, point_id, pixel, CAMERA, 1 / 1.2**2,
                                    40 if stereo else 0, extrinsic)


@pytest.mark.parametrize("stereo,right_camera", [(False, False), (True, False), (False, True)])
def test_projection_pose_and_landmark_jacobians(stereo, right_camera):
    pose = _se3_exp(np.array([0.1, -0.03, 0.05, 0.2, -0.1, 0.4]))
    point = np.array([0.5, -0.4, 4.5])
    extrinsic = _se3_exp(np.array([0.04, 0.01, -0.03, -0.12, 0.01, 0.02])) if right_camera else None
    edge = observation(point, 0, stereo=stereo, extrinsic=extrinsic)
    error, jac_pose, jac_point, depth = projection_factor(pose, point, edge)
    epsilon = 1e-7
    numerical_pose = np.column_stack([
        (projection_factor(_se3_exp(np.eye(6)[i] * epsilon) @ pose, point, edge)[0]
         - projection_factor(_se3_exp(-np.eye(6)[i] * epsilon) @ pose, point, edge)[0]) / (2 * epsilon)
        for i in range(6)
    ])
    numerical_point = np.column_stack([
        (projection_factor(pose, point + np.eye(3)[i] * epsilon, edge)[0]
         - projection_factor(pose, point - np.eye(3)[i] * epsilon, edge)[0]) / (2 * epsilon)
        for i in range(3)
    ])
    assert error.shape == (3 if stereo else 2,)
    assert depth > 0
    np.testing.assert_allclose(jac_pose, numerical_pose, atol=3e-7, rtol=2e-7)
    np.testing.assert_allclose(jac_point, numerical_point, atol=3e-7, rtol=2e-7)


def synthetic_points(count=60):
    rng = np.random.default_rng(63)
    return rng.uniform([-1.5, -1.0, 3.0], [1.5, 1.0, 7.0], (count, 3))


def test_pose_optimization_recovers_pose_and_rejects_wrong_matches():
    points = synthetic_points()
    true_pose = _se3_exp(np.array([0.02, -0.04, 0.01, 0.12, -0.05, 0.15]))
    edges = [observation(point, i, pose=true_pose, stereo=(i % 3 == 0))
             for i, point in enumerate(points)]
    for edge in edges[:7]:
        edge.measurement[:2] += [55, -35]
    start = _se3_exp(np.array([0.015, -0.012, 0.02, -0.03, 0.07, -0.05])) @ true_pose
    result = optimize_pose(start, points, edges)
    np.testing.assert_allclose(result.pose, true_pose, atol=1e-7)
    np.testing.assert_array_equal(result.outliers, np.arange(len(points)) < 7)
    assert result.inliers == len(points) - 7
    assert len(result.stage_iterations) == 4
    assert all(1 <= iterations <= 10 for iterations in result.stage_iterations)


def test_small_pose_graph_uses_one_stage_and_underconstrained_pose_is_unchanged():
    points = synthetic_points(7)
    edges = [observation(point, i) for i, point in enumerate(points)]
    pose = _se3_exp(np.array([0.01, 0.02, -0.02, 0.03, 0.02, -0.01]))
    result = optimize_pose(pose, points, edges)
    np.testing.assert_allclose(result.pose, np.eye(4), atol=1e-7)
    assert len(result.stage_iterations) == 1
    underconstrained = optimize_pose(pose, points, edges[:2])
    np.testing.assert_array_equal(underconstrained.pose, pose)
    assert underconstrained.inliers == 0


def test_huber_uses_block_norm_and_information_weighting():
    cost, weight = huber(100.0, 4.0)
    assert cost == 36.0
    assert weight == 0.2
    assert huber(3.0, 4.0) == (3.0, 1.0)
    epsilon = 1e-5
    derivative = (huber(100 + epsilon, 4)[0] - huber(100 - epsilon, 4)[0]) / (2 * epsilon)
    assert derivative == pytest.approx(weight)


def make_bundle():
    points = {i: point for i, point in enumerate(synthetic_points(35))}
    poses = {
        0: np.eye(4),
        1: _se3_exp(np.array([0.005, -0.02, 0.01, -0.4, 0.0, 0.05])),
        2: _se3_exp(np.array([-0.01, -0.04, -0.01, -0.8, 0.02, 0.08])),
    }
    # Rectified stereo anchors metric scale, while remaining edges exercise mono.
    edges = [observation(point, i, k, pose, stereo=(i % 2 == 0))
             for k, pose in poses.items() for i, point in points.items()]
    rng = np.random.default_rng(21)
    noisy_points = {key: point + rng.normal(0, 0.03, 3) for key, point in points.items()}
    noisy_poses = {key: pose.copy() for key, pose in poses.items()}
    for key in (1, 2):
        noisy_poses[key] = _se3_exp(rng.normal(0, 0.01, 6)) @ poses[key]
    return poses, points, noisy_poses, noisy_points, edges


def test_bundle_adjustment_fixed_gauge_and_metric_structure():
    poses, points, noisy_poses, noisy_points, edges = make_bundle()
    original_points = {key: point.copy() for key, point in noisy_points.items()}
    result = bundle_adjustment(noisy_poses, noisy_points, edges, {0}, iterations=20)
    np.testing.assert_array_equal(result.poses[0], poses[0])
    assert result.final_cost < result.initial_cost * 1e-10
    for key in poses:
        np.testing.assert_allclose(result.poses[key], poses[key], atol=2e-6)
    for key in points:
        np.testing.assert_allclose(result.points[key], points[key], atol=5e-6)
        np.testing.assert_array_equal(noisy_points[key], original_points[key])
    assert not result.outliers.any()


def test_local_bundle_adjustment_outliers_and_fixed_observers():
    _, _, noisy_poses, noisy_points, edges = make_bundle()
    edges[0].measurement[:] += 200
    result = local_bundle_adjustment(noisy_poses, noisy_points, edges, {0, 2})
    assert result.outliers[0]
    np.testing.assert_array_equal(result.poses[0], noisy_poses[0])
    np.testing.assert_array_equal(result.poses[2], noisy_poses[2])
    assert result.final_cost < result.initial_cost
    with pytest.raises(ValueError, match="no fixed"):
        local_bundle_adjustment(noisy_poses, noisy_points, edges, set())


def test_merge_bundle_adjustment_rejects_edges_between_two_stages():
    poses, points, noisy_poses, noisy_points, edges = make_bundle()
    edges[0].measurement[:] += 100
    result = merge_bundle_adjustment(noisy_poses, noisy_points, edges, {0})
    assert result.outliers[0]
    # A wrong observation can distort its landmark enough that other views of
    # that same landmark are also rejected during the five-iteration stage.
    assert all(edges[i].point_id == 0 for i in np.flatnonzero(result.outliers))
    assert result.iterations <= 15
    for key in poses:
        np.testing.assert_allclose(result.poses[key], poses[key], atol=2e-5)


def test_unobserved_landmarks_and_zero_iterations_stay_unchanged():
    poses, points, _, _, edges = make_bundle()
    points[100] = np.array([9, 9, 9.0])
    result = bundle_adjustment(poses, points, edges, {0}, iterations=0)
    np.testing.assert_array_equal(result.points[100], points[100])
    assert result.iterations == 0
    assert result.final_cost < 1e-20


CPP_POSE_REFERENCE = r'''
#include <iomanip>
#include <iostream>
#include <vector>
#include "OptimizableTypes.h"
#include "CameraModels/Pinhole.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
int main() {
    using namespace g2o;
    ORB_SLAM3::Pinhole camera(std::vector<float>{500,510,320,240});
    SparseOptimizer optimizer;
    auto linear = new LinearSolverDense<BlockSolver_6_3::PoseMatrixType>();
    auto solver = new OptimizationAlgorithmLevenberg(new BlockSolver_6_3(linear));
    optimizer.setAlgorithm(solver);
    Eigen::Matrix4d initial;
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) std::cin >> initial(i,j);
    const SE3Quat original(initial.topLeftCorner<3,3>(), initial.topRightCorner<3,1>());
    auto vertex = new VertexSE3Expmap();
    vertex->setId(0); vertex->setEstimate(original); optimizer.addVertex(vertex);
    int n; std::cin >> n;
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> edges;
    for(int i=0;i<n;i++) {
        auto edge = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();
        Eigen::Vector3d point; Eigen::Vector2d pixel; double invvar;
        std::cin >> point[0] >> point[1] >> point[2] >> pixel[0] >> pixel[1] >> invvar;
        edge->Xw=point; edge->pCamera=&camera;
        edge->setVertex(0,vertex); edge->setMeasurement(pixel);
        edge->setInformation(Eigen::Matrix2d::Identity()*invvar);
        auto huber = new RobustKernelHuber(); huber->setDelta(std::sqrt(5.991));
        edge->setRobustKernel(huber); optimizer.addEdge(edge); edges.push_back(edge);
    }
    std::vector<int> outliers(n,0);
    for(int stage=0;stage<4;stage++) {
        vertex->setEstimate(original);
        optimizer.initializeOptimization(0); optimizer.optimize(10);
        for(int i=0;i<n;i++) {
            edges[i]->computeError(); outliers[i] = edges[i]->chi2()>5.991;
            edges[i]->setLevel(outliers[i]);
            if(stage==2) edges[i]->setRobustKernel(nullptr);
        }
        if(n<10) break;
    }
    std::cout << std::setprecision(17);
    const Eigen::Matrix4d result = vertex->estimate().to_homogeneous_matrix();
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) std::cout<<result(i,j)<<" ";
    std::cout<<"\n";
    for(int i=0;i<n;i++) std::cout<<outliers[i]<<" ";
    std::cout<<"\n";
}
'''


def test_recorded_g2o_pose_reference():
    """Standalone mathematical regression against shipped original-C++ output."""
    fixture_path = Path(__file__).parent / "fixtures/pose_optimization_reference.json"
    fixture = json.loads(fixture_path.read_text())
    points = np.array(fixture["points"])
    edges = [ReprojectionObservation(0, i, np.array(pixel), CAMERA,
                                      fixture["inverse_variance"])
             for i, pixel in enumerate(fixture["measurements"])]
    result = optimize_pose(np.array(fixture["start_pose"]), points, edges)
    np.testing.assert_allclose(result.pose, fixture["expected_pose"], atol=2e-8, rtol=2e-8)
    np.testing.assert_array_equal(result.outliers, fixture["expected_outliers"])


@pytest.mark.reference
def test_pose_optimizer_against_original_g2o(tmp_path):
    """Compile/run the exact local factors and optimizer, never a runtime binding."""
    reference_path = os.environ.get("ORBSLAM3_REFERENCE")
    if reference_path is None:
        pytest.skip("Set ORBSLAM3_REFERENCE to opt into compiling the C++ oracle")
    reference = Path(reference_path)
    if not (reference / "lib/libORB_SLAM3.so").exists() or not shutil.which("g++"):
        pytest.skip("Built original C++ reference is unavailable")
    native_prefix = reference / ".venv"
    source = tmp_path / "pose_reference.cpp"
    source.write_text(CPP_POSE_REFERENCE)
    executable = tmp_path / "pose_reference"
    include_paths = [reference, reference / "include", reference / "Thirdparty/Sophus",
                     native_prefix / "include", native_prefix / "include/eigen3",
                     native_prefix / "include/opencv4"]
    library_paths = [reference / "lib", reference / "Thirdparty/g2o/lib",
                     reference / "Thirdparty/DBoW2/lib", native_prefix / "lib",
                     reference / ".local/lib"]
    # Eigen's vector alignment ABI must match the locally built libraries.
    command = ["g++", "-std=c++17", "-O1", "-march=native", "-DNDEBUG",
               str(source), "-o", str(executable)]
    command += [f"-I{path}" for path in include_paths]
    command += [f"-L{path}" for path in library_paths]
    command += ["-lORB_SLAM3", "-lg2o"]
    command += [f"-Wl,-rpath,{path}" for path in library_paths]
    subprocess.run(command, check=True, capture_output=True, text=True)
    points = synthetic_points(45)
    true_pose = _se3_exp(np.array([0.02, -0.04, 0.01, 0.12, -0.05, 0.15]))
    start = _se3_exp(np.array([0.01, -0.015, 0.005, 0.03, 0.02, -0.02])) @ true_pose
    edges = [observation(point, i, pose=true_pose) for i, point in enumerate(points)]
    for edge in edges[:5]:
        edge.measurement[:] += [60, -30]
    # Add noise to ensure comparison exercises optimization, rather than only
    # converging to a trivially perfect synthetic solution.
    rng = np.random.default_rng(11)
    for edge in edges:
        edge.measurement[:] += rng.normal(0, 0.4, 2)
    rows = [" ".join(map(str, start.ravel())), str(len(points))]
    rows += [" ".join(map(str, [*point, *edge.measurement, edge.inverse_variance]))
             for point, edge in zip(points, edges)]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(map(str, library_paths))
    result = subprocess.run([str(executable)], input="\n".join(rows), text=True,
                            capture_output=True, check=True, env=environment)
    lines = result.stdout.strip().splitlines()
    cpp_pose = np.fromstring(lines[0], sep=" ").reshape(4, 4)
    cpp_outliers = np.fromstring(lines[1], sep=" ").astype(bool)
    python = optimize_pose(start, points, edges)
    np.testing.assert_allclose(python.pose, cpp_pose, atol=2e-8, rtol=2e-8)
    np.testing.assert_array_equal(python.outliers, cpp_outliers)
