"""Analytical checks and optional differential checks against local C++ source."""

import os
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from orbslam3.imu import (
    Bias, ImuCalibration, ImuMeasurement, Preintegrated, frame_intervals,
    hat, inertial_jacobians, inertial_residual, inverse_right_jacobian,
    marginalize_hessian, normalize_rotation, predict_body_state,
    right_jacobian, rotation_exp, rotation_log,
)


def calibration():
    return ImuCalibration(np.eye(4), .01, .1, .001, .01)


def samples(count=60):
    for index in range(count):
        yield (np.array([np.sin(.1 * index), .2 * np.cos(.2 * index), 9.8 + .01 * index]),
               np.array([.03 + .001 * index, -.02 * np.cos(.1 * index), .05]),
               .003 + .0001 * (index % 4))


def initial_bias():
    return Bias([.01, -.02, .03], [.004, -.005, .006])


def integrated(bias=None):
    preintegrated = Preintegrated(initial_bias() if bias is None else bias, calibration())
    for acceleration, gyro, dt in samples():
        preintegrated.integrate(acceleration, gyro, dt)
    return preintegrated


class PreintegrationTests(unittest.TestCase):
    def test_calibration_noise_density_and_extrinsic_inverse(self):
        transform = np.eye(4)
        transform[:3, :3] = rotation_exp([.1, -.2, .05])
        transform[:3, 3] = [.1, -.3, .2]
        calib = ImuCalibration.from_noise_density(transform, .01, .1, .001, .01, 100.)
        assert_allclose(calib.noise_covariance.diagonal(), [.01] * 3 + [1.] * 3)
        assert_allclose(calib.walk_covariance.diagonal(), [1e-8] * 3 + [1e-6] * 3)
        assert_allclose(calib.T_camera_body @ transform, np.eye(4), atol=1e-15)
        with self.assertRaises(ValueError):
            ImuCalibration.from_noise_density(transform, .01, .1, .001, .01, 0.)

    def test_constant_acceleration_position_velocity_and_covariance(self):
        preintegrated = Preintegrated(Bias(), calibration())
        acceleration = np.array([1., -2., .3])
        for _ in range(100):
            preintegrated.integrate(acceleration, np.zeros(3), .01)
        assert_allclose(preintegrated.dt, 1.)
        assert_allclose(preintegrated.position, .5 * acceleration, atol=2e-15)
        assert_allclose(preintegrated.velocity, acceleration, atol=2e-15)
        assert_allclose(preintegrated.rotation, np.eye(3), atol=1e-15)
        assert_allclose(preintegrated.position_accel_jacobian, -.5 * np.eye(3), atol=1e-15)
        assert_allclose(preintegrated.velocity_accel_jacobian, -np.eye(3), atol=1e-15)
        assert_allclose(preintegrated.rotation_gyro_jacobian, -np.eye(3), atol=1e-15)
        assert_allclose(preintegrated.covariance[9:, 9:], 100 * calibration().walk_covariance)
        assert_allclose(preintegrated.covariance, preintegrated.covariance.T, atol=1e-18)
        self.assertGreaterEqual(np.linalg.eigvalsh(preintegrated.covariance).min(), -1e-15)

    def test_bias_jacobians_against_reintegration(self):
        base = integrated()
        epsilon = 1e-6
        for sensor in ("gyroscope", "accelerometer"):
            rotation_j = np.zeros((3, 3))
            velocity_j = np.zeros((3, 3))
            position_j = np.zeros((3, 3))
            for axis in range(3):
                plus, minus = initial_bias(), initial_bias()
                getattr(plus, sensor)[axis] += epsilon
                getattr(minus, sensor)[axis] -= epsilon
                positive, negative = integrated(plus), integrated(minus)
                rotation_j[:, axis] = rotation_log(negative.rotation.T @ positive.rotation) / (2 * epsilon)
                velocity_j[:, axis] = (positive.velocity - negative.velocity) / (2 * epsilon)
                position_j[:, axis] = (positive.position - negative.position) / (2 * epsilon)
            if sensor == "gyroscope":
                assert_allclose(rotation_j, base.rotation_gyro_jacobian, atol=2e-7)
                assert_allclose(velocity_j, base.velocity_gyro_jacobian, atol=2e-7)
                assert_allclose(position_j, base.position_gyro_jacobian, atol=2e-8)
            else:
                assert_allclose(velocity_j, base.velocity_accel_jacobian, atol=1e-9)
                assert_allclose(position_j, base.position_accel_jacobian, atol=1e-10)

    def test_bias_correction_reintegration_and_merge(self):
        preintegrated = integrated()
        new_bias = Bias([.01001, -.01999, .02999], [.00401, -.00501, .00601])
        preintegrated.set_new_bias(new_bias)
        exact = integrated(new_bias)
        for corrected, reference in zip(preintegrated.corrected_deltas(),
                                        (exact.rotation, exact.velocity, exact.position)):
            assert_allclose(corrected, reference, atol=1e-10)
        preintegrated.reintegrate()
        assert_allclose(preintegrated.rotation, exact.rotation, atol=1e-15)
        assert_allclose(preintegrated.covariance, exact.covariance, atol=1e-15)
        previous = Preintegrated(Bias(), calibration())
        current = Preintegrated(Bias(), calibration())
        for index, sample in enumerate(samples()):
            (previous if index < 20 else current).integrate(*sample)
        current.set_new_bias(new_bias)
        current.merge_previous(previous)
        assert_allclose(current.position, exact.position, atol=1e-15)
        assert_allclose(current.covariance, exact.covariance, atol=1e-15)
        copied = current.copy()
        copied.integrate([1, 2, 3], [0, 0, 0], .01)
        self.assertEqual(len(current.measurements), 60)
        current.merge_previous(current)
        self.assertEqual(len(current.measurements), 60)

    def test_frame_interval_original_interpolation(self):
        measurements = [ImuMeasurement(t, [t, 2 * t, 0], [0, 0, t])
                        for t in [0., .01, .02, .03]]
        intervals = frame_intervals(measurements, .002, .027)
        assert_allclose([interval[2] for interval in intervals], [.008, .01, .007])
        assert_allclose([interval[0][0] for interval in intervals], [.006, .015, .0235])
        one = frame_intervals(measurements[:2], .002, .009)
        assert_allclose(one[0][0], measurements[0].acceleration)
        assert_allclose(one[0][2], .007)

    def test_so3_right_jacobian_and_source_small_angle_branch(self):
        vector = np.array([.1, -.2, .3])
        assert_allclose(right_jacobian(vector) @ inverse_right_jacobian(vector), np.eye(3), atol=1e-15)
        assert_allclose(right_jacobian([1e-5, 0, 0]), np.eye(3))
        rotation = rotation_exp(vector)
        epsilon = 1e-7
        numerical = np.column_stack([
            rotation_log(rotation.T @ rotation_exp(vector + np.eye(3)[axis] * epsilon)) / epsilon
            for axis in range(3)
        ])
        assert_allclose(numerical, right_jacobian(vector), atol=5e-9)

    def test_reject_invalid_intervals_and_inputs(self):
        preintegrated = integrated()
        for dt in (0., -1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                preintegrated.integrate([0, 0, 0], [0, 0, 0], dt)
        with self.assertRaises(ValueError):
            preintegrated.integrate([0, 0, np.nan], [0, 0, 0], .01)


class InertialFactorTests(unittest.TestCase):
    def test_predicted_state_has_zero_metric_inertial_residual(self):
        preintegrated = integrated()
        rotation1, position1 = rotation_exp([.2, -.1, .3]), np.array([1., 2., 3.])
        velocity1 = np.array([.5, .1, -.2])
        rotation2, position2, velocity2 = predict_body_state(
            rotation1, position1, velocity1, preintegrated)
        assert_allclose(inertial_residual(
            preintegrated, rotation1, position1, velocity1, initial_bias(),
            rotation2, position2, velocity2), np.zeros(9), atol=2e-15)
        information = preintegrated.information()
        assert_allclose(information, information.T, atol=1e-7)
        self.assertGreater(np.linalg.eigvalsh(information).min(), 0.)

    def test_analytic_metric_and_gravity_scale_jacobians(self):
        preintegrated = integrated()
        rotation1, rotation2 = rotation_exp([.2, -.1, .3]), rotation_exp([.25, -.08, .32])
        position1, position2 = np.array([1., 2., 3.]), np.array([1.2, 2.3, 3.2])
        velocity1, velocity2 = np.array([.1, -.2, .3]), np.array([.2, -.1, .2])
        bias = Bias([.02, -.01, .04], [.005, -.006, .008])
        epsilon = 1e-6
        for with_gravity in (False, True):
            gravity = rotation_exp([.02, -.03, .01]) if with_gravity else None
            scale = 1.6 if with_gravity else 1.
            jacobians = inertial_jacobians(preintegrated, rotation1, position1, velocity1, bias,
                                           rotation2, position2, velocity2, gravity, scale)
            def residual(vertex, perturbation):
                r1, p1, v1, b = rotation1, position1, velocity1, Bias(bias.accelerometer, bias.gyroscope)
                r2, p2, v2, g, s = rotation2, position2, velocity2, gravity, scale
                if vertex == 0:
                    r1 = rotation1 @ rotation_exp(perturbation[:3])
                    p1 = position1 + rotation1 @ perturbation[3:]
                elif vertex == 1:
                    v1 = velocity1 + perturbation
                elif vertex == 2:
                    b.gyroscope += perturbation
                elif vertex == 3:
                    b.accelerometer += perturbation
                elif vertex == 4:
                    r2 = rotation2 @ rotation_exp(perturbation[:3])
                    p2 = position2 + rotation2 @ perturbation[3:]
                elif vertex == 5:
                    v2 = velocity2 + perturbation
                elif vertex == 6:
                    g = gravity @ rotation_exp(np.r_[perturbation, 0.])
                else:
                    # Source EdgeInertialGS Jacobian uses additive scale.
                    s = scale + perturbation[0]
                return inertial_residual(preintegrated, r1, p1, v1, b, r2, p2, v2, g, s)
            for vertex, jacobian in enumerate(jacobians):
                dimension = jacobian.shape[1]
                numerical = np.column_stack([
                    (residual(vertex, np.eye(dimension)[axis] * epsilon)
                     - residual(vertex, -np.eye(dimension)[axis] * epsilon)) / (2 * epsilon)
                    for axis in range(dimension)
                ])
                assert_allclose(numerical, jacobian, atol=2e-8, err_msg=f"vertex {vertex}")

    def test_schur_marginalization_retains_zero_block_and_absolute_tolerance(self):
        rng = np.random.default_rng(10)
        matrix = rng.normal(size=(12, 12))
        hessian = matrix.T @ matrix + np.eye(12)
        marginalized = marginalize_hessian(hessian, 3, 7)
        keep, removed = np.r_[0:3, 8:12], np.r_[3:8]
        expected = hessian[np.ix_(keep, keep)] - hessian[np.ix_(keep, removed)] @ np.linalg.solve(
            hessian[np.ix_(removed, removed)], hessian[np.ix_(removed, keep)])
        assert_allclose(marginalized[np.ix_(keep, keep)], expected, atol=1e-14)
        assert_allclose(marginalized[3:8], 0.)
        assert_allclose(marginalized[:, 3:8], 0.)
        degenerate = np.diag([1., 1e-7, 2.])
        assert_allclose(marginalize_hessian(degenerate, 1, 1), np.diag([1., 0., 2.]))


_CPP_REFERENCE = r'''
#include "ImuTypes.h"
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace ORB_SLAM3;
template<class Derived> void output(const Eigen::MatrixBase<Derived>& x) {
    for(int row=0;row<x.rows();row++) for(int col=0;col<x.cols();col++)
        std::cout << x(row,col) << " ";
}
int main() {
    IMU::Calib calibration(Sophus::SE3f(),.01f,.1f,.001f,.01f);
    IMU::Bias bias(.01f,-.02f,.03f,.004f,-.005f,.006f);
    IMU::Preintegrated p(bias,calibration);
    for(int i=0;i<60;i++) {
        Eigen::Vector3f a(std::sin(.1*i),.2*std::cos(.2*i),9.8+.01*i);
        Eigen::Vector3f w(.03+.001*i,-.02*std::cos(.1*i),.05);
        p.IntegrateNewMeasurement(a,w,.003+.0001*(i%4));
    }
    std::cout << std::setprecision(10) << p.dT << " ";
    output(p.dR); output(p.dV); output(p.dP); output(p.JRg); output(p.JVg);
    output(p.JVa); output(p.JPg); output(p.JPa); output(p.C);
    output(p.avgA); output(p.avgW);
}
'''


class CppPreintegrationDifferentialTests(unittest.TestCase):
    @staticmethod
    def assert_reference(testcase, actual, p):
        expected = np.concatenate([
            np.atleast_1d(p.dt), p.rotation.ravel(), p.velocity, p.position,
            p.rotation_gyro_jacobian.ravel(), p.velocity_gyro_jacobian.ravel(),
            p.velocity_accel_jacobian.ravel(), p.position_gyro_jacobian.ravel(),
            p.position_accel_jacobian.ravel(), p.covariance.ravel(),
            p.average_acceleration, p.average_angular_velocity,
        ])
        testcase.assertEqual(actual.shape, expected.shape)
        # Repeated Eigen float SVD normalizations differ from NumPy double SVD:
        # measured maxima were 4.4e-7 rotation and 2.0e-6 average acceleration.
        # Check covariance more tightly so state tolerances cannot conceal a
        # covariance-update discrepancy. See docs/inertial_audit.md.
        assert_allclose(actual[:61], expected[:61], atol=1e-6, rtol=3e-5)
        assert_allclose(actual[61:286], expected[61:286], atol=1e-8, rtol=3e-5)
        assert_allclose(actual[286:289], expected[286:289], atol=3e-6, rtol=3e-5)
        assert_allclose(actual[289:], expected[289:], atol=1e-7, rtol=3e-5)

    def test_checked_in_original_cpp_oracle(self):
        directory = Path(__file__).parent / "fixtures"
        provenance = json.loads((directory / "imu_reference.json").read_text())
        fixture = directory / "imu_reference.npz"
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         provenance["fixture_sha256"])
        self.assertEqual(hashlib.sha256(_CPP_REFERENCE.encode()).hexdigest(),
                         provenance["generator_sha256"])
        with np.load(fixture, allow_pickle=False) as data:
            bias = Bias(data["initial_bias"][3:], data["initial_bias"][:3])
            preintegrated = Preintegrated(bias, ImuCalibration(np.eye(4), *data["noise"]))
            for acceleration, gyro, dt in zip(data["acceleration"], data["angular_velocity"], data["dt"]):
                preintegrated.integrate(acceleration, gyro, dt)
            self.assert_reference(self, data["expected"], preintegrated)

    def test_original_imu_types_on_identical_measurements(self):
        if "ORBSLAM3_REFERENCE" not in os.environ:
            self.skipTest("set ORBSLAM3_REFERENCE to explicitly request optional C++ recompilation")
        reference = Path(os.environ["ORBSLAM3_REFERENCE"])
        compiler = shutil.which("g++")
        prefix = reference / ".venv"
        if not compiler or not (reference / "src/ImuTypes.cc").is_file() or not prefix.exists():
            self.skipTest("optional local C++ reference compiler/dependencies unavailable")
        with tempfile.TemporaryDirectory(prefix="orbslam3-imu-reference-") as directory:
            cpp = Path(directory) / "reference.cc"
            executable = Path(directory) / "reference"
            cpp.write_text(_CPP_REFERENCE)
            command = [compiler, "-std=c++17", "-O0", str(cpp), str(reference / "src/ImuTypes.cc"),
                       "-o", str(executable)]
            for include in (reference, reference / "include", reference / "include/CameraModels",
                            reference / "Thirdparty/Sophus", prefix / "include/eigen3",
                            prefix / "include/opencv4", prefix / "include"):
                command += ["-I", str(include)]
            command += ["-L", str(prefix / "lib"), "-lopencv_core",
                        "-Wl,-rpath," + str(prefix / "lib")]
            built = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            output = subprocess.check_output([str(executable)], text=True)
        actual = np.fromstring(output, sep=" ")
        self.assert_reference(self, actual, integrated())


if __name__ == "__main__":
    unittest.main()
