"""ORB-SLAM3 IMU preintegration and inertial factors.

Ported from ``src/ImuTypes.cc`` and ``src/G2oTypes.cc`` at the recorded
reference revision. Body poses map body coordinates into world coordinates;
camera calibration ``T_body_camera`` maps camera into body coordinates.
The preintegration error order is rotation, velocity, position, gyro bias,
accelerometer bias. Numerical operations use float64 rather than C++ float.

Copyright (C) 2017-2021 the ORB-SLAM3 authors, University of Zaragoza.
Copyright (C) 2014-2016 the ORB-SLAM2 authors, University of Zaragoza.
Python translation licensed under GPL-3.0-or-later; see LICENSE and NOTICE.
"""

from dataclasses import dataclass, field
import copy
import numpy as np

GRAVITY = 9.81
_EPS = 1e-4


def _vector3(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("expected a finite three-vector")
    return value.copy()


def hat(vector):
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def normalize_rotation(rotation):
    """The source uses U V^T without an extra determinant correction."""
    u, _, vt = np.linalg.svd(rotation)
    return u @ vt


def rotation_exp(vector):
    """Rodrigues exponential, with a stable series near zero."""
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    skew = hat(vector)
    if angle < 1e-8:
        return np.eye(3) + skew + .5 * skew @ skew
    return np.eye(3) + np.sin(angle) / angle * skew + (
        (1. - np.cos(angle)) / angle**2 * skew @ skew
    )


def rotation_log(rotation):
    """G2oTypes.cc LogSO3, including its small-sine branch."""
    cosine = (np.trace(rotation) - 1.) * .5
    vector = np.array([rotation[2, 1] - rotation[1, 2],
                       rotation[0, 2] - rotation[2, 0],
                       rotation[1, 0] - rotation[0, 1]]) * .5
    if cosine > 1. or cosine < -1.:
        return vector
    angle = np.arccos(cosine)
    sine = np.sin(angle)
    return vector if abs(sine) < 1e-5 else angle / sine * vector


def right_jacobian(vector, epsilon=_EPS):
    angle = np.linalg.norm(vector)
    if angle < epsilon:
        return np.eye(3)
    skew = hat(vector)
    return (np.eye(3) - (1. - np.cos(angle)) / angle**2 * skew
            + (angle - np.sin(angle)) / angle**3 * skew @ skew)


def inverse_right_jacobian(vector, epsilon=_EPS):
    angle = np.linalg.norm(vector)
    if angle < epsilon:
        return np.eye(3)
    skew = hat(vector)
    return (np.eye(3) + .5 * skew +
            (1. / angle**2 - (1. + np.cos(angle)) /
             (2. * angle * np.sin(angle))) * skew @ skew)


@dataclass
class Bias:
    accelerometer: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyroscope: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.accelerometer = _vector3(self.accelerometer)
        self.gyroscope = _vector3(self.gyroscope)

    def vector(self):
        """Optimizer order: gyroscope first, accelerometer second."""
        return np.r_[self.gyroscope, self.accelerometer]


@dataclass
class ImuMeasurement:
    timestamp: float
    acceleration: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.timestamp):
            raise ValueError("IMU timestamp must be finite")
        self.acceleration = _vector3(self.acceleration)
        self.angular_velocity = _vector3(self.angular_velocity)


@dataclass
class ImuCalibration:
    """Discrete sample standard deviations, matching IMU::Calib inputs.

    Use ``from_noise_density`` for the continuous-time noise-density values
    normally supplied to ORB-SLAM3 configuration files. Bias random-walk
    standard deviations already include division by sqrt(sample frequency).
    """
    T_body_camera: np.ndarray
    gyro_noise: float
    accel_noise: float
    gyro_walk: float
    accel_walk: float

    def __post_init__(self):
        self.T_body_camera = np.asarray(self.T_body_camera, dtype=float).copy()
        if self.T_body_camera.shape != (4, 4) or not np.all(
                np.isfinite(self.T_body_camera)):
            raise ValueError("T_body_camera must be a finite 4x4 transform")
        rotation = self.T_body_camera[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)
                or not np.allclose(self.T_body_camera[3], [0, 0, 0, 1])):
            raise ValueError("T_body_camera must be a rigid transform")
        noise = [self.gyro_noise, self.accel_noise, self.gyro_walk, self.accel_walk]
        if not np.all(np.isfinite(noise)) or np.any(np.asarray(noise) < 0):
            raise ValueError("IMU noise standard deviations must be nonnegative")

    @classmethod
    def from_noise_density(cls, T_body_camera, gyro_noise, accel_noise,
                           gyro_walk, accel_walk, frequency):
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("IMU sample frequency must be positive")
        scale = np.sqrt(frequency)
        return cls(T_body_camera, gyro_noise * scale, accel_noise * scale,
                   gyro_walk / scale, accel_walk / scale)

    @property
    def T_camera_body(self):
        rotation = self.T_body_camera[:3, :3]
        inverse = np.eye(4)
        inverse[:3, :3] = rotation.T
        inverse[:3, 3] = -rotation.T @ self.T_body_camera[:3, 3]
        return inverse

    @property
    def noise_covariance(self):
        return np.diag(np.r_[np.full(3, self.gyro_noise**2),
                             np.full(3, self.accel_noise**2)])

    @property
    def walk_covariance(self):
        return np.diag(np.r_[np.full(3, self.gyro_walk**2),
                             np.full(3, self.accel_walk**2)])


def integrated_rotation(angular_velocity, bias, dt):
    vector = (np.asarray(angular_velocity) - bias.gyroscope) * dt
    skew = hat(vector)
    # Keep the source's first-order rotation branch; normalize after composition.
    if np.linalg.norm(vector) < _EPS:
        return np.eye(3) + skew, np.eye(3)
    return rotation_exp(vector), right_jacobian(vector)


class Preintegrated:
    """Discrete preintegration from ImuTypes.cc, with retained raw samples.

    ``integrate`` consumes the average acceleration/angular velocity during
    a positive time interval. Timestamp interpolation belongs outside this
    operation; see ``frame_intervals`` for the original Tracking.cc logic.
    """

    def __init__(self, bias, calibration):
        self.calibration = copy.deepcopy(calibration)
        self.noise_covariance = calibration.noise_covariance
        self.walk_covariance = calibration.walk_covariance
        self.initialize(bias)

    def initialize(self, bias):
        self.bias = copy.deepcopy(bias)
        self.updated_bias = copy.deepcopy(bias)
        self.delta_bias = np.zeros(6)
        self.dt = 0.
        self.rotation = np.eye(3)
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.rotation_gyro_jacobian = np.zeros((3, 3))
        self.velocity_gyro_jacobian = np.zeros((3, 3))
        self.velocity_accel_jacobian = np.zeros((3, 3))
        self.position_gyro_jacobian = np.zeros((3, 3))
        self.position_accel_jacobian = np.zeros((3, 3))
        self.covariance = np.zeros((15, 15))
        self.average_acceleration = np.zeros(3)
        self.average_angular_velocity = np.zeros(3)
        self.measurements = []

    def copy(self):
        return copy.deepcopy(self)

    def integrate(self, acceleration, angular_velocity, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("preintegration interval must be finite and positive")
        acceleration, angular_velocity = _vector3(acceleration), _vector3(angular_velocity)
        self.measurements.append((acceleration.copy(), angular_velocity.copy(), float(dt)))
        accel = acceleration - self.bias.accelerometer
        gyro = angular_velocity - self.bias.gyroscope
        rotation = self.rotation
        self.average_acceleration = (
            self.dt * self.average_acceleration + rotation @ accel * dt
        ) / (self.dt + dt)
        self.average_angular_velocity = (
            self.dt * self.average_angular_velocity + gyro * dt
        ) / (self.dt + dt)

        self.position += self.velocity * dt + .5 * rotation @ accel * dt**2
        self.velocity += rotation @ accel * dt

        transition = np.eye(9)
        noise_jacobian = np.zeros((9, 6))
        accel_skew = hat(accel)
        transition[3:6, :3] = -rotation @ accel_skew * dt
        transition[6:9, :3] = -.5 * rotation @ accel_skew * dt**2
        transition[6:9, 3:6] = np.eye(3) * dt
        noise_jacobian[3:6, 3:6] = rotation * dt
        noise_jacobian[6:9, 3:6] = .5 * rotation * dt**2

        # Both position Jacobians must use the old velocity Jacobians.
        self.position_accel_jacobian += self.velocity_accel_jacobian * dt - .5 * rotation * dt**2
        self.position_gyro_jacobian += (
            self.velocity_gyro_jacobian * dt - .5 * rotation @ accel_skew @
            self.rotation_gyro_jacobian * dt**2
        )
        self.velocity_accel_jacobian -= rotation * dt
        self.velocity_gyro_jacobian -= rotation @ accel_skew @ self.rotation_gyro_jacobian * dt

        delta_rotation, right_j = integrated_rotation(angular_velocity, self.bias, dt)
        self.rotation = normalize_rotation(rotation @ delta_rotation)
        transition[:3, :3] = delta_rotation.T
        noise_jacobian[:3, :3] = right_j * dt
        self.covariance[:9, :9] = (
            transition @ self.covariance[:9, :9] @ transition.T
            + noise_jacobian @ self.noise_covariance @ noise_jacobian.T
        )
        # This deliberately has no dt multiplier; Calib inputs are discrete.
        self.covariance[9:, 9:] += self.walk_covariance
        self.rotation_gyro_jacobian = delta_rotation.T @ self.rotation_gyro_jacobian - right_j * dt
        self.dt += dt

    def set_new_bias(self, bias):
        self.updated_bias = copy.deepcopy(bias)
        self.delta_bias = bias.vector() - self.bias.vector()

    def corrected_deltas(self, bias=None):
        """Return rotation, velocity, position with first-order bias correction."""
        if bias is None:
            bias = self.updated_bias
        delta_gyro = bias.gyroscope - self.bias.gyroscope
        delta_accel = bias.accelerometer - self.bias.accelerometer
        rotation = normalize_rotation(self.rotation @ rotation_exp(
            self.rotation_gyro_jacobian @ delta_gyro))
        velocity = (self.velocity + self.velocity_gyro_jacobian @ delta_gyro
                    + self.velocity_accel_jacobian @ delta_accel)
        position = (self.position + self.position_gyro_jacobian @ delta_gyro
                    + self.position_accel_jacobian @ delta_accel)
        return rotation, velocity, position

    def reintegrate(self):
        samples = self.measurements.copy()
        self.initialize(self.updated_bias)
        for acceleration, angular_velocity, dt in samples:
            self.integrate(acceleration, angular_velocity, dt)

    def merge_previous(self, previous):
        if previous is self:
            return
        samples = previous.measurements + self.measurements
        self.initialize(self.updated_bias)
        for acceleration, angular_velocity, dt in samples:
            self.integrate(acceleration, angular_velocity, dt)

    def information(self):
        """Information matrix used by EdgeInertial and EdgeInertialGS.

        A degenerate covariance raises LinAlgError instead of propagating the
        NaNs produced by an unchecked inverse in the original implementation.
        """
        information = np.linalg.inv(self.covariance[:9, :9])
        information = .5 * (information + information.T)
        eigenvalues, eigenvectors = np.linalg.eigh(information)
        eigenvalues[eigenvalues < 1e-12] = 0.
        return (eigenvectors * eigenvalues) @ eigenvectors.T


def frame_intervals(measurements, previous_timestamp, current_timestamp):
    """Tracking.cc:1685 boundary interpolation after queue selection.

    ``measurements`` must be the samples selected by the original queue rule:
    discard samples before previous_timestamp - 0.001; consume samples before
    current_timestamp - 0.001 and retain the first sample at/after that bound.
    This function preserves the original first/last-interval formulas, including
    the single-interval use of the first measurement without averaging.
    """
    if not np.isfinite(previous_timestamp + current_timestamp) or current_timestamp <= previous_timestamp:
        raise ValueError("frame timestamps must be finite and increasing")
    if len(measurements) < 2:
        raise ValueError("at least two selected IMU samples are required")
    if any(b.timestamp <= a.timestamp for a, b in zip(measurements, measurements[1:])):
        raise ValueError("IMU sample timestamps must be strictly increasing")
    count = len(measurements) - 1
    intervals = []
    for index in range(count):
        first, second = measurements[index:index + 2]
        if index == 0 and index < count - 1:
            fraction = (first.timestamp - previous_timestamp) / (second.timestamp - first.timestamp)
            acceleration = .5 * (first.acceleration + second.acceleration -
                                  (second.acceleration - first.acceleration) * fraction)
            gyro = .5 * (first.angular_velocity + second.angular_velocity -
                        (second.angular_velocity - first.angular_velocity) * fraction)
            dt = second.timestamp - previous_timestamp
        elif index < count - 1:
            acceleration = .5 * (first.acceleration + second.acceleration)
            gyro = .5 * (first.angular_velocity + second.angular_velocity)
            dt = second.timestamp - first.timestamp
        elif index > 0:
            fraction = (second.timestamp - current_timestamp) / (second.timestamp - first.timestamp)
            acceleration = .5 * (first.acceleration + second.acceleration -
                                  (second.acceleration - first.acceleration) * fraction)
            gyro = .5 * (first.angular_velocity + second.angular_velocity -
                        (second.angular_velocity - first.angular_velocity) * fraction)
            dt = current_timestamp - first.timestamp
        else:
            acceleration, gyro = first.acceleration, first.angular_velocity
            dt = current_timestamp - previous_timestamp
        if dt <= 0:
            raise ValueError("selected IMU samples do not bracket the frame interval")
        intervals.append((acceleration, gyro, dt))
    return intervals


def predict_body_state(rotation, position, velocity, preintegrated, bias=None,
                       gravity=None):
    """Tracking::PredictStateIMU propagation from frame or keyframe state."""
    gravity = np.array([0., 0., -GRAVITY]) if gravity is None else _vector3(gravity)
    delta_rotation, delta_velocity, delta_position = preintegrated.corrected_deltas(bias)
    dt = preintegrated.dt
    return (normalize_rotation(rotation @ delta_rotation),
            position + velocity * dt + .5 * gravity * dt**2 + rotation @ delta_position,
            velocity + gravity * dt + rotation @ delta_velocity)


def inertial_residual(preintegrated, rotation1, position1, velocity1, bias1,
                      rotation2, position2, velocity2, gravity_rotation=None,
                      scale=1.):
    """Nine-dimensional EdgeInertial/EdgeInertialGS residual.

    With gravity_rotation supplied, velocities and positions use the visual
    scale exactly as in EdgeInertialGS; metric tracking uses scale=1.
    """
    gravity = np.array([0., 0., -GRAVITY])
    if gravity_rotation is not None:
        gravity = gravity_rotation @ gravity
    d_rotation, d_velocity, d_position = preintegrated.corrected_deltas(bias1)
    dt, body_world = preintegrated.dt, rotation1.T
    return np.r_[
        rotation_log(d_rotation.T @ body_world @ rotation2),
        body_world @ (scale * (velocity2 - velocity1) - gravity * dt) - d_velocity,
        body_world @ (scale * (position2 - position1 - velocity1 * dt)
                      - .5 * gravity * dt**2) - d_position,
    ]


def inertial_jacobians(preintegrated, rotation1, position1, velocity1, bias1,
                       rotation2, position2, velocity2, gravity_rotation=None,
                       scale=1.):
    """Analytic G2oTypes.cc EdgeInertial{GS} vertex Jacobians.

    Pose perturbations are (body rotation, body translation):
    Rwb <- Rwb Exp(dr), twb <- twb + Rwb dt. Velocity/bias increments
    are additive world/body vectors. Gravity has two right rotation DOFs.
    The source scale factor Jacobian is with respect to additive scale;
    VertexScale updates multiplicatively, an inherited inconsistency recorded
    in docs/inertial_audit.md. Return order: pose1, velocity1, gyro1, accel1,
    pose2, velocity2, and (when GS enabled) gravity, scale.
    """
    gravity = np.array([0., 0., -GRAVITY])
    if gravity_rotation is not None:
        gravity = gravity_rotation @ gravity
    dt, body_world = preintegrated.dt, rotation1.T
    d_rotation = preintegrated.corrected_deltas(bias1)[0]
    error_rotation = d_rotation.T @ body_world @ rotation2
    error = rotation_log(error_rotation)
    # G2oTypes.cc uses its double-precision Jacobian epsilon (1e-5).
    inverse_j = inverse_right_jacobian(error, epsilon=1e-5)
    pose1, pose2 = np.zeros((9, 6)), np.zeros((9, 6))
    velocity1_j, velocity2_j = np.zeros((9, 3)), np.zeros((9, 3))
    gyro_j, accel_j = np.zeros((9, 3)), np.zeros((9, 3))
    pose1[:3, :3] = -inverse_j @ rotation2.T @ rotation1
    pose1[3:6, :3] = hat(body_world @ (scale * (velocity2 - velocity1) - gravity * dt))
    pose1[6:9, :3] = hat(body_world @ (
        scale * (position2 - position1 - velocity1 * dt) - .5 * gravity * dt**2))
    pose1[6:9, 3:] = -scale * np.eye(3)
    velocity1_j[3:6] = -scale * body_world
    velocity1_j[6:9] = -scale * body_world * dt
    gyro_delta = bias1.gyroscope - preintegrated.bias.gyroscope
    gyro_j[:3] = (-inverse_j @ error_rotation.T @ right_jacobian(
        preintegrated.rotation_gyro_jacobian @ gyro_delta, epsilon=1e-5)
        @ preintegrated.rotation_gyro_jacobian)
    gyro_j[3:6] = -preintegrated.velocity_gyro_jacobian
    gyro_j[6:9] = -preintegrated.position_gyro_jacobian
    accel_j[3:6] = -preintegrated.velocity_accel_jacobian
    accel_j[6:9] = -preintegrated.position_accel_jacobian
    pose2[:3, :3] = inverse_j
    pose2[6:9, 3:] = scale * body_world @ rotation2
    velocity2_j[3:6] = scale * body_world
    result = (pose1, velocity1_j, gyro_j, accel_j, pose2, velocity2_j)
    if gravity_rotation is not None:
        gravity_j = np.zeros((9, 2))
        gravity_derivative = gravity_rotation @ np.array([[0., -GRAVITY], [GRAVITY, 0.], [0., 0.]])
        gravity_j[3:6] = -body_world @ gravity_derivative * dt
        gravity_j[6:9] = -.5 * body_world @ gravity_derivative * dt**2
        scale_j = np.zeros((9, 1))
        scale_j[3:6, 0] = body_world @ (velocity2 - velocity1)
        scale_j[6:9, 0] = body_world @ (position2 - position1 - velocity1 * dt)
        result += (gravity_j, scale_j)
    return result


def bias_random_walk_residual(previous_bias, current_bias):
    """EdgeGyroRW/EdgeAccRW combined, gyro first."""
    return current_bias.vector() - previous_bias.vector()


def marginalize_hessian(hessian, start, end):
    """Optimizer::Marginalize inclusive index range, retaining zero blocks.

    Singular values <=1e-6 are discarded exactly as in the source; this is
    not a relative pseudo-inverse tolerance.
    """
    hessian = np.asarray(hessian, dtype=float)
    if (hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]
            or not 0 <= start <= end < hessian.shape[0]):
        raise ValueError("expected square Hessian and valid inclusive block")
    keep = np.r_[np.arange(start), np.arange(end + 1, hessian.shape[0])]
    removed = np.arange(start, end + 1)
    u, singular, vt = np.linalg.svd(hessian[np.ix_(removed, removed)])
    inverse_values = np.zeros_like(singular)
    np.divide(1., singular, out=inverse_values, where=singular > 1e-6)
    inverse = (vt.T * inverse_values) @ u.T
    result = np.zeros_like(hessian)
    result[np.ix_(keep, keep)] = (
        hessian[np.ix_(keep, keep)] - hessian[np.ix_(keep, removed)] @
        inverse @ hessian[np.ix_(removed, keep)]
    )
    return result
