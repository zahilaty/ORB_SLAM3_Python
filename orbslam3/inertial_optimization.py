"""Readable visual-inertial graphs from ORB-SLAM3 Optimizer.cc.

Body poses map body into world coordinates. Pose increments are right body
increments, velocity/bias/landmark increments are additive, gravity has two
right rotation DOFs, and scale updates multiplicatively. Factor blocks retain
the original information matrices, Huber objectives, and variable selection.
SciPy sparse linear algebra replaces Eigen's numerical factorization.

Derived from ORB-SLAM3, Copyright (C) 2017-2021 Carlos Campos, Richard Elvira,
Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of
Zaragoza; Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D.
Tardós, University of Zaragoza. GPL-3.0-or-later; see LICENSE and NOTICE.
"""

from dataclasses import dataclass, field
import copy
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .imu import (
    Bias, Preintegrated, GRAVITY, hat, rotation_exp, rotation_log,
    inverse_right_jacobian, inertial_residual, inertial_jacobians,
    marginalize_hessian,
)
from .optimization import _levenberg_marquardt, huber, MONO_CHI2, STEREO_CHI2


@dataclass
class BodyState:
    rotation: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    bias: Bias = field(default_factory=Bias)

    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, float).copy()
        self.position = np.asarray(self.position, float).copy()
        self.velocity = np.asarray(self.velocity, float).copy()
        self.bias = copy.deepcopy(self.bias)
        if (self.rotation.shape != (3, 3) or self.position.shape != (3,)
                or self.velocity.shape != (3,)):
            raise ValueError("body state requires 3x3 rotation and three-vector position/velocity")
        if not all(np.all(np.isfinite(value)) for value in
                   (self.rotation, self.position, self.velocity)):
            raise ValueError("body state must be finite")


@dataclass(frozen=True)
class InertialLink:
    previous: int
    current: int
    preintegrated: Preintegrated


@dataclass
class StatePrior:
    """ConstraintPoseImu: pose, velocity, gyro bias, accel bias, information."""
    state: BodyState
    information: np.ndarray

    def __post_init__(self):
        self.state = copy.deepcopy(self.state)
        self.information = np.asarray(self.information, float).copy()
        if self.information.shape != (15, 15):
            raise ValueError("inertial state prior requires a 15x15 information matrix")


@dataclass
class InertialOptimizationResult:
    states: dict
    points: dict
    gravity_rotation: np.ndarray
    scale: float
    initial_cost: float
    final_cost: float
    iterations: int
    chi2: np.ndarray = field(default_factory=lambda: np.empty(0))
    outliers: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    accepted: bool = True
    # Applying bias/reintegration changes to an actual map is explicit.
    reintegrate_ids: tuple = ()
    prior: StatePrior | None = None

    @property
    def inliers(self):
        return int(np.count_nonzero(~self.outliers))


@dataclass
class _Factor:
    keys: tuple
    information: np.ndarray
    evaluate: object
    huber_squared: float | None = None
    active: bool = True


def _state_parameters(states, points=None, shared_bias=False):
    parameters = {}
    for identifier, state in states.items():
        parameters["pose", identifier] = (state.rotation.copy(), state.position.copy())
        parameters["velocity", identifier] = state.velocity.copy()
        if not shared_bias:
            parameters["gyro", identifier] = state.bias.gyroscope.copy()
            parameters["accel", identifier] = state.bias.accelerometer.copy()
    if shared_bias and states:
        bias = next(iter(states.values())).bias
        parameters["gyro", "shared"] = bias.gyroscope.copy()
        parameters["accel", "shared"] = bias.accelerometer.copy()
    for identifier, point in (points or {}).items():
        parameters["point", identifier] = np.asarray(point, float).copy()
    return parameters


def _extract_states(parameters, states, shared_bias=False):
    result = {}
    for identifier in states:
        rotation, position = parameters["pose", identifier]
        bias_id = "shared" if shared_bias else identifier
        result[identifier] = BodyState(rotation, position, parameters["velocity", identifier],
                                       Bias(parameters["accel", bias_id], parameters["gyro", bias_id]))
    return result


def _dimension(key):
    return {"pose": 6, "gravity": 2, "scale": 1}.get(key[0], 3)


def _retract(parameters, order, offsets, increment):
    updated = parameters.copy()
    for key in order:
        start = offsets[key]
        delta = increment[start:start + _dimension(key)]
        if key[0] == "pose":
            rotation, position = parameters[key]
            updated[key] = (rotation @ rotation_exp(delta[:3]), position + rotation @ delta[3:])
        elif key[0] == "gravity":
            updated[key] = parameters[key] @ rotation_exp(np.r_[delta, 0.])
        elif key[0] == "scale":
            updated[key] = float(parameters[key] * np.exp(delta[0]))
        else:
            updated[key] = parameters[key] + delta
    return updated


def _graph_layout(parameters, fixed):
    # Keep landmark blocks last so the Schur partition remains apparent.
    order = tuple(key for key in parameters if key not in fixed and key[0] != "point")
    state_dimension = sum(_dimension(key) for key in order)
    order += tuple(key for key in parameters if key not in fixed and key[0] == "point")
    offsets, dimension = {}, 0
    for key in order:
        offsets[key] = dimension
        dimension += _dimension(key)
    return order, offsets, state_dimension, dimension


def _normal_equations(parameters, factors, offsets, dimension, robust=True):
    blocks, rhs = {}, np.zeros(dimension)
    for factor in factors:
        if not factor.active:
            continue
        residual, jacobians = factor.evaluate(parameters)
        chi2 = float(residual @ factor.information @ residual)
        weight = (huber(chi2, factor.huber_squared)[1]
                  if robust and factor.huber_squared is not None else 1.)
        information = factor.information * weight
        free = [(key, jacobian) for key, jacobian in zip(factor.keys, jacobians) if key in offsets]
        for key, jacobian in free:
            start, width = offsets[key], _dimension(key)
            rhs[start:start + width] -= jacobian.T @ information @ residual
            for other, other_jacobian in free:
                block = jacobian.T @ information @ other_jacobian
                pair = key, other
                blocks[pair] = blocks.get(pair, 0.) + block
    rows, columns, values = [], [], []
    for (key, other), block in blocks.items():
        row, column = np.indices(block.shape)
        rows.extend((row + offsets[key]).ravel())
        columns.extend((column + offsets[other]).ravel())
        values.extend(block.ravel())
    matrix = sparse.coo_matrix((values, (rows, columns)), shape=(dimension, dimension)).tocsr()
    return matrix, rhs


def _graph_cost(parameters, factors):
    total = 0.
    for factor in factors:
        if factor.active:
            residual, _ = factor.evaluate(parameters)
            chi2 = float(residual @ factor.information @ residual)
            total += huber(chi2, factor.huber_squared)[0] if factor.huber_squared is not None else chi2
    return total


def _solve_schur(normal, damping, state_dimension):
    """Damp first, eliminate independent 3D landmarks, solve retained states."""
    matrix, rhs = normal
    dimension = matrix.shape[0]
    matrix = matrix + sparse.eye(dimension, format="csr") * damping
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        if state_dimension == dimension:
            increment = spsolve(matrix.tocsc(), rhs)
        else:
            point_count = (dimension - state_dimension) // 3
            inverse_blocks = [np.linalg.inv(matrix[
                state_dimension + 3 * index:state_dimension + 3 * index + 3,
                state_dimension + 3 * index:state_dimension + 3 * index + 3].toarray())
                for index in range(point_count)]
            inverse_points = sparse.block_diag(inverse_blocks, format="csr")
            if state_dimension == 0:
                increment = inverse_points @ rhs
            else:
                h_ss = matrix[:state_dimension, :state_dimension]
                h_sp = matrix[:state_dimension, state_dimension:]
                h_ps = matrix[state_dimension:, :state_dimension]
                reduced = h_ss - h_sp @ inverse_points @ h_ps
                reduced_rhs = rhs[:state_dimension] - h_sp @ (inverse_points @ rhs[state_dimension:])
                state_delta = spsolve(reduced.tocsc(), reduced_rhs)
                point_delta = inverse_points @ (rhs[state_dimension:] - h_ps @ state_delta)
                increment = np.r_[state_delta, point_delta]
    if not np.all(np.isfinite(increment)):
        raise np.linalg.LinAlgError("inertial graph normal equations are singular")
    return increment, rhs


def _optimize(parameters, factors, fixed, iterations, initial_lambda=None, gauss_newton=False):
    order, offsets, state_dimension, dimension = _graph_layout(parameters, fixed)
    initial_cost = _graph_cost(parameters, factors)
    if dimension == 0 or iterations == 0:
        return parameters, initial_cost, initial_cost, 0

    def assemble(current):
        normal = _normal_equations(current, factors, offsets, dimension)
        largest = max(float(np.max(np.abs(normal[0].diagonal()))), 1e-20)
        return normal, largest

    def solve(normal, damping):
        try:
            return _solve_schur(normal, damping, state_dimension)
        except MatrixRankWarning as error:
            raise np.linalg.LinAlgError(str(error)) from error

    def retract(current, increment):
        with np.errstate(over="raise", invalid="raise"):
            return _retract(current, order, offsets, increment)

    if gauss_newton:
        current, completed = parameters, 0
        for _ in range(iterations):
            normal, _ = assemble(current)
            delta, _ = solve(normal, 0.)
            current = retract(current, delta)
            completed += 1
        return current, initial_cost, _graph_cost(current, factors), completed
    return _levenberg_marquardt(parameters, assemble, solve, retract,
                                lambda current: _graph_cost(current, factors),
                                iterations, initial_lambda)


def body_projection_factor(state, point, observation, calibration):
    """G2oTypes.cc EdgeMono/EdgeStereo using right body-pose increments.

    ``observation`` is optimization.ReprojectionObservation. Right-camera
    factors compose its camera_from_reference with the left IMU extrinsic.
    """
    rotation, position = state.rotation, state.position
    camera_body = calibration.T_camera_body
    if observation.camera_from_reference is not None:
        camera_body = observation.camera_from_reference @ camera_body
    rotation_cb, translation_cb = camera_body[:3, :3], camera_body[:3, 3]
    point_body = rotation.T @ (point - position)
    point_camera = rotation_cb @ point_body + translation_cb
    prediction = np.asarray(observation.camera.project(point_camera), float)
    jacobian = observation.camera.project_jacobian(point_camera)
    if observation.measurement.size == 3:
        prediction = np.r_[prediction, prediction[0] - observation.baseline_fx / point_camera[2]]
        right = jacobian[0].copy()
        right[2] += observation.baseline_fx / point_camera[2]**2
        jacobian = np.vstack([jacobian, right])
    pose_jacobian = jacobian @ rotation_cb @ np.c_[-hat(point_body), np.eye(3)]
    point_jacobian = -jacobian @ rotation_cb @ rotation.T
    return observation.measurement - prediction, pose_jacobian, point_jacobian, point_camera[2]


def _visual_factor(observation, calibration):
    keys = (("pose", observation.pose_id), ("point", observation.point_id))

    def evaluate(parameters):
        rotation, position = parameters[keys[0]]
        state = BodyState(rotation, position, np.zeros(3))
        residual, pose_j, point_j, _ = body_projection_factor(state, parameters[keys[1]], observation, calibration)
        return residual, (pose_j, point_j)

    return _Factor(keys, np.eye(observation.measurement.size) * observation.inverse_variance,
                   evaluate, MONO_CHI2 if observation.measurement.size == 2 else STEREO_CHI2)


def _inertial_factor(link, shared_bias=False, gravity_scale=False, robust=None, downweight=1.):
    bias_id = "shared" if shared_bias else link.previous
    keys = (("pose", link.previous), ("velocity", link.previous), ("gyro", bias_id),
            ("accel", bias_id), ("pose", link.current), ("velocity", link.current))
    if gravity_scale:
        keys += (("gravity", 0), ("scale", 0))

    def evaluate(parameters):
        rotation1, position1 = parameters[keys[0]]
        rotation2, position2 = parameters[keys[4]]
        arguments = (link.preintegrated, rotation1, position1, parameters[keys[1]],
                     Bias(parameters[keys[3]], parameters[keys[2]]), rotation2, position2, parameters[keys[5]])
        keywords = ({"gravity_rotation": parameters["gravity", 0], "scale": parameters["scale", 0]}
                    if gravity_scale else {})
        # Source EdgeInertialGS uses additive-scale derivative and log-scale
        # retraction. Preserve that inherited discrepancy, documented in audit.
        return inertial_residual(*arguments, **keywords), inertial_jacobians(*arguments, **keywords)

    return _Factor(keys, link.preintegrated.information() * downweight, evaluate, robust)


def _walk_factors(link):
    factors = []
    for kind, start in (("gyro", 9), ("accel", 12)):
        keys = ((kind, link.previous), (kind, link.current))

        def evaluate(parameters, keys=keys):
            return parameters[keys[1]] - parameters[keys[0]], (-np.eye(3), np.eye(3))

        factors.append(_Factor(keys, np.linalg.inv(link.preintegrated.covariance[start:start + 3, start:start + 3]), evaluate))
    return factors


def _bias_prior_factor(kind, strength):
    key = kind, "shared"
    return _Factor((key,), np.eye(3) * strength,
                   lambda parameters: (parameters[key], (np.eye(3),)))


def state_prior_factor(state, prior):
    """Return EdgePriorPoseImu residual/Jacobians in its 15-state order."""
    reference = prior.state
    rotation_error = rotation_log(reference.rotation.T @ state.rotation)
    residual = np.r_[rotation_error, reference.rotation.T @ (state.position - reference.position),
                     state.velocity - reference.velocity,
                     state.bias.gyroscope - reference.bias.gyroscope,
                     state.bias.accelerometer - reference.bias.accelerometer]
    pose_j = np.zeros((15, 6))
    pose_j[:3, :3] = inverse_right_jacobian(rotation_error, epsilon=1e-5)
    pose_j[3:6, 3:6] = reference.rotation.T @ state.rotation
    velocity_j, gyro_j, accel_j = np.zeros((15, 3)), np.zeros((15, 3)), np.zeros((15, 3))
    velocity_j[6:9], gyro_j[9:12], accel_j[12:15] = np.eye(3), np.eye(3), np.eye(3)
    return residual, (pose_j, velocity_j, gyro_j, accel_j)


def _prior_factor(identifier, prior):
    keys = (("pose", identifier), ("velocity", identifier), ("gyro", identifier), ("accel", identifier))

    def evaluate(parameters):
        rotation, position = parameters[keys[0]]
        state = BodyState(rotation, position, parameters[keys[1]], Bias(parameters[keys[3]], parameters[keys[2]]))
        return state_prior_factor(state, prior)

    return _Factor(keys, prior.information, evaluate, 25.)


def _calibration_for(calibrations, identifier):
    return calibrations[identifier] if isinstance(calibrations, dict) else calibrations


def _visual_diagnostics(parameters, observations, calibrations):
    chi2, depth = np.empty(len(observations)), np.empty(len(observations))
    for index, observation in enumerate(observations):
        rotation, position = parameters["pose", observation.pose_id]
        state = BodyState(rotation, position, np.zeros(3))
        residual, _, _, depth[index] = body_projection_factor(
            state, parameters["point", observation.point_id], observation,
            _calibration_for(calibrations, observation.pose_id))
        chi2[index] = observation.inverse_variance * (residual @ residual)
    return chi2, depth


def optimize_inertial_initialization(states, links, monocular=True, gravity_rotation=None,
                                     scale=1., fixed_velocity=False, prior_gyro=0.,
                                     prior_accel=0., iterations=200):
    """Optimizer::InertialOptimization with fixed poses and common biases.

    All supplied poses are fixed. Velocities and common biases are optimized
    unless fixed_velocity; gravity direction is free and stereo scale fixed.
    Map updates/reintegration are returned as decisions, without mutating input.
    """
    if not states:
        raise ValueError("inertial initialization requires states")
    parameters = _state_parameters(states, shared_bias=True)
    parameters["gravity", 0] = np.eye(3) if gravity_rotation is None else np.asarray(gravity_rotation, float).copy()
    parameters["scale", 0] = float(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    fixed = {("pose", identifier) for identifier in states}
    if fixed_velocity:
        fixed |= {("velocity", identifier) for identifier in states}
        fixed |= {("gyro", "shared"), ("accel", "shared")}
    if not monocular:
        fixed.add(("scale", 0))
    factors = [_inertial_factor(link, shared_bias=True, gravity_scale=True) for link in links]
    factors += [_bias_prior_factor("gyro", prior_gyro), _bias_prior_factor("accel", prior_accel)]
    parameters, initial, final, completed = _optimize(
        parameters, factors, fixed, iterations, 1e3 if prior_gyro != 0 else None)
    updated_states = _extract_states(parameters, states, shared_bias=True)
    reintegrate = tuple(identifier for identifier, state in states.items()
                        if np.linalg.norm(state.bias.gyroscope - updated_states[identifier].bias.gyroscope) > .01)
    return InertialOptimizationResult(updated_states, {}, parameters["gravity", 0], parameters["scale", 0],
                                      initial, final, completed, reintegrate_ids=reintegrate)


def refine_inertial_biases(states, links, prior_gyro=0., prior_accel=0., iterations=200):
    """Bias-only overload: fixed poses/gravity/scale, free velocities/common bias."""
    parameters = _state_parameters(states, shared_bias=True)
    parameters["gravity", 0], parameters["scale", 0] = np.eye(3), 1.
    fixed = {("pose", identifier) for identifier in states} | {("gravity", 0), ("scale", 0)}
    factors = [_inertial_factor(link, shared_bias=True, gravity_scale=True) for link in links]
    factors += [_bias_prior_factor("gyro", prior_gyro), _bias_prior_factor("accel", prior_accel)]
    parameters, initial, final, completed = _optimize(parameters, factors, fixed, iterations, 1e3)
    updated_states = _extract_states(parameters, states, shared_bias=True)
    reintegrate = tuple(identifier for identifier, state in states.items()
                        if np.linalg.norm(state.bias.gyroscope - updated_states[identifier].bias.gyroscope) > .01)
    return InertialOptimizationResult(updated_states, {}, np.eye(3), 1., initial, final, completed,
                                      reintegrate_ids=reintegrate)


def refine_gravity_scale(states, links, gravity_rotation=None, scale=1., iterations=10):
    """Scale-refinement overload: only gravity and log scale are free, Huber=1."""
    parameters = _state_parameters(states)
    # The source initializes every fixed bias vertex from the first keyframe.
    if states:
        bias = next(iter(states.values())).bias
        for identifier in states:
            parameters["gyro", identifier] = bias.gyroscope.copy()
            parameters["accel", identifier] = bias.accelerometer.copy()
    fixed = set(parameters)
    parameters["gravity", 0] = np.eye(3) if gravity_rotation is None else np.asarray(gravity_rotation, float).copy()
    parameters["scale", 0] = float(scale)
    factors = [_inertial_factor(link, gravity_scale=True, robust=1.) for link in links]
    parameters, initial, final, completed = _optimize(parameters, factors, fixed, iterations, gauss_newton=True)
    return InertialOptimizationResult(_extract_states(parameters, states), {}, parameters["gravity", 0],
                                      parameters["scale", 0], initial, final, completed)


def initial_gravity_velocities(states, links):
    """LocalMapping::InitializeIMU gravity and finite-difference velocity seed.

    The source's exactly parallel/antiparallel gravity case divides by zero.
    The continuous aligned limit is handled explicitly; antiparallel gravity
    uses a pi rotation about x, recording this degenerate-input repair.
    """
    updated = copy.deepcopy(states)
    direction = np.zeros(3)
    for link in links:
        previous, current = states[link.previous], states[link.current]
        direction -= previous.rotation @ link.preintegrated.corrected_deltas()[1]
        velocity = (current.position - previous.position) / link.preintegrated.dt
        updated[link.current].velocity = velocity.copy()
        updated[link.previous].velocity = velocity.copy()
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("gravity direction is unobservable from these IMU increments")
    direction /= norm
    gravity_axis = np.array([0., 0., -1.])
    axis = np.cross(gravity_axis, direction)
    axis_norm = np.linalg.norm(axis)
    angle = np.arccos(np.clip(gravity_axis @ direction, -1., 1.))
    if axis_norm < 1e-12:
        rotation = np.eye(3) if angle < 1e-6 else rotation_exp([np.pi, 0., 0.])
    else:
        rotation = rotation_exp(axis * (angle / axis_norm))
    return updated, rotation


def inertial_initialization_ready(states, timestamps, monocular=True):
    """Minimum keyframes/duration of LocalMapping::InitializeIMU."""
    return len(states) >= 10 and (max(timestamps.values()) - min(timestamps.values()) >= (2. if monocular else 1.))


def full_inertial_bundle_adjustment(states, points, observations, links, calibrations,
                                    iterations=10, initialize_bias=False, prior_gyro=0.,
                                    prior_accel=0., fixed_states=(), fix_local=False,
                                    local_marks=None, fixed_marks=None, inertial_ids=None):
    """Optimizer::FullInertialBA graph and optimization core.

    The source leaves all poses free unless fix_local selects keyframes by
    their BA generation markers. No implicit first-pose constraint is added.
    With initialize_bias it has one shared bias and zero-mean bias priors;
    otherwise each temporal state has biases linked by random walks.
    Points seen only by fixed poses are excluded, as in the original.
    """
    observations, links = tuple(observations), tuple(links)
    inertial_ids = set(states) if inertial_ids is None else set(inertial_ids)
    fixed_ids = set(fixed_states)
    if fix_local and states:
        maximum = max(states)
        fixed_ids |= {identifier for identifier in states if
                      (local_marks or {}).get(identifier, 0) >= maximum - 1 or
                      (fixed_marks or {}).get(identifier, 0) >= maximum - 1}
        if len(set(states) - fixed_ids) < 3:
            return InertialOptimizationResult(copy.deepcopy(states), copy.deepcopy(points), np.eye(3),
                                              1., 0., 0., 0, accepted=False)
    active_points = {observation.point_id for observation in observations if observation.pose_id not in fixed_ids}
    active_observations = tuple(observation for observation in observations if observation.point_id in active_points)
    parameters = _state_parameters(states, points, shared_bias=initialize_bias)
    if initialize_bias and states:
        # FullInertialBA initializes the shared bias from its last keyframe.
        bias = next(reversed(states.values())).bias
        parameters["gyro", "shared"], parameters["accel", "shared"] = bias.gyroscope.copy(), bias.accelerometer.copy()
    fixed = {key for key in parameters if key[1] in fixed_ids and key[0] != "point"}
    fixed |= {key for key in parameters if key[0] in ("velocity", "gyro", "accel")
              and key[1] != "shared" and key[1] not in inertial_ids}
    fixed |= {("point", identifier) for identifier in points if identifier not in active_points}
    factors = []
    for link in links:
        if link.previous not in inertial_ids or link.current not in inertial_ids:
            continue
        factors.append(_inertial_factor(link, shared_bias=initialize_bias, robust=16.92))
        if not initialize_bias:
            factors.extend(_walk_factors(link))
    if initialize_bias:
        factors += [_bias_prior_factor("gyro", prior_gyro), _bias_prior_factor("accel", prior_accel)]
    factors.extend(_visual_factor(observation, _calibration_for(calibrations, observation.pose_id))
                   for observation in active_observations)
    parameters, initial, final, completed = _optimize(parameters, factors, fixed, iterations, 1e-5)
    chi2, depth = _visual_diagnostics(parameters, observations, calibrations)
    outliers = np.array([value > observation.chi2_threshold or not np.isfinite(value)
                         for value, observation in zip(chi2, observations)])
    # Full BA does not erase observations; outliers are returned for inspection.
    updated_states = _extract_states(parameters, states, initialize_bias)
    for identifier in set(states) - inertial_ids:
        updated_states[identifier].bias = copy.deepcopy(states[identifier].bias)
    return InertialOptimizationResult(updated_states,
                                      {identifier: parameters["point", identifier] for identifier in points},
                                      np.eye(3), 1., initial, final, completed, chi2, outliers)


def local_inertial_bundle_adjustment(states, points, observations, links, calibrations,
                                     optimizable_ids, fixed_ids, large=False,
                                     reinitialize=False, track_depths=None):
    """LocalInertialBA graph after explicit temporal/fixed-window selection.

    ``links`` are newest first, matching the source temporal window. Its last
    link is the fixed-boundary edge, downweighted by .01 and Huber-robustified.
    Every inertial link is Huber robust during reinitialization. Caller applies
    returned landmark/observation changes only if accepted.
    """
    observations, links = tuple(observations), tuple(links)
    optimizable_ids, fixed_ids = set(optimizable_ids), set(fixed_ids)
    included = optimizable_ids | fixed_ids
    if included != set(states) or optimizable_ids & fixed_ids:
        raise ValueError("local state IDs must be partitioned into optimizable and fixed sets")
    parameters = _state_parameters(states, points)
    fixed = {key for key in parameters if key[0] != "point" and key[1] in fixed_ids}
    factors = []
    for index, link in enumerate(links):
        boundary = index == len(links) - 1
        factors.append(_inertial_factor(link, robust=16.92 if boundary or reinitialize else None,
                                        downweight=.01 if boundary else 1.))
        factors.extend(_walk_factors(link))
    factors.extend(_visual_factor(observation, _calibration_for(calibrations, observation.pose_id)) for observation in observations)
    parameters, initial, final, completed = _optimize(parameters, factors, fixed, 4 if large else 10,
                                                      1e-2 if large else 1.)
    chi2, depth = _visual_diagnostics(parameters, observations, calibrations)
    outliers = []
    for index, observation in enumerate(observations):
        if observation.measurement.size == 2:
            close = (track_depths or {}).get(observation.point_id, np.inf) < 10.
            outliers.append(chi2[index] > MONO_CHI2 * (1.5 if close else 1.) or depth[index] <= 0)
        else:
            outliers.append(chi2[index] > STEREO_CHI2)
    accepted = large or (np.isfinite(initial) and np.isfinite(final) and final <= 2 * initial)
    return InertialOptimizationResult(_extract_states(parameters, states),
                                      {identifier: parameters["point", identifier] for identifier in points},
                                      np.eye(3), 1., initial, final, completed, chi2, np.asarray(outliers), accepted)


def select_inertial_local_window(current, predecessors, observation_ids, large=False,
                                  bad_ids=(), maximum_fixed=200):
    """Source temporal/fixed keyframe selection, without mutating map markers.

    ``predecessors`` maps all map IDs to their previous temporal ID or None.
    ``observation_ids`` maps each map point to observing keyframe IDs in source
    order. Return temporal IDs newest first, fixed IDs, and local point IDs.
    Covisible visual-only optimization is disabled by source maxCovKF=0.
    """
    maximum = min(len(predecessors) - 2, 25 if large else 10)
    temporal = [current]
    for _ in range(1, maximum):
        previous = predecessors.get(temporal[-1])
        if previous is None:
            break
        temporal.append(previous)
    points = [identifier for identifier, observers in observation_ids.items() if any(k in temporal for k in observers)]
    previous = predecessors.get(temporal[-1])
    fixed = [previous] if previous is not None else [temporal.pop()]
    bad, selected = set(bad_ids), set(temporal) | set(fixed)
    for point in points:
        for observer in observation_ids[point]:
            if observer not in selected:
                selected.add(observer)
                if observer not in bad:
                    fixed.append(observer)
                    break
        if len(fixed) >= maximum_fixed:
            break
    return tuple(temporal), tuple(fixed), tuple(points)


def merge_inertial_bundle_adjustment(states, points, observations, links, calibrations,
                                      temporal_ids, covisible_ids, fixed_ids, inertial_ids=None):
    """Optimizer::MergeInertialBA graph after its two-map window selection.

    Temporal states have free poses/velocities/biases. Additional covisible
    states have only free poses, and the older-map boundary state is fixed.
    Eight LM iterations start at lambda 1e3; all inertial and visual edges are
    Huber robust. Source merge BA has left-camera observations only. Inputs
    must already enforce its keyframe-ID and valid-observation filters.
    """
    temporal_ids, covisible_ids, fixed_ids = set(temporal_ids), set(covisible_ids), set(fixed_ids)
    inertial_ids = set(states) if inertial_ids is None else set(inertial_ids)
    if (temporal_ids & covisible_ids or temporal_ids & fixed_ids or covisible_ids & fixed_ids
            or temporal_ids | covisible_ids | fixed_ids != set(states)):
        raise ValueError("merge states must be partitioned into temporal, covisible and fixed IDs")
    parameters = _state_parameters(states, points)
    fixed = {key for key in parameters if key[0] != "point" and key[1] in fixed_ids}
    fixed |= {key for key in parameters if key[0] in ("velocity", "gyro", "accel") and key[1] in covisible_ids}
    fixed |= {key for key in parameters if key[0] in ("velocity", "gyro", "accel") and key[1] not in inertial_ids}
    factors = []
    for link in links:
        # Pose-only covisibles have no velocity/bias graph vertices in C++.
        if (link.current not in temporal_ids or link.previous in covisible_ids
                or link.previous not in inertial_ids or link.current not in inertial_ids):
            continue
        factors.append(_inertial_factor(link, robust=16.92))
        factors.extend(_walk_factors(link))
    factors.extend(_visual_factor(observation, _calibration_for(calibrations, observation.pose_id))
                   for observation in observations)
    parameters, initial, final, completed = _optimize(parameters, factors, fixed, 8, 1e3)
    chi2, _ = _visual_diagnostics(parameters, observations, calibrations)
    outliers = np.array([value > observation.chi2_threshold for value, observation in zip(chi2, observations)])
    return InertialOptimizationResult(_extract_states(parameters, states),
                                      {identifier: parameters["point", identifier] for identifier in points},
                                      np.eye(3), 1., initial, final, completed, chi2, outliers)


def optimize_inertial_tracking(previous, current, preintegrated, points, observations,
                                calibration, previous_prior=None, previous_id=0,
                                current_id=1, reinitialize=False, track_depths=None):
    """Last-keyframe or last-frame pose-inertial optimization and next prior.

    No previous_prior means a fixed last keyframe. Supplying one optimizes
    both consecutive frames with the original robust 15-state prior and then
    Schur-marginalizes the previous state. Landmarks are fixed in both cases.
    Visual observations must use current_id. The four GN stages retain their
    current state while outlier inclusion changes, unlike visual pose-only BA.
    """
    observations = tuple(observations)
    if any(observation.pose_id != current_id for observation in observations):
        raise ValueError("inertial tracking observations must belong to the current frame")
    states = {previous_id: previous, current_id: current}
    parameters = _state_parameters(states, points)
    fixed = {("point", identifier) for identifier in points}
    if previous_prior is None:
        fixed |= {key for key in parameters if key[0] != "point" and key[1] == previous_id}
    link = InertialLink(previous_id, current_id, preintegrated)
    inertial_factors = [_inertial_factor(link)] + _walk_factors(link)
    if previous_prior is not None:
        inertial_factors.append(_prior_factor(previous_id, previous_prior))
    visual_factors = [_visual_factor(observation, calibration) for observation in observations]
    factors = inertial_factors + visual_factors
    mono_thresholds = [12., 7.5, 5.991, 5.991] if previous_prior is None else [5.991] * 4
    stereo_thresholds = [15.6, 9.8, 7.815, 7.815]
    initial_cost, completed = _graph_cost(parameters, factors), 0
    outliers = np.zeros(len(observations), dtype=bool)
    for stage in range(4):
        parameters, _, _, count = _optimize(parameters, factors, fixed, 10, gauss_newton=True)
        completed += count
        chi2, depth = _visual_diagnostics(parameters, observations, calibration)
        for index, (observation, factor) in enumerate(zip(observations, visual_factors)):
            if observation.measurement.size == 2:
                close = (track_depths or {}).get(observation.point_id, np.inf) < 10.
                outliers[index] = chi2[index] > mono_thresholds[stage] * (1.5 if close else 1.) or depth[index] <= 0
            else:
                outliers[index] = chi2[index] > stereo_thresholds[stage]
            factor.active = not outliers[index]
            if stage == 2:
                factor.huber_squared = None
        if len(factors) < 10:
            break
    if np.count_nonzero(~outliers) < 30 and not reinitialize:
        # The source recovery pass deliberately uses only chi2, ignoring depth.
        outliers = np.array([value >= (18. if observation.measurement.size == 2 else 24.)
                             for value, observation in zip(chi2, observations)])
    for factor, is_outlier in zip(visual_factors, outliers):
        factor.active = not is_outlier
    # Recover an unrobustified Hessian in [previous 15, current 15] order.
    state_order = tuple((kind, identifier) for identifier in (previous_id, current_id)
                        for kind in ("pose", "velocity", "gyro", "accel"))
    offsets, dimension = {}, 0
    for key in state_order:
        offsets[key], dimension = dimension, dimension + _dimension(key)
    hessian, _ = _normal_equations(parameters, factors, offsets, dimension, robust=False)
    hessian = hessian.toarray()
    if previous_prior is not None:
        hessian = marginalize_hessian(hessian, 0, 14)
    next_information = hessian[15:, 15:]
    updated_states = _extract_states(parameters, states)
    return InertialOptimizationResult(updated_states, copy.deepcopy(points), np.eye(3), 1., initial_cost,
                                      _graph_cost(parameters, factors), completed, chi2, outliers,
                                      prior=StatePrior(updated_states[current_id], next_information))
