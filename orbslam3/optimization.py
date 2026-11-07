"""ORB-SLAM3 visual factors, pose optimization, and bundle adjustment.

Port of ``src/Optimizer.cc`` and ``src/OptimizableTypes.cpp`` at the recorded
baseline. Poses are T_cw matrices; increments are left multiplied SE(3), with
rotation first. Residuals are measurement minus prediction. Every observation
is weighted by the inverse image-pyramid level variance, including stereo's
synthetic right coordinate for RGB-D.

The numerical solver below follows the bundled g2o Levenberg implementation.
Landmarks are marginalized by a Schur complement, rather than put into one
large dense normal matrix. State selection and map mutations belong to mapping.

Derived from ORB-SLAM3, Copyright (C) 2017-2021 Carlos Campos, Richard Elvira,
Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of
Zaragoza; Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D.
Tardós, University of Zaragoza. GNU GPL version 3 or later; see LICENSE.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from .geometry import se3_exp as _se3_exp, skew as _skew


MONO_CHI2 = 5.991
STEREO_CHI2 = 7.815


@dataclass(frozen=True)
class ReprojectionObservation:
    """One mono, rectified stereo/RGB-D, or rigid-right-camera factor.

    For unrectified stereo, supply two 2D observations. The right observation
    carries T_right_left in ``camera_from_reference`` and the right camera.
    For rectified stereo, measurement is [u_left, v_left, u_right] and
    ``baseline_fx`` is the calibrated baseline times left fx.
    """

    pose_id: int
    point_id: int
    measurement: np.ndarray
    camera: Any
    inverse_variance: float = 1.0
    baseline_fx: float = 0.0
    camera_from_reference: np.ndarray | None = None

    def __post_init__(self):
        measurement = np.asarray(self.measurement, dtype=float)
        if measurement.shape not in ((2,), (3,)):
            raise ValueError("Measurement must have two or three coordinates")
        if not np.isfinite(measurement).all():
            raise ValueError("Measurement must be finite")
        if not np.isfinite(self.inverse_variance) or self.inverse_variance <= 0:
            raise ValueError("Inverse variance must be finite and positive")
        if measurement.size == 3 and self.camera_from_reference is not None:
            raise ValueError("A rectified stereo factor uses the reference camera")
        object.__setattr__(self, "measurement", measurement.copy())
        if self.camera_from_reference is not None:
            transform = np.asarray(self.camera_from_reference, dtype=float)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("Camera extrinsic must be a finite 4x4 matrix")
            object.__setattr__(self, "camera_from_reference", transform.copy())

    @property
    def chi2_threshold(self):
        return STEREO_CHI2 if self.measurement.size == 3 else MONO_CHI2


@dataclass
class PoseOptimizationResult:
    pose: np.ndarray
    outliers: np.ndarray
    chi2: np.ndarray
    inliers: int
    stage_iterations: tuple[int, ...]


@dataclass
class BundleAdjustmentResult:
    poses: dict[int, np.ndarray]
    points: dict[int, np.ndarray]
    outliers: np.ndarray
    chi2: np.ndarray
    initial_cost: float
    final_cost: float
    iterations: int


def projection_factor(pose, point, observation):
    """Return error, d(error)/d(left pose), d(error)/d(point), and depth.

    This is the same factor for pose-only optimization and visual BA. The
    fixed extrinsic is applied after the optimized left camera pose; its
    rotation consequently multiplies both Jacobians for a right observation.
    """
    point_left = pose[:3, :3] @ point + pose[:3, 3]
    point_camera = point_left
    chain_rotation = np.eye(3)
    if observation.camera_from_reference is not None:
        transform = observation.camera_from_reference
        chain_rotation = transform[:3, :3]
        point_camera = chain_rotation @ point_left + transform[:3, 3]

    prediction = np.asarray(observation.camera.project(point_camera), dtype=float)
    projection_jacobian = observation.camera.project_jacobian(point_camera)
    if observation.measurement.size == 3:
        z = point_camera[2]
        prediction = np.r_[prediction, prediction[0] - observation.baseline_fx / z]
        right_jacobian = projection_jacobian[0].copy()
        right_jacobian[2] += observation.baseline_fx / (z * z)
        projection_jacobian = np.vstack([projection_jacobian, right_jacobian])

    projection_jacobian = -projection_jacobian @ chain_rotation
    pose_jacobian = projection_jacobian @ np.c_[-_skew(point_left), np.eye(3)]
    point_jacobian = projection_jacobian @ pose[:3, :3]
    return observation.measurement - prediction, pose_jacobian, point_jacobian, point_camera[2]


def huber(chi2, delta_squared):
    """g2o's rho(chi²) and rho'(chi²), applied to an entire factor block.

    Applying a library Huber loss separately to x/y coordinates would change
    the objective. The bundled g2o ignores the rho'' Hessian term.
    """
    if chi2 <= delta_squared:
        return float(chi2), 1.0
    norm = np.sqrt(chi2)
    delta = np.sqrt(delta_squared)
    return float(2 * delta * norm - delta_squared), float(delta / norm)


def _edge_cost(error, observation, robust, mono_huber=MONO_CHI2):
    chi2 = observation.inverse_variance * float(error @ error)
    delta_squared = STEREO_CHI2 if error.size == 3 else mono_huber
    return huber(chi2, delta_squared) if robust else (chi2, 1.0)


def _diagnostics(poses, points, observations, check_depth):
    chi2 = np.empty(len(observations))
    outliers = np.zeros(len(observations), dtype=bool)
    for i, observation in enumerate(observations):
        error, _, _, depth = projection_factor(
            poses[observation.pose_id], points[observation.point_id], observation
        )
        chi2[i] = observation.inverse_variance * (error @ error)
        outliers[i] = chi2[i] > observation.chi2_threshold
        if check_depth:
            outliers[i] |= depth <= 0
        # A nonfinite projection is unusable rather than an optimizer success.
        outliers[i] |= not np.isfinite(chi2[i])
    return chi2, outliers


def _levenberg_marquardt(state, assemble, solve, retract, cost, iterations,
                         initial_lambda=None):
    """Bundled g2o LM update/acceptance, with problem-specific normal equations.

    ``assemble`` supplies a normal system and its largest diagonal; ``solve``
    adds lambda to every nonfixed scalar diagonal before Schur elimination.
    Each accepted state remains on its SE(3) manifold.
    """
    damping = None
    multiplier = 2.0
    low_improvement_iterations = 0
    initial_cost = cost(state)
    current_cost = initial_cost
    completed = 0
    for iteration in range(iterations):
        normal, largest_diagonal = assemble(state)
        if damping is None:
            damping = initial_lambda if initial_lambda is not None else 1e-5 * largest_diagonal
        previous_cost = current_cost
        accepted = False
        gain = 0.0
        for trial in range(10):
            try:
                increment, right_hand_side = solve(normal, damping)
                candidate = retract(state, increment)
                candidate_cost = cost(candidate)
                predicted_reduction = increment @ (damping * increment + right_hand_side) + 1e-3
                gain = (current_cost - candidate_cost) / predicted_reduction
            except (np.linalg.LinAlgError, FloatingPointError):
                gain = -np.inf
            if gain > 0 and np.isfinite(candidate_cost):
                damping *= np.clip(1 - (2 * gain - 1)**3, 1 / 3, 2 / 3)
                multiplier = 2
                state = candidate
                current_cost = candidate_cost
                accepted = True
                break
            damping *= multiplier
            multiplier *= 2
            if gain == 0:
                break
        completed = iteration + 1
        if not accepted or trial == 9:
            break
        if (previous_cost - current_cost) * 1e3 < previous_cost:
            low_improvement_iterations += 1
        else:
            low_improvement_iterations = 0
        if low_improvement_iterations >= 3:
            break
    return state, initial_cost, current_cost, completed


def optimize_pose(pose, points_w, observations: Sequence[ReprojectionObservation]):
    """Optimize one camera pose using ORB-SLAM3's four-stage rejection policy.

    ``points_w`` can be a mapping keyed by point_id or an array indexed by it.
    Returned outliers/chi² are aligned with observations, allowing callers to
    map them back to frame feature indices. Like the source, every stage
    restarts from the supplied original pose, while edge inclusion survives.
    Pose-only rejection does not add a positive-depth test.
    """
    pose = np.array(pose, dtype=float, copy=True)
    observations = tuple(observations)
    if len(observations) < 3:
        return PoseOptimizationResult(pose, np.zeros(len(observations), bool),
                                      np.zeros(len(observations)), 0, ())
    active = np.ones(len(observations), dtype=bool)
    outliers = ~active
    stage_iterations = []
    optimized_pose = pose.copy()
    for stage in range(4):
        robust = stage < 3

        def cost(current_pose):
            total = 0.0
            for use, observation in zip(active, observations):
                if use:
                    error, _, _, _ = projection_factor(
                        current_pose, points_w[observation.point_id], observation
                    )
                    total += _edge_cost(error, observation, robust)[0]
            return total

        def assemble(current_pose):
            hessian = np.zeros((6, 6))
            rhs = np.zeros(6)
            for use, observation in zip(active, observations):
                if use:
                    error, jacobian, _, _ = projection_factor(
                        current_pose, points_w[observation.point_id], observation
                    )
                    _, weight = _edge_cost(error, observation, robust)
                    information = observation.inverse_variance * weight
                    hessian += information * (jacobian.T @ jacobian)
                    rhs -= information * (jacobian.T @ error)
            return (hessian, rhs), np.max(np.abs(hessian.diagonal()))

        def solve(normal, damping):
            hessian, rhs = normal
            return np.linalg.solve(hessian + damping * np.eye(6), rhs), rhs

        optimized_pose, _, _, count = _levenberg_marquardt(
            pose.copy(), assemble, solve, lambda value, update: _se3_exp(update) @ value,
            cost, 10
        )
        stage_iterations.append(count)
        chi2 = np.empty(len(observations))
        for i, observation in enumerate(observations):
            error, _, _, _ = projection_factor(
                optimized_pose, points_w[observation.point_id], observation
            )
            chi2[i] = observation.inverse_variance * (error @ error)
            outliers[i] = not np.isfinite(chi2[i]) or chi2[i] > observation.chi2_threshold
        active = ~outliers
        if len(observations) < 10:
            break
    return PoseOptimizationResult(optimized_pose, outliers, chi2, int(active.sum()),
                                  tuple(stage_iterations))


def bundle_adjustment(poses: Mapping[int, np.ndarray], points: Mapping[int, np.ndarray],
                      observations: Sequence[ReprojectionObservation], fixed_pose_ids,
                      iterations=5, robust=True, *, mono_huber=5.99,
                      initial_lambda=None):
    """Joint visual BA with fixed camera selection supplied explicitly.

    Global BA fixes the map's initial keyframe. Local mapping selects local
    keyframes, fixes that initial keyframe when present and every external
    keyframe observing local points, then passes exactly those vertices here.
    All referenced landmarks are optimized; unobserved landmarks stay intact.
    Global BA uses sqrt(5.99) for mono Huber, primary local BA sqrt(5.991).

    This function returns optimized arrays and rejected-edge diagnostics; it
    does not erase map observations (global BA in the source does not erase).
    """
    observations = tuple(observations)
    fixed_pose_ids = set(fixed_pose_ids)
    unknown_fixed = fixed_pose_ids - poses.keys()
    if unknown_fixed:
        raise ValueError(f"Unknown fixed pose IDs: {unknown_fixed}")
    if iterations < 0:
        raise ValueError("Iteration count cannot be negative")
    for observation in observations:
        if observation.pose_id not in poses or observation.point_id not in points:
            raise ValueError("Every factor must reference an existing pose and point")
    pose_ids = sorted({o.pose_id for o in observations} - fixed_pose_ids)
    point_ids = sorted({o.point_id for o in observations})
    pose_index = {key: i for i, key in enumerate(pose_ids)}
    point_index = {key: i for i, key in enumerate(point_ids)}
    pose_size = 6 * len(pose_ids)
    point_size = 3 * len(point_ids)
    original_poses = {key: np.array(value, dtype=float, copy=True) for key, value in poses.items()}
    original_points = {key: np.array(value, dtype=float, copy=True) for key, value in points.items()}

    def cost(state):
        current_poses, current_points = state
        total = 0.0
        for observation in observations:
            error, _, _, _ = projection_factor(
                current_poses[observation.pose_id], current_points[observation.point_id], observation
            )
            total += _edge_cost(error, observation, robust, mono_huber)[0]
        return total

    def assemble(state):
        current_poses, current_points = state
        pose_hessian = np.zeros((pose_size, pose_size))
        pose_rhs = np.zeros(pose_size)
        point_hessian = np.zeros((len(point_ids), 3, 3))
        point_rhs = np.zeros((len(point_ids), 3))
        # A landmark only has cross blocks with cameras that observe it.
        cross_blocks = [dict() for _ in point_ids]
        for observation in observations:
            error, pose_jac, point_jac, _ = projection_factor(
                current_poses[observation.pose_id], current_points[observation.point_id], observation
            )
            _, weight = _edge_cost(error, observation, robust, mono_huber)
            information = weight * observation.inverse_variance
            j = point_index[observation.point_id]
            point_hessian[j] += information * point_jac.T @ point_jac
            point_rhs[j] -= information * point_jac.T @ error
            if observation.pose_id in pose_index:
                i = pose_index[observation.pose_id]
                block = slice(6 * i, 6 * i + 6)
                pose_hessian[block, block] += information * pose_jac.T @ pose_jac
                pose_rhs[block] -= information * pose_jac.T @ error
                cross = information * pose_jac.T @ point_jac
                cross_blocks[j][i] = cross_blocks[j].get(i, np.zeros((6, 3))) + cross
        largest = max(np.max(np.abs(pose_hessian.diagonal()), initial=0),
                      np.max(np.abs(np.diagonal(point_hessian, axis1=1, axis2=2)), initial=0))
        return (pose_hessian, pose_rhs, point_hessian, point_rhs, cross_blocks), largest

    def solve(normal, damping):
        hpp, bp, hll, bl, cross_blocks = normal
        # Apply damping BEFORE marginalization, as g2o's block solver does.
        landmark_inverse = np.linalg.inv(hll + damping * np.eye(3))
        schur = hpp + damping * np.eye(pose_size)
        reduced_rhs = bp.copy()
        for j, blocks in enumerate(cross_blocks):
            for i, cross_i in blocks.items():
                rows = slice(6 * i, 6 * i + 6)
                product = cross_i @ landmark_inverse[j]
                reduced_rhs[rows] -= product @ bl[j]
                for k, cross_k in blocks.items():
                    columns = slice(6 * k, 6 * k + 6)
                    schur[rows, columns] -= product @ cross_k.T
        delta_pose = np.linalg.solve(schur, reduced_rhs) if pose_size else np.empty(0)
        delta_point = np.empty((len(point_ids), 3))
        for j, blocks in enumerate(cross_blocks):
            right_hand_side = bl[j].copy()
            for i, cross in blocks.items():
                right_hand_side -= cross.T @ delta_pose[6 * i:6 * i + 6]
            delta_point[j] = landmark_inverse[j] @ right_hand_side
        return np.r_[delta_pose, delta_point.ravel()], np.r_[bp, bl.ravel()]

    def retract(state, increment):
        current_poses, current_points = state
        updated_poses = current_poses.copy()
        updated_points = current_points.copy()
        for key, i in pose_index.items():
            updated_poses[key] = _se3_exp(increment[6 * i:6 * i + 6]) @ current_poses[key]
        for key, j in point_index.items():
            start = pose_size + 3 * j
            updated_points[key] = current_points[key] + increment[start:start + 3]
        return updated_poses, updated_points

    state = (original_poses, original_points)
    if not observations or not pose_size + point_size:
        return BundleAdjustmentResult(*state, np.zeros(len(observations), bool),
                                      np.zeros(len(observations)), 0.0, 0.0, 0)
    state, before, after, count = _levenberg_marquardt(
        state, assemble, solve, retract, cost, iterations, initial_lambda
    )
    chi2, outliers = _diagnostics(*state, observations, check_depth=True)
    return BundleAdjustmentResult(*state, outliers, chi2, before, after, count)


def local_bundle_adjustment(poses, points, observations, fixed_pose_ids, *, inertial_map=False):
    """Primary visual LBA: 10 robust iterations, then chi²/depth diagnostics.

    ``inertial_map`` changes the source LM starting lambda to 100. It does NOT
    turn this visual objective into visual-inertial BA. The caller must erase
    rejected observations and update normals/depth bounds after committing.
    """
    fixed_pose_ids = set(fixed_pose_ids)
    if not fixed_pose_ids:
        raise ValueError("ORB-SLAM3 skips local BA when there are no fixed keyframes")
    return bundle_adjustment(poses, points, observations, fixed_pose_ids, iterations=10,
                             robust=True, mono_huber=MONO_CHI2,
                             initial_lambda=100.0 if inertial_map else None)


def merge_bundle_adjustment(poses, points, observations, fixed_pose_ids):
    """Visual welding BA overload: 5 robust iterations, reject, 10 least squares.

    This is separate from the primary local BA's single robust solve. Final
    diagnostics cover every original edge, including edges omitted in stage 2.
    """
    observations = tuple(observations)
    first = bundle_adjustment(poses, points, observations, fixed_pose_ids, iterations=5)
    active = [observation for observation, outlier in zip(observations, first.outliers) if not outlier]
    second = bundle_adjustment(first.poses, first.points, active, fixed_pose_ids,
                              iterations=10, robust=False)
    chi2, outliers = _diagnostics(second.poses, second.points, observations, check_depth=True)
    return BundleAdjustmentResult(second.poses, second.points, outliers, chi2,
                                  first.initial_cost, second.final_cost,
                                  first.iterations + second.iterations)
