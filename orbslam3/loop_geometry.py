"""Source Sim(3) refinement and essential graph optimization.

Source: Optimizer::OptimizeSim3, OptimizeEssentialGraph, and its 4DoF variant;
OptimizableTypes::{EdgeSim3ProjectXYZ,EdgeInverseSim3ProjectXYZ}; g2o::EdgeSim3;
G2oTypes::{VertexPose4DoF,Edge4DoF}. Place recognition and map mutation are
separate responsibilities; all graph variable/factor selection is visible.

Derived from ORB-SLAM3, Copyright (C) 2017-2021 Carlos Campos, Richard Elvira,
Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of
Zaragoza; Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D.
Tardós, University of Zaragoza. GPL-3.0-or-later; see LICENSE.
"""

from dataclasses import dataclass, field

import numpy as np

from .geometry import Sim3, inverse_pose, make_pose, skew, so3_exp, so3_log
from .optimization import _levenberg_marquardt, huber


@dataclass
class Sim3ReprojectionMatch:
    point1: np.ndarray
    point2: np.ndarray
    pixel1: np.ndarray
    pixel2: np.ndarray
    camera1: object
    camera2: object
    inverse_variance1: float = 1.0
    inverse_variance2: float = 1.0
    observed2: bool = True


@dataclass
class Sim3OptimizationResult:
    transform: Sim3
    inliers: np.ndarray
    chi2: np.ndarray
    count: int
    iterations: int
    hessian: np.ndarray


def sim3_projection_factors(transform, match):
    """Paired reprojection residuals and analytic left-Sim3 Jacobians.

    Forward projects S12 X2; inverse projects S21 X1. Fixed XYZ points remain
    in camera coordinates, exactly as the source's two fixed point vertices.
    """
    point1 = transform.map(match.point2)
    point2 = transform.inverse().map(match.point1)
    forward_error = np.asarray(match.pixel1) - match.camera1.project(point1)
    inverse_error = np.asarray(match.pixel2) - match.camera2.project(point2)
    forward_jacobian = -match.camera1.project_jacobian(point1) @ np.c_[
        -skew(point1), np.eye(3), point1
    ]
    inverse_action = transform.rotation.T / transform.scale
    inverse_jacobian = -match.camera2.project_jacobian(point2) @ inverse_action @ np.c_[
        skew(match.point1), -np.eye(3), -np.asarray(match.point1)
    ]
    return forward_error, inverse_error, forward_jacobian, inverse_jacobian


def optimize_sim3(initial, matches, threshold=10.0, *, fix_scale=False, all_points=False):
    """Five robust iterations, bilateral rejection, then five/ten plain ones.

    Source permits a point absent from KF2 only when bAllPoints is true. Its
    caller supplies the source's synthetic normalized observation and level-0
    variance in that case; this routine does not replace it with a pixel.
    Invalid/negative-Z2 pairs are omitted before building the graph. The
    returned mask is aligned with the caller's match list.
    """
    matches = tuple(matches)
    if threshold <= 0:
        raise ValueError("The chi-square threshold must be positive")
    active = np.array([m.point2[2] >= 0 and (m.observed2 or all_points) for m in matches])
    # bool dtype matters for an empty list: NumPy otherwise makes float64.
    active = active.astype(bool)
    dimension = 6 if fix_scale else 7

    def run(start, mask, robust, iterations):
        def cost(transform):
            total = 0.0
            for use, match in zip(mask, matches):
                if not use:
                    continue
                e1, e2, _, _ = sim3_projection_factors(transform, match)
                for error, inverse_variance in ((e1, match.inverse_variance1),
                                                  (e2, match.inverse_variance2)):
                    chi2 = inverse_variance * (error @ error)
                    total += huber(chi2, threshold)[0] if robust else chi2
            return total

        def assemble(transform):
            hessian = np.zeros((dimension, dimension))
            rhs = np.zeros(dimension)
            for use, match in zip(mask, matches):
                if not use:
                    continue
                e1, e2, j1, j2 = sim3_projection_factors(transform, match)
                for error, jacobian, invvar in ((e1, j1, match.inverse_variance1),
                                                 (e2, j2, match.inverse_variance2)):
                    jacobian = jacobian[:, :dimension]
                    weight = huber(invvar * (error @ error), threshold)[1] if robust else 1.0
                    information = weight * invvar
                    hessian += information * jacobian.T @ jacobian
                    rhs -= information * jacobian.T @ error
            return (hessian, rhs), np.max(np.abs(hessian.diagonal()), initial=0)

        def solve(normal, damping):
            hessian, rhs = normal
            return np.linalg.solve(hessian + damping * np.eye(dimension), rhs), rhs

        def retract(transform, increment):
            tangent = np.r_[increment, 0.0] if fix_scale else increment
            return Sim3.exp(tangent) @ transform

        return _levenberg_marquardt(start, assemble, solve, retract, cost, iterations)

    def classify(transform):
        chi2 = np.full((len(matches), 2), np.inf)
        for i, (use, match) in enumerate(zip(active, matches)):
            if use:
                e1, e2, _, _ = sim3_projection_factors(transform, match)
                chi2[i] = [match.inverse_variance1 * (e1 @ e1),
                           match.inverse_variance2 * (e2 @ e2)]
        return chi2, np.all(np.isfinite(chi2) & (chi2 <= threshold), axis=1)

    if not active.any():
        return Sim3OptimizationResult(initial, active, np.full((len(matches), 2), np.inf),
                                       0, 0, np.zeros((7, 7)))
    first, _, _, iterations1 = run(initial, active, True, 5)
    chi2, inliers = classify(first)
    removed = int(active.sum() - inliers.sum())
    if inliers.sum() < 10:
        # Source returns before assigning g2oS12, leaving the supplied estimate
        # unchanged, but still erases first-stage outlier correspondences.
        return Sim3OptimizationResult(initial, inliers, chi2, 0, iterations1, np.zeros((7, 7)))
    refined, _, _, iterations2 = run(first, inliers, False, 10 if removed else 5)
    chi2, final_inliers = classify(refined)
    final_inliers &= inliers  # Removed graph edges never re-enter this optimizer.
    # mAcumHessian is assigned zero and never accumulated in this exact baseline.
    return Sim3OptimizationResult(refined, final_inliers, chi2, int(final_inliers.sum()),
                                   iterations1 + iterations2, np.zeros((7, 7)))


@dataclass
class EssentialGraphEdge:
    first: int
    second: int
    measurement: Sim3
    # For Sim3 measurement is S_second_first; for 4DoF it is S_first_second.
    # These follow the two distinct residual definitions in the source.
    information: np.ndarray | None = None
    kind: str = "covisibility"


@dataclass
class EssentialGraphResult:
    similarities: dict[int, Sim3]
    poses: dict[int, np.ndarray]
    initial_cost: float
    final_cost: float
    iterations: int


def essential_graph_edges(initial_poses, *, noncorrected=None, parents=None,
                          historical_loops=None, covisibility=None, previous=None,
                          loop_connections=None, current_id=None, loop_id=None,
                          four_dof=False):
    """Construct exactly the standard source's loop/tree/covisibility graph.

    initial_poses already includes the corrected neighborhood. Normal edge
    measurements use noncorrected poses when supplied; new loop-connection
    measurements use corrected initial_poses. For 4DoF, spanning-tree edges are
    intentionally disabled in this local baseline and temporal edges replace
    them. Covisibility is a nested dict with shared-map-point counts.
    """
    noncorrected = {} if noncorrected is None else noncorrected
    parents = {} if parents is None else parents
    historical_loops = {} if historical_loops is None else historical_loops
    covisibility = {} if covisibility is None else covisibility
    previous = {} if previous is None else previous
    loop_connections = {} if loop_connections is None else loop_connections
    children = {key: set() for key in initial_poses}
    next_frames = {}
    for child, parent in parents.items():
        if parent is not None:
            children.setdefault(parent, set()).add(child)
    for current, prior in previous.items():
        if prior is not None:
            next_frames[prior] = current
    information = np.diag([1000., 1000., 1., 1., 1., 1.]) if four_dof else np.eye(7)
    edges = []
    inserted = set()

    def edge(first, second, poses, kind):
        relative = poses[first] @ poses[second].inverse() if four_dof else poses[second] @ poses[first].inverse()
        return EssentialGraphEdge(first, second, relative, information.copy(), kind)

    for first, neighbors in loop_connections.items():
        if first not in initial_poses:
            continue
        for second in sorted(neighbors):
            if second not in initial_poses:
                continue
            special_loop = first == current_id and second == loop_id
            if not special_loop and covisibility.get(first, {}).get(second, 0) < 100:
                continue
            edges.append(edge(first, second, initial_poses, "new_loop"))
            inserted.add(tuple(sorted((first, second))))

    normal_poses = {key: noncorrected.get(key, value) for key, value in initial_poses.items()}
    for first in sorted(initial_poses):
        parent = None if four_dof else parents.get(first)
        if parent is not None and parent in initial_poses:
            edges.append(edge(first, parent, normal_poses, "spanning_tree"))
        if four_dof and previous.get(first) in initial_poses:
            edges.append(edge(first, previous[first], normal_poses, "inertial_temporal"))
        loops = historical_loops.get(first, set())
        for second in sorted(loops):
            if second < first and second in initial_poses:
                edges.append(edge(first, second, normal_poses, "historical_loop"))
        for second, weight in sorted(covisibility.get(first, {}).items()):
            if weight < 100 or second >= first or second not in initial_poses:
                continue
            if second == parent or second in children.get(first, set()):
                continue
            if tuple(sorted((first, second))) in inserted:
                continue
            if four_dof and second in (previous.get(first), next_frames.get(first)):
                continue
            if four_dof and second in loops:
                continue
            edges.append(edge(first, second, normal_poses, "covisibility"))
        if not four_dof and previous.get(first) in initial_poses:
            edges.append(edge(first, previous[first], normal_poses, "inertial_temporal"))
    return edges


def sim3_graph_factor(first, second, measurement):
    """g2o::EdgeSim3: log(measurement * S_first_world * S_world_second)."""
    return (measurement @ first @ second.inverse()).log()


def four_dof_graph_factor(first, second, measurement):
    """Source Edge4DoF's 6D error despite its 4D state updates."""
    rotation = first.rotation @ second.rotation.T
    relative_translation = first.rotation @ (-second.rotation.T @ second.translation) + first.translation
    return np.r_[so3_log(rotation @ measurement.rotation.T),
                 relative_translation - measurement.translation]


def _four_dof_retract(value, increment, camera_from_body):
    world_from_camera = inverse_pose(value.camera_pose())
    world_from_body = world_from_camera @ camera_from_body
    world_from_body[:3, :3] = so3_exp([0., 0., increment[0]]) @ world_from_body[:3, :3]
    world_from_body[:3, 3] += increment[1:]
    camera_from_world = camera_from_body @ inverse_pose(world_from_body)
    return Sim3(camera_from_world[:3, :3], camera_from_world[:3, 3], 1.)


def optimize_essential_graph(initial_poses, edges, fixed_pose_ids, *, fix_scale=False,
                              four_dof=False, camera_from_body=None, iterations=20):
    """Solve an explicitly constructed Sim3 or inertial yaw/translation graph.

    The standard Sim3 graph fixes the map-origin keyframe; the inertial 4DoF
    graph fixes the loop keyframe. The caller supplies these choices explicitly.
    camera_from_body is a mapping of T_cb extrinsics for 4DoF updates; identity
    is appropriate only when camera and body frames coincide. Roll and pitch
    of R_wb remain unchanged while yaw and world position are updated.
    """
    fixed_pose_ids = set(fixed_pose_ids)
    if not fixed_pose_ids or not fixed_pose_ids <= initial_poses.keys():
        raise ValueError("Fix at least one existing pose to anchor the graph")
    camera_from_body = {} if camera_from_body is None else camera_from_body
    edges = tuple(edges)
    ids = sorted(set(initial_poses) - fixed_pose_ids)
    index = {key: i for i, key in enumerate(ids)}
    dimension = 4 if four_dof else (6 if fix_scale else 7)
    total_dimension = dimension * len(ids)
    factor = four_dof_graph_factor if four_dof else sim3_graph_factor

    def update(value, increment, key):
        if four_dof:
            return _four_dof_retract(value, increment, camera_from_body.get(key, np.eye(4)))
        return Sim3.exp(np.r_[increment, 0.] if fix_scale else increment) @ value

    def information(edge):
        if edge.information is not None:
            return edge.information
        return np.diag([1000., 1000., 1., 1., 1., 1.]) if four_dof else np.eye(7)

    def cost(state):
        total = 0.0
        for edge in edges:
            error = factor(state[edge.first], state[edge.second], edge.measurement)
            total += error @ information(edge) @ error
        return total

    def assemble(state):
        hessian = np.zeros((total_dimension, total_dimension))
        rhs = np.zeros(total_dimension)
        # Both original graph edge classes use g2o's numerical derivatives.
        epsilon = 1e-9
        for edge in edges:
            first, second = state[edge.first], state[edge.second]
            error = factor(first, second, edge.measurement)
            omega = information(edge)
            jacobians = {}
            for endpoint in (edge.first, edge.second):
                if endpoint not in index:
                    continue
                columns = []
                for axis in range(dimension):
                    delta = np.eye(dimension)[axis] * epsilon
                    positive = update(state[endpoint], delta, endpoint)
                    negative = update(state[endpoint], -delta, endpoint)
                    if endpoint == edge.first:
                        difference = factor(positive, second, edge.measurement) - factor(negative, second, edge.measurement)
                    else:
                        difference = factor(first, positive, edge.measurement) - factor(first, negative, edge.measurement)
                    columns.append(difference / (2 * epsilon))
                jacobians[endpoint] = np.column_stack(columns)
            for first_id, jac_first in jacobians.items():
                first_slice = slice(dimension * index[first_id], dimension * (index[first_id] + 1))
                rhs[first_slice] -= jac_first.T @ omega @ error
                for second_id, jac_second in jacobians.items():
                    second_slice = slice(dimension * index[second_id], dimension * (index[second_id] + 1))
                    hessian[first_slice, second_slice] += jac_first.T @ omega @ jac_second
        return (hessian, rhs), np.max(np.abs(hessian.diagonal()), initial=0)

    def solve(normal, damping):
        hessian, rhs = normal
        return np.linalg.solve(hessian + damping * np.eye(total_dimension), rhs), rhs

    def retract(state, increment):
        updated = state.copy()
        for key, i in index.items():
            updated[key] = update(state[key], increment[dimension*i:dimension*(i+1)], key)
        return updated

    state = initial_poses.copy()
    if total_dimension and edges:
        state, before, after, count = _levenberg_marquardt(
            state, assemble, solve, retract, cost, iterations,
            initial_lambda=None if four_dof else 1e-16
        )
    else:
        before = after = cost(state)
        count = 0
    return EssentialGraphResult(state, {key: pose.camera_pose() for key, pose in state.items()},
                                 before, after, count)


def correct_map_points(points, reference_ids, initial_poses, optimized_poses):
    """Carry landmarks through their old and optimized reference-camera Sim3.

    For full Sim3 loops, reference_ids must already incorporate the source's
    mnCorrectedByKF/mnCorrectedReference override; the normal reference applies
    otherwise. The 4DoF source always uses the normal map-point reference.
    """
    return {key: optimized_poses[reference_ids[key]].inverse().map(
        initial_poses[reference_ids[key]].map(point)
    ) for key, point in points.items()}
