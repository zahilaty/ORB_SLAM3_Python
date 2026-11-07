"""Maximum-likelihood PnP used by the local ORB-SLAM3 relocalizer.

Source: src/MLPnPsolver.cpp, including its six-point bearing-nullspace solve,
separate planar reconstruction, and five-step additive Rodrigues refinement.
There is no OpenCV PnP solver in this module.

Derived from ORB-SLAM3, Copyright (C) 2017-2021 Carlos Campos, Richard Elvira,
Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of
Zaragoza; Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D.
Tardós, University of Zaragoza. GPL-3.0-or-later; see LICENSE.

MLPnP portions: Copyright (c) 2016 Steffen Urban, ANU. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of ANU nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL ANU OR THE CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import make_pose, right_jacobian_so3, skew, so3_exp, so3_log


def bearing_nullspaces(bearings):
    """Orthonormal 3x2 bases perpendicular to each measured bearing."""
    result = []
    for bearing in bearings:
        _, _, vt = np.linalg.svd(np.asarray(bearing)[None, :], full_matrices=True)
        result.append(vt[1:].T)
    return np.asarray(result)


def mlpnp_residuals_and_jacobians(state, points, nullspaces):
    """Bearing-nullspace residuals and their additive Rodrigues/XYZ Jacobian.

    This compact chain rule is the same derivative as the source's expanded
    symbolic mlpnpJacs. d(normalize(q))/dq=(I-uu.T)/||q||, and additive Rodrigues
    derivatives use the SO(3) right Jacobian. The zero-angle limit is finite.
    """
    omega, translation = state[:3], state[3:]
    rotation = so3_exp(omega)
    right_jacobian = right_jacobian_so3(omega)
    residuals, jacobians = [], []
    for point, basis in zip(points, nullspaces):
        transformed = rotation @ point + translation
        distance = np.linalg.norm(transformed)
        if distance == 0:
            raise ValueError("A PnP landmark coincides with the camera center")
        direction = transformed / distance
        normalize_jacobian = (np.eye(3) - np.outer(direction, direction)) / distance
        camera_jacobian = np.c_[-rotation @ skew(point) @ right_jacobian, np.eye(3)]
        residuals.append(basis.T @ direction)
        jacobians.append(basis.T @ normalize_jacobian @ camera_jacobian)
    return np.asarray(residuals).ravel(), np.vstack(jacobians)


def _gauss_newton(state, points, nullspaces, information):
    state = state.copy()
    for _ in range(5):
        residual, jacobian = mlpnp_residuals_and_jacobians(state, points, nullspaces)
        weighted_jacobian = jacobian.T if information is None else jacobian.T @ information
        normal = weighted_jacobian @ jacobian
        rhs = weighted_jacobian @ residual
        delta = np.linalg.solve(normal, rhs)
        # Preserve BOTH max and min coefficient guards in mlpnp_gn.
        if np.max(np.abs(delta)) > 5 or np.min(np.abs(delta)) > 1:
            break
        observation_update = jacobian @ delta
        state -= delta
        if np.max(np.abs(observation_update)) < 1e-5:
            break
    return state


def compute_pose(bearings, points, covariances=None):
    """MLPnP's linear initialization, planar handling, and GN bearing fit.

    Bearings are z-normalized by the caller as in the ORB-SLAM3 constructor.
    Rank testing deliberately uses uncentered point scatter, as in this
    baseline. Arbitrary offset planes therefore do not necessarily enter its
    special planar branch. Covariance projection follows the original optional
    3D-bearing covariance path; ORB-SLAM3 itself passes no such covariances.
    """
    bearings, points = np.asarray(bearings, dtype=float), np.asarray(points, dtype=float)
    if bearings.shape != points.shape or points.ndim != 2 or points.shape[1] != 3 or len(points) < 6:
        raise ValueError("MLPnP needs at least six matched Nx3 points and bearings")
    count = len(points)
    nullspaces = bearing_nullspaces(bearings)
    scatter = points.T @ points
    planar = np.linalg.matrix_rank(scatter) == 2
    rotated_points = points
    eigen_rotation = np.eye(3)
    if planar:
        _, eigenvectors = np.linalg.eigh(scatter)
        eigen_rotation = eigenvectors.T
        rotated_points = points @ eigen_rotation.T
    information = None
    if covariances is not None and len(covariances) == count:
        information = np.zeros((2 * count, 2 * count))
        for i, (basis, covariance) in enumerate(zip(nullspaces, covariances)):
            information[2*i:2*i+2, 2*i:2*i+2] = np.linalg.inv(basis.T @ covariance @ basis)

    rotation_columns = 2 if planar else 3
    design = np.zeros((2 * count, 3 * rotation_columns + 3))
    for i, (point, basis) in enumerate(zip(rotated_points, nullspaces)):
        coordinates = point[1:] if planar else point
        design[2*i:2*i+2, :3*rotation_columns] = np.kron(basis.T, coordinates[None, :])
        design[2*i:2*i+2, 3*rotation_columns:] = basis.T
    normal = design.T @ design if information is None else design.T @ information @ design
    _, _, vt = np.linalg.svd(normal)
    linear = vt[-1]

    def rotation_projection(matrix):
        u, _, vt_rotation = np.linalg.svd(matrix)
        result = u @ vt_rotation
        return -result if np.linalg.det(result) < 0 else result

    def bearing_score(rotation, translation):
        predicted = points[:6] @ rotation.T + translation
        predicted /= np.linalg.norm(predicted, axis=1)[:, None]
        return np.sum(1 - np.sum(predicted * bearings[:6], axis=1))

    if planar:
        temporary = np.zeros((3, 3))
        temporary[:, 1:] = linear[:6].reshape(3, 2)
        temporary[:, 0] = np.cross(temporary[:, 1], temporary[:, 2])
        temporary = temporary.T
        scale = 1 / np.sqrt(abs(np.linalg.norm(temporary[:, 1]) * np.linalg.norm(temporary[:, 2])))
        rotation1 = eigen_rotation.T @ rotation_projection(temporary)
        translation = scale * linear[6:]
        rotation1 = -rotation1.T
        if np.linalg.det(rotation1) < 0:
            rotation1[:, 2] *= -1
        rotation2 = rotation1.copy()
        rotation2[:, :2] *= -1
        candidates = [(rotation1, translation), (rotation1, -translation),
                      (rotation2, translation), (rotation2, -translation)]
        rotation, translation = min(candidates, key=lambda pair: bearing_score(*pair))
    else:
        temporary = linear[:9].reshape(3, 3).T
        scale = 1 / np.cbrt(abs(np.prod(np.linalg.norm(temporary, axis=0))))
        inverse_rotation = rotation_projection(temporary)
        inverse_translation = inverse_rotation @ (scale * linear[9:])
        rotation = inverse_rotation.T
        candidate1 = -rotation @ inverse_translation
        candidate2 = rotation @ inverse_translation
        translation = candidate1 if bearing_score(rotation, candidate1) < bearing_score(rotation, candidate2) else candidate2
    state = _gauss_newton(np.r_[so3_log(rotation), translation], points, nullspaces, information)
    return make_pose(so3_exp(state[:3]), state[3:])


@dataclass
class PnPRansacResult:
    pose: np.ndarray | None
    inliers: np.ndarray
    count: int
    exhausted: bool
    converged: bool
    iterations: int


class MLPnPSolver:
    """The source's RANSAC/relocalization adapter for map.Frame records.

    Two observable baseline quirks are retained: the iteration loop uses OR
    rather than AND, and Refine's newly computed local result is discarded
    before inlier classification, leaving the sampled pose as the returned
    pose. These are explicit here so they cannot silently become algorithm
    changes while studying this exact revision.
    """

    def __init__(self, frame, matches, seed=0):
        self.camera = frame.camera
        self.output_size = len(matches)
        self.indices, points, pixels, variances = [], [], [], []
        # A single-camera PnP factor uses left observations. Unrectified right
        # observations have distinct extrinsics and do not belong in this fit.
        limit = frame.n_left if frame.n_left >= 0 else len(frame.features)
        for i, point in enumerate(matches):
            if point is not None and not point.bad and i < limit:
                self.indices.append(i)
                points.append(point.position)
                pixels.append(frame.features.points[i])
                variances.append(1 / frame.inverse_variances[frame.features.octaves[i]])
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        self.variances = np.asarray(variances)
        self.indices = np.asarray(self.indices, dtype=int)
        self.bearings = np.asarray([self.camera.unproject(pixel) for pixel in self.pixels]).reshape(-1, 3)
        if len(self.bearings):
            self.bearings /= self.bearings[:, 2:3]
        self.rng = np.random.default_rng(seed)
        self.iterations = 0
        self.best_count = 0
        self.best_pose = None
        self.best_inliers = np.zeros(len(self.points), dtype=bool)
        self.set_ransac_parameters(.99, 8, 300, 6, .4, 5.991)

    def set_ransac_parameters(self, probability=.99, min_inliers=10, max_iterations=300,
                              min_set=6, epsilon=.5, chi2=5.991):
        if not 0 < probability < 1 or not 0 < epsilon <= 1 or min_set < 6 or max_iterations < 1:
            raise ValueError("Invalid MLPnP RANSAC parameters")
        count = len(self.points)
        self.minimum_inliers = max(int(count * epsilon), min_inliers, min_set)
        self.minimum_set = min_set
        epsilon = max(epsilon, self.minimum_inliers / count) if count else 1
        if count <= self.minimum_inliers:
            budget = 1
        else:
            # The exponent 3 is inherited from source despite its six-point set.
            budget = int(np.ceil(np.log1p(-probability) / np.log1p(-epsilon**3)))
        self.maximum_iterations = max(1, min(budget, max_iterations))
        self.thresholds = self.variances * chi2

    def check_inliers(self, pose):
        camera_points = self.points @ pose[:3, :3].T + pose[:3, 3]
        projections = np.asarray([self.camera.project(point) for point in camera_points])
        errors = np.sum((self.pixels - projections)**2, axis=1)
        # The original does not add a positive-depth test here.
        return errors < self.thresholds

    def iterate(self, iterations=5):
        output = np.zeros(self.output_size, dtype=bool)
        if len(self.points) < self.minimum_inliers:
            return PnPRansacResult(None, output, 0, True, False, self.iterations)
        current_iterations = 0
        while self.iterations < self.maximum_iterations or current_iterations < iterations:
            self.iterations += 1
            current_iterations += 1
            available = list(range(len(self.points)))
            sample = []
            for _ in range(self.minimum_set):
                index = int(self.rng.integers(len(available)))
                sample.append(available[index])
                available[index] = available[-1]
                available.pop()
            try:
                pose = compute_pose(self.bearings[sample], self.points[sample])
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue
            inliers = self.check_inliers(pose)
            count = int(inliers.sum())
            if count < self.minimum_inliers:
                continue
            if count > self.best_count:
                self.best_count, self.best_pose, self.best_inliers = count, pose, inliers.copy()
            # Refine computes on BEST support but discards its local output.
            try:
                compute_pose(self.bearings[self.best_inliers], self.points[self.best_inliers])
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                # The sampled pose still exists; an unusable refinement cannot
                # improve it. Keep searching instead of committing NaN state.
                continue
            refined_inliers = self.check_inliers(pose)
            refined_count = int(refined_inliers.sum())
            if refined_count > self.minimum_inliers:
                output[self.indices[refined_inliers]] = True
                return PnPRansacResult(pose, output, refined_count, False, True, self.iterations)
        exhausted = self.iterations >= self.maximum_iterations
        if exhausted and self.best_count >= self.minimum_inliers:
            output[self.indices[self.best_inliers]] = True
            return PnPRansacResult(self.best_pose, output, self.best_count, True, True, self.iterations)
        return PnPRansacResult(None, output, 0, exhausted, False, self.iterations)
