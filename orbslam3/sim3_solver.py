"""Horn's three-point Sim(3) RANSAC from src/Sim3Solver.cc.

Inputs are corresponding 3D map points expressed in their two camera frames.
The estimated S12 maps camera 2 to camera 1. This is the source's quaternion
eigenproblem, not an SVD alignment substitution. Scene/map filtering remains
with the caller so invalid or missing map points never enter the solver.

Derived from ORB-SLAM3, Copyright (C) 2017-2021 Carlos Campos, Richard Elvira,
Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of
Zaragoza; Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D.
Tardós, University of Zaragoza. GPL-3.0-or-later; see LICENSE.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import Sim3


def horn_sim3(points1, points2, fix_scale=False):
    """Solve Horn's quaternion eigenproblem for S12 (three or more pairs)."""
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)
    if points1.shape != points2.shape or points1.ndim != 2 or points1.shape[1] != 3:
        raise ValueError("Matched point arrays must both be Nx3")
    if len(points1) < 3:
        raise ValueError("Horn alignment needs at least three matched points")
    centroid1, centroid2 = points1.mean(axis=0), points2.mean(axis=0)
    centered1, centered2 = points1 - centroid1, points2 - centroid2
    covariance = centered2.T @ centered1
    m = covariance
    quaternion_matrix = np.array([
        [np.trace(m), m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]],
        [m[1, 2] - m[2, 1], m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]],
        [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], -m[0, 0] + m[1, 1] - m[2, 2], m[1, 2] + m[2, 1]],
        [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], -m[0, 0] - m[1, 1] + m[2, 2]],
    ])
    _, eigenvectors = np.linalg.eigh(quaternion_matrix)
    w, x, y, z = eigenvectors[:, -1]
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    rotated2 = centered2 @ rotation.T
    denominator = np.sum(rotated2 * rotated2)
    if denominator <= np.finfo(float).eps or np.linalg.matrix_rank(centered2) < 2:
        raise ValueError("Degenerate three-point alignment")
    scale = 1.0 if fix_scale else np.sum(centered1 * rotated2) / denominator
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The fitted similarity has nonpositive scale")
    translation = centroid1 - scale * rotation @ centroid2
    return Sim3(rotation, translation, scale)


@dataclass
class Sim3RansacResult:
    transform: Sim3 | None
    inliers: np.ndarray
    count: int
    converged: bool
    exhausted: bool
    iterations: int


class Sim3Solver:
    """Incremental RANSAC with the exact source's support/threshold decisions.

    ``original_indices`` maps prefiltered correspondence rows back to the
    original matching vector. ``output_size`` is that vector's length. The
    octave variances belong to the keyframes where each point is observed.
    NumPy's seeded RNG replaces DUtils random integers; sampled sets differ.
    """

    def __init__(self, points1, points2, camera1, camera2, variances1=None,
                 variances2=None, *, fix_scale=True, original_indices=None,
                 output_size=None, seed=None):
        self.points1 = np.asarray(points1, dtype=float)
        self.points2 = np.asarray(points2, dtype=float)
        if self.points1.shape != self.points2.shape or self.points1.ndim != 2 or self.points1.shape[1] != 3:
            raise ValueError("Matched camera-frame point arrays must be Nx3")
        self.camera1, self.camera2 = camera1, camera2
        count = len(self.points1)
        variances1 = np.ones(count) if variances1 is None else np.asarray(variances1)
        variances2 = np.ones(count) if variances2 is None else np.asarray(variances2)
        if variances1.shape != (count,) or variances2.shape != (count,):
            raise ValueError("Provide one pyramid variance per correspondence")
        # The source stores these in vector<size_t>, so fractional thresholds
        # are truncated despite being computed from 9.210 * float variance.
        self.thresholds1 = np.trunc(9.210 * variances1)
        self.thresholds2 = np.trunc(9.210 * variances2)
        self.projections1 = np.asarray([camera1.project(p) for p in self.points1])
        self.projections2 = np.asarray([camera2.project(p) for p in self.points2])
        self.fix_scale = fix_scale
        self.original_indices = np.arange(count) if original_indices is None else np.asarray(original_indices, dtype=int)
        if self.original_indices.shape != (count,) or (self.original_indices < 0).any():
            raise ValueError("Original indices must have one nonnegative entry per match")
        self.output_size = count if output_size is None else output_size
        if count and self.original_indices.max() >= self.output_size:
            raise ValueError("Original index exceeds output matching vector size")
        self.rng = np.random.default_rng(seed)
        self.best_transform = None
        self.best_count = 0
        self.best_inliers = np.zeros(count, dtype=bool)
        self.set_ransac_parameters()

    def set_ransac_parameters(self, probability=.99, minimum_inliers=6, maximum_iterations=300):
        if not 0 < probability < 1 or minimum_inliers < 3 or maximum_iterations < 1:
            raise ValueError("Invalid RANSAC probability, support, or iteration budget")
        self.minimum_inliers = minimum_inliers
        count = len(self.points1)
        if count <= minimum_inliers:
            iterations = 1
        else:
            epsilon = minimum_inliers / count
            iterations = int(np.ceil(np.log1p(-probability) / np.log1p(-epsilon**3)))
        self.maximum_iterations = max(1, min(iterations, maximum_iterations))
        self.iterations = 0

    def check_inliers(self, transform):
        projected1 = np.asarray([self.camera1.project(p) for p in transform.map(self.points2)])
        projected2 = np.asarray([self.camera2.project(p) for p in transform.inverse().map(self.points1)])
        errors1 = np.sum((self.projections1 - projected1)**2, axis=1)
        errors2 = np.sum((self.projections2 - projected2)**2, axis=1)
        # Source does not add a depth-positive check. Comparisons exclude NaNs.
        return (errors1 < self.thresholds1) & (errors2 < self.thresholds2)

    def iterate(self, iterations=5):
        output_inliers = np.zeros(self.output_size, dtype=bool)
        if len(self.points1) < self.minimum_inliers:
            return Sim3RansacResult(None, output_inliers, 0, False, True, self.iterations)
        for _ in range(min(iterations, self.maximum_iterations - self.iterations)):
            self.iterations += 1
            # DUtils removes by swapping the selected entry with the last one.
            available = list(range(len(self.points1)))
            sample = []
            for _ in range(3):
                index = int(self.rng.integers(len(available)))
                sample.append(available[index])
                available[index] = available[-1]
                available.pop()
            try:
                transform = horn_sim3(self.points1[sample], self.points2[sample], self.fix_scale)
            except (ValueError, np.linalg.LinAlgError):
                continue
            inliers = self.check_inliers(transform)
            count = int(inliers.sum())
            if count >= self.best_count:
                self.best_transform, self.best_count = transform, count
                self.best_inliers = inliers
                # Strict greater-than is intentional: source's equality never
                # returns a converged solution, even when all N == min agree.
                if count > self.minimum_inliers:
                    output_inliers[self.original_indices[inliers]] = True
                    return Sim3RansacResult(transform, output_inliers, count, True,
                                             False, self.iterations)
        return Sim3RansacResult(self.best_transform, output_inliers, 0, False,
                                 self.iterations >= self.maximum_iterations, self.iterations)

    def find(self):
        return self.iterate(self.maximum_iterations)
