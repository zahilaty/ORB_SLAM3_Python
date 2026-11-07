"""Two-view initialization, ported from TwoViewReconstruction.cc.

Both models use the SAME 200 sets of eight correspondences; scoring is
symmetric and the source's hypothesis ambiguity/parallax tests are explicit.
No findHomography/findEssentialMat/recoverPose shortcuts are used.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from dataclasses import dataclass
import numpy as np
from .geometry import make_pose, triangulate


@dataclass
class Reconstruction:
    success: bool
    pose: np.ndarray | None
    points: np.ndarray
    triangulated: np.ndarray
    model: str
    homography_score: float
    fundamental_score: float
    inliers: np.ndarray
    parallax: float = 0.


def normalize_points(points):
    """Source uses separate mean absolute deviations, not Hartley RMS."""
    mean = points.mean(axis=0)
    deviation = np.abs(points - mean).mean(axis=0)
    if np.any(deviation == 0):
        raise ValueError("Degenerate two-view coordinates")
    scales = 1 / deviation
    matrix = np.array([[scales[0], 0., -mean[0]*scales[0]],
                       [0., scales[1], -mean[1]*scales[1]], [0., 0., 1.]])
    return (points - mean) * scales, matrix


def compute_homography(points1, points2):
    count = len(points1)
    x, y = points1.T
    u, v = points2.T
    matrix = np.zeros((2 * count, 9))
    matrix[0::2, 3:6] = -np.c_[x, y, np.ones(count)]
    matrix[0::2, 6:] = np.c_[v*x, v*y, v]
    matrix[1::2, :3] = np.c_[x, y, np.ones(count)]
    matrix[1::2, 6:] = -np.c_[u*x, u*y, u]
    return np.linalg.svd(matrix, full_matrices=True)[2][-1].reshape(3, 3)


def compute_fundamental(points1, points2):
    x, y = points1.T
    u, v = points2.T
    matrix = np.c_[u*x, u*y, u, v*x, v*y, v, x, y, np.ones(len(x))]
    estimate = np.linalg.svd(matrix, full_matrices=True)[2][-1].reshape(3, 3)
    left, values, right = np.linalg.svd(estimate)
    values[-1] = 0
    return left @ np.diag(values) @ right


def score_homography(h21, points1, points2, sigma):
    h12 = np.linalg.inv(h21)
    homogeneous1, homogeneous2 = np.c_[points1, np.ones(len(points1))], np.c_[points2, np.ones(len(points2))]
    projected1, projected2 = homogeneous2 @ h12.T, homogeneous1 @ h21.T
    with np.errstate(divide="ignore", invalid="ignore"):
        errors1 = np.sum((points1 - projected1[:, :2] / projected1[:, 2, None])**2, axis=1) / sigma**2
        errors2 = np.sum((points2 - projected2[:, :2] / projected2[:, 2, None])**2, axis=1) / sigma**2
    accepted1, accepted2 = errors1 <= 5.991, errors2 <= 5.991
    score = np.sum(5.991 - errors1[accepted1]) + np.sum(5.991 - errors2[accepted2])
    return score, accepted1 & accepted2


def score_fundamental(f21, points1, points2, sigma):
    homogeneous1, homogeneous2 = np.c_[points1, np.ones(len(points1))], np.c_[points2, np.ones(len(points2))]
    lines2, lines1 = homogeneous1 @ f21.T, homogeneous2 @ f21
    numerator = np.sum(homogeneous2 * lines2, axis=1)**2
    with np.errstate(divide="ignore", invalid="ignore"):
        errors1 = numerator / np.sum(lines2[:, :2]**2, axis=1) / sigma**2
        errors2 = numerator / np.sum(lines1[:, :2]**2, axis=1) / sigma**2
    accepted1, accepted2 = errors1 <= 3.841, errors2 <= 3.841
    score = np.sum(5.991 - errors1[accepted1]) + np.sum(5.991 - errors2[accepted2])
    return score, accepted1 & accepted2


def decompose_essential(essential):
    left, _, right = np.linalg.svd(essential)
    translation = left[:, 2] / np.linalg.norm(left[:, 2])
    w = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotation1, rotation2 = left @ w @ right, left @ w.T @ right
    if np.linalg.det(rotation1) < 0:
        rotation1 = -rotation1
    if np.linalg.det(rotation2) < 0:
        rotation2 = -rotation2
    return [(rotation1, translation), (rotation2, translation),
            (rotation1, -translation), (rotation2, -translation)]


def decompose_homography(h21, calibration):
    """Faugeras' eight hypotheses; no library decomposition changes."""
    calibrated = np.linalg.inv(calibration) @ h21 @ calibration
    left, (d1, d2, d3), right = np.linalg.svd(calibrated)
    if d2 == 0 or d3 == 0 or d1 / d2 < 1.00001 or d2 / d3 < 1.00001:
        return []
    sign = np.linalg.det(left) * np.linalg.det(right)
    aux1 = np.sqrt((d1*d1 - d2*d2) / (d1*d1 - d3*d3))
    aux3 = np.sqrt((d2*d2 - d3*d3) / (d1*d1 - d3*d3))
    x1, x3 = [aux1, aux1, -aux1, -aux1], [aux3, -aux3, aux3, -aux3]
    root = np.sqrt((d1*d1-d2*d2) * (d2*d2-d3*d3))
    hypotheses = []
    for negative in (False, True):
        denominator = (d1 - d3 if negative else d1 + d3) * d2
        cosine = (d1*d3 - d2*d2 if negative else d2*d2 + d1*d3) / denominator
        sine = root / denominator
        for i, st in enumerate((sine, -sine, -sine, sine)):
            if negative:
                rp = np.array([[cosine, 0., st], [0., -1., 0.], [st, 0., -cosine]])
                tp = np.array([x1[i], 0., x3[i]]) * (d1 + d3)
            else:
                rp = np.array([[cosine, 0., -st], [0., 1., 0.], [st, 0., cosine]])
                tp = np.array([x1[i], 0., -x3[i]]) * (d1 - d3)
            translation = left @ tp
            hypotheses.append((sign * left @ rp @ right, translation / np.linalg.norm(translation)))
    return hypotheses


def check_motion(rotation, translation, keys1, keys2, pairs, inliers, calibration, threshold2):
    points = np.zeros((len(keys1), 3))
    good = np.zeros(len(keys1), dtype=bool)
    projection1 = calibration @ np.eye(4)[:3]
    projection2 = calibration @ np.c_[rotation, translation]
    center2 = -rotation.T @ translation
    cosines = []
    for (i, j), inlier in zip(pairs, inliers):
        if not inlier:
            continue
        point = triangulate(keys1[i], keys2[j], projection1, projection2)
        if point is None:
            continue
        normal1, normal2 = point, point - center2
        cosine = normal1 @ normal2 / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
        local2 = rotation @ point + translation
        if (point[2] <= 0 or local2[2] <= 0) and cosine < .99998:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            projected1, projected2 = calibration @ point, calibration @ local2
            error1 = np.sum((projected1[:2] / projected1[2] - keys1[i])**2)
            error2 = np.sum((projected2[:2] / projected2[2] - keys2[j])**2)
        if not np.isfinite([error1, error2]).all() or error1 > threshold2 or error2 > threshold2:
            continue
        cosines.append(cosine)
        points[i] = point
        good[i] = cosine < .99998
    parallax = 0.
    if cosines:
        cosines.sort()
        parallax = np.degrees(np.arccos(np.clip(cosines[min(50, len(cosines)-1)], -1., 1.)))
    return len(cosines), points, good, parallax


class TwoViewReconstruction:
    def __init__(self, calibration, sigma=1., iterations=200, seed=0):
        self.calibration = np.asarray(calibration, dtype=float)
        self.sigma, self.iterations, self.seed = sigma, iterations, seed

    def reconstruct(self, keys1, keys2, matches12):
        keys1, keys2 = np.asarray(keys1, dtype=float), np.asarray(keys2, dtype=float)
        matches12 = np.asarray(matches12, dtype=int)
        valid = np.flatnonzero(matches12 >= 0)
        if len(matches12) != len(keys1) or np.any(matches12[valid] >= len(keys2)):
            raise ValueError("matches12 must index second-view keys once for each first-view key")
        pairs = np.c_[valid, matches12[valid]]
        result = Reconstruction(False, None, np.zeros((len(keys1), 3)), np.zeros(len(keys1), bool), "none", 0., 0., np.zeros(len(pairs), bool))
        if len(pairs) < 8:
            return result
        try:
            normalized1, t1 = normalize_points(keys1)
            normalized2, t2 = normalize_points(keys2)
        except ValueError:
            return result
        t2inv = np.linalg.inv(t2)
        rng = np.random.default_rng(self.seed)
        best_h = best_f = None
        best_hi = best_fi = result.inliers
        points1, points2 = keys1[pairs[:, 0]], keys2[pairs[:, 1]]
        for _ in range(self.iterations):
            sample = pairs[rng.choice(len(pairs), 8, replace=False)]
            p1, p2 = normalized1[sample[:, 0]], normalized2[sample[:, 1]]
            homography = t2inv @ compute_homography(p1, p2) @ t1
            try:
                score, inliers = score_homography(homography, points1, points2, self.sigma)
                if score > result.homography_score:
                    result.homography_score, best_h, best_hi = score, homography, inliers
            except np.linalg.LinAlgError:
                pass
            fundamental = t2.T @ compute_fundamental(p1, p2) @ t1
            score, inliers = score_fundamental(fundamental, points1, points2, self.sigma)
            if score > result.fundamental_score:
                result.fundamental_score, best_f, best_fi = score, fundamental, inliers
        total = result.homography_score + result.fundamental_score
        if total == 0:
            return result
        use_h = result.homography_score / total > .50
        result.model = "homography" if use_h else "fundamental"
        result.inliers = best_hi if use_h else best_fi
        if use_h:
            hypotheses = decompose_homography(best_h, self.calibration)
        else:
            hypotheses = decompose_essential(self.calibration.T @ best_f @ self.calibration)
        evaluated = [check_motion(r, t, keys1, keys2, pairs, result.inliers, self.calibration, 4*self.sigma**2)
                     for r, t in hypotheses]
        if not evaluated:
            return result
        counts = np.array([entry[0] for entry in evaluated])
        best = int(np.argmax(counts))
        count, points, good, parallax = evaluated[best]
        result.parallax = parallax
        n = result.inliers.sum()
        if use_h:
            second = np.sort(counts)[-2]
            accepted = second < .75*count and parallax >= 1. and count > 50 and count > .9*n
        else:
            accepted = count >= max(int(.9*n), 50) and np.sum(counts > .7*count) <= 1 and parallax > 1.
        if accepted:
            result.success = True
            result.pose = make_pose(*hypotheses[best])
            # ReconstructH in this revision computes bestP3D but forgets to assign
            # vP3D. Return it here; this explicit source defect correction is in COVERAGE.
            result.points, result.triangulated = points, good
        return result
