"""Lie operations and projective geometry used by the source algorithms.

Poses T_cw map world to camera: p_c = R_cw p_w + t_cw. Tangents use
g2o's rotation-then-translation ordering (Sophus itself uses the reverse).
Source: GeometricTools.cc, Sophus SO3/SE3, Thirdparty/g2o/types/sim3.h.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


def skew(vector):
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def so3_exp(omega):
    omega = np.asarray(omega, dtype=float)
    theta2 = omega @ omega
    W = skew(omega)
    if theta2 < 1e-12:
        a = 1 - theta2 / 6 + theta2 * theta2 / 120
        b = .5 - theta2 / 24 + theta2 * theta2 / 720
    else:
        theta = np.sqrt(theta2)
        a, b = np.sin(theta) / theta, (1 - np.cos(theta)) / theta2
    return np.eye(3) + a * W + b * W @ W


def so3_log(rotation):
    # Rotation's logarithm also handles the pi singularity, unlike acos-only code.
    return Rotation.from_matrix(rotation).as_rotvec()


def right_jacobian_so3(omega):
    omega = np.asarray(omega, dtype=float)
    theta2 = omega @ omega
    W = skew(omega)
    if theta2 < 1e-12:
        return np.eye(3) - .5 * W + W @ W / 6
    theta = np.sqrt(theta2)
    return np.eye(3) - (1 - np.cos(theta)) / theta2 * W + (theta - np.sin(theta)) / theta**3 * W @ W


def inverse_right_jacobian_so3(omega):
    omega = np.asarray(omega, dtype=float)
    theta2 = omega @ omega
    W = skew(omega)
    if theta2 < 1e-12:
        return np.eye(3) + .5 * W + W @ W / 12
    theta = np.sqrt(theta2)
    return np.eye(3) + .5 * W + (1 / theta2 - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * W @ W


def se3_exp(tangent):
    omega, translation = np.asarray(tangent, dtype=float).reshape(2, 3)
    pose = np.eye(4)
    pose[:3, :3] = so3_exp(omega)
    pose[:3, 3] = right_jacobian_so3(-omega) @ translation
    return pose


def se3_log(pose):
    omega = so3_log(pose[:3, :3])
    return np.r_[omega, inverse_right_jacobian_so3(-omega) @ pose[:3, 3]]


def make_pose(rotation=None, translation=None):
    pose = np.eye(4)
    if rotation is not None:
        pose[:3, :3] = rotation
    if translation is not None:
        pose[:3, 3] = translation
    return pose


def inverse_pose(pose):
    rotation = pose[:3, :3].T
    return make_pose(rotation, -rotation @ pose[:3, 3])


def transform_points(pose, points):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def triangulate(ray1, ray2, projection1, projection2):
    """Source DLT; rays can be normalized bearings or homogeneous pixels.

    The corresponding matrices must use the same coordinate system. Returns
    None for a homogeneous point at infinity rather than dividing by zero.
    """
    a = np.stack((ray1[0] * projection1[2] - projection1[0],
                  ray1[1] * projection1[2] - projection1[1],
                  ray2[0] * projection2[2] - projection2[0],
                  ray2[1] * projection2[2] - projection2[1]))
    _, _, vt = np.linalg.svd(a)
    homogeneous = vt[-1]
    if homogeneous[3] == 0:
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.isfinite(point).all() else None


def fundamental(camera1, pose1, camera2, pose2):
    """F12: pixel1.T @ F12 @ pixel2 = 0, as in ComputeF12."""
    relative = pose1 @ inverse_pose(pose2)
    return np.linalg.inv(camera1.K).T @ skew(relative[:3, 3]) @ relative[:3, :3] @ np.linalg.inv(camera2.K)


@dataclass
class Sim3:
    """p_out = scale * rotation @ p_in + translation; g2o tangent order."""
    rotation: np.ndarray
    translation: np.ndarray
    scale: float = 1.

    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=float).reshape(3)
        if self.scale <= 0:
            raise ValueError("Sim3 scale must be positive")

    def map(self, points):
        return self.scale * np.asarray(points) @ self.rotation.T + self.translation

    def inverse(self):
        r = self.rotation.T
        return Sim3(r, -r @ self.translation / self.scale, 1 / self.scale)

    def __matmul__(self, other):
        return Sim3(self.rotation @ other.rotation,
                    self.scale * self.rotation @ other.translation + self.translation,
                    self.scale * other.scale)

    @classmethod
    def exp(cls, tangent):
        omega, translation, sigma = tangent[:3], tangent[3:6], tangent[6]
        algebra = np.zeros((4, 4))
        algebra[:3, :3] = skew(omega) + np.eye(3) * sigma
        algebra[:3, 3] = translation
        matrix = expm(algebra)
        scale = np.exp(sigma)
        return cls(matrix[:3, :3] / scale, matrix[:3, 3], scale)

    def log(self):
        omega, sigma = so3_log(self.rotation), np.log(self.scale)
        # W = integral_0^1 exp((hat(omega)+sigma*I)*t) dt. The upper
        # right block computes this integral even when its generator is
        # singular. Taking real(logm(S)) would choose the wrong branch at pi.
        augmented = np.zeros((6, 6))
        augmented[:3, :3] = skew(omega) + sigma*np.eye(3)
        augmented[:3, 3:] = np.eye(3)
        jacobian = expm(augmented)[:3, 3:]
        return np.r_[omega, np.linalg.solve(jacobian, self.translation), sigma]

    def camera_pose(self):
        """Equivalent SE3 camera pose after dividing Sim3 translation by scale."""
        return make_pose(self.rotation, self.translation / self.scale)
