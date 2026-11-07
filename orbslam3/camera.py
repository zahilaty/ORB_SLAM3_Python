"""Pinhole and Kannala-Brandt8 models from src/CameraModels.

Pinhole factors operate on undistorted pixels; sensor distortion belongs in
the separate input preprocessing. Fisheye factors project distorted pixels.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from dataclasses import dataclass
import numpy as np
from .geometry import skew, triangulate, inverse_pose, transform_points


@dataclass(frozen=True)
class Pinhole:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")

    @property
    def K(self):
        return np.array([[self.fx, 0., self.cx], [0., self.fy, self.cy], [0., 0., 1.]])

    def project(self, point):
        point = np.asarray(point, dtype=float)
        return point[..., :2] / point[..., 2, None] * [self.fx, self.fy] + [self.cx, self.cy]

    def unproject(self, pixel):
        pixel = np.asarray(pixel, dtype=float)
        xy = (pixel - [self.cx, self.cy]) / [self.fx, self.fy]
        return np.concatenate((xy, np.ones(xy.shape[:-1] + (1,))), axis=-1)

    def project_jacobian(self, point):
        x, y, z = point
        return np.array([[self.fx / z, 0, -self.fx * x / z**2],
                         [0, self.fy / z, -self.fy * y / z**2]])

    def uncertainty2(self, pixel):
        return 1.

    def epipolar_constrain(self, other, pixel1, pixel2, rotation12, translation12, variance1, variance2):
        f12 = np.linalg.inv(self.K).T @ skew(translation12) @ rotation12 @ np.linalg.inv(other.K)
        line = np.r_[pixel1, 1.] @ f12
        denominator = line[0]**2 + line[1]**2
        return denominator != 0 and (line @ np.r_[pixel2, 1.])**2 / denominator < 3.84 * variance2


@dataclass(frozen=True)
class KannalaBrandt8(Pinhole):
    k1: float = 0.
    k2: float = 0.
    k3: float = 0.
    k4: float = 0.
    precision: float = 1e-6

    def project(self, point):
        point = np.asarray(point, dtype=float)
        x, y, z = np.moveaxis(point, -1, 0)
        radius = np.hypot(x, y)
        theta = np.arctan2(radius, z)
        t2 = theta**2
        radial = theta * (1 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4))))
        psi = np.arctan2(y, x)
        return np.stack((self.fx * radial * np.cos(psi) + self.cx,
                         self.fy * radial * np.sin(psi) + self.cy), axis=-1)

    def unproject(self, pixel):
        pixel = np.asarray(pixel, dtype=float)
        xy = (pixel - [self.cx, self.cy]) / [self.fx, self.fy]
        theta_d = np.clip(np.linalg.norm(xy, axis=-1), -np.pi / 2, np.pi / 2)
        theta = np.array(theta_d, copy=True)
        active = theta_d > 1e-8
        for _ in range(10):
            t2 = theta**2
            a, b, c, d = self.k1 * t2, self.k2 * t2**2, self.k3 * t2**3, self.k4 * t2**4
            delta = (theta * (1 + a + b + c + d) - theta_d) / (1 + 3*a + 5*b + 7*c + 9*d)
            theta = np.where(active, theta - delta, theta)
            active = active & (np.abs(delta) >= self.precision)
            if not np.any(active):
                break
        scale = np.ones_like(theta_d)
        np.divide(np.tan(theta), theta_d, out=scale, where=theta_d > 1e-8)
        return np.concatenate((xy * scale[..., None], np.ones(xy.shape[:-1] + (1,))), axis=-1)

    def project_jacobian(self, point):
        x, y, z = point
        radius2 = x*x + y*y
        if radius2 < 1e-16 and z > 0:
            # Continuous optical-axis limit; source formula evaluates 0/0 here.
            return np.array([[self.fx / z, 0., 0.], [0., self.fy / z, 0.]])
        radius = np.sqrt(radius2)
        theta = np.arctan2(radius, z)
        t2 = theta**2
        radial = theta * (1 + self.k1*t2 + self.k2*t2**2 + self.k3*t2**3 + self.k4*t2**4)
        derivative = 1 + 3*self.k1*t2 + 5*self.k2*t2**2 + 7*self.k3*t2**3 + 9*self.k4*t2**4
        total2 = radius2 + z*z
        cross = derivative*z*x*y/(radius2*total2) - radial*x*y/radius**3
        return np.array([[self.fx*(derivative*z*x*x/(radius2*total2) + radial*y*y/radius**3),
                          self.fx*cross, -self.fx*derivative*x/total2],
                         [self.fy*cross, self.fy*(derivative*z*y*y/(radius2*total2) + radial*x*x/radius**3),
                          -self.fy*derivative*y/total2]])

    def epipolar_constrain(self, other, pixel1, pixel2, rotation12, translation12, variance1, variance2):
        t12 = np.eye(4)
        t12[:3, :3], t12[:3, 3] = rotation12, translation12
        point = triangulate_match(self, other, pixel1, pixel2, np.eye(4), inverse_pose(t12), variance1, variance2)
        return point is not None and point[2] > .0001


def triangulate_match(camera1, camera2, pixel1, pixel2, pose1, pose2, variance1=1., variance2=1.):
    """KB8 matchAndtriangulate checks, also usable with pinhole bearings."""
    ray1, ray2 = camera1.unproject(pixel1), camera2.unproject(pixel2)
    world_ray1, world_ray2 = pose1[:3, :3].T @ ray1, pose2[:3, :3].T @ ray2
    cosine = world_ray1 @ world_ray2 / (np.linalg.norm(world_ray1) * np.linalg.norm(world_ray2))
    if cosine > .9998:
        return None
    point = triangulate(ray1, ray2, pose1[:3], pose2[:3])
    if point is None:
        return None
    for camera, pixel, pose, variance in ((camera1, pixel1, pose1, variance1), (camera2, pixel2, pose2, variance2)):
        local = transform_points(pose, point)
        if local[2] <= 0 or np.sum((camera.project(local) - pixel)**2) > 5.991 * variance:
            return None
    return point
