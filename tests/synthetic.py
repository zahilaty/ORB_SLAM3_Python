"""Deterministic textured 3D planes and physically consistent IMU test inputs.

This is a renderer for algorithm tests, not a dataset adapter. Textures are
fixed in world coordinates. Ray-plane intersection and depth buffering move
both texture and patch boundaries when the camera translates.
"""

import cv2
import numpy as np

from orbslam3.camera import Pinhole, KannalaBrandt8
from orbslam3.imu import ImuMeasurement
from orbslam3.input import FrameInput, Sensor


class TexturedPlanes:
    def __init__(self, width=320, height=240, seed=314, camera=None):
        self.width, self.height = width, height
        self.camera = camera or Pinhole(260., 260., width / 2., height / 2.)
        self.depths = (3., 4., 5.)
        self.patch_bounds = ((-10., -.45), (-.45, .45), (.45, 10.))
        rng = np.random.default_rng(seed)
        self.margin = width * 2
        shape = height + 2 * self.margin, width + 2 * self.margin
        # Fine texture supplies FAST corners; slight blur makes interpolation
        # behave like a sampled camera rather than changing white noise.
        self.textures = [cv2.GaussianBlur(rng.integers(0, 256, shape, np.uint8), (3, 3), .6)
                         for _ in self.depths]
        self.pixel_x, self.pixel_y = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))

    @staticmethod
    def position(time):
        return np.array([.25 * time + .02 * np.sin(4. * time), 0., 0.])

    @staticmethod
    def velocity(time):
        return np.array([.25 + .08 * np.cos(4. * time), 0., 0.])

    @staticmethod
    def acceleration(time):
        return np.array([-.32 * np.sin(4. * time), 0., 0.])

    def render(self, time, camera_shift=0.):
        center = self.position(time) + [camera_shift, 0., 0.]
        image = np.zeros((self.height, self.width), np.uint8)
        depth = np.full((self.height, self.width), np.inf)
        camera = self.camera
        for z, bounds, texture in zip(self.depths, self.patch_bounds, self.textures):
            world_x = (self.pixel_x - camera.cx) * z / camera.fx + center[0]
            world_y = (self.pixel_y - camera.cy) * z / camera.fy + center[1]
            visible = (world_x >= bounds[0]) & (world_x < bounds[1]) & (z < depth)
            texture_x = (camera.fx * world_x / z + camera.cx + self.margin).astype(np.float32)
            texture_y = (camera.fy * world_y / z + camera.cy + self.margin).astype(np.float32)
            sampled = cv2.remap(texture, texture_x, texture_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            image[visible], depth[visible] = sampled[visible], z
        # A distant back plane fills rays outside the bounded foreground
        # patches, avoiding invalid artificial image borders in small motions.
        absent = ~np.isfinite(depth)
        depth[absent] = 5.
        image[absent] = 0
        return image, depth

    def imu(self, begin, end, frequency=200.):
        # Include both frame boundaries. The tracker intentionally retains an
        # endpoint measurement; adjacent packets repeat only that exact sample.
        first, last = int(round(begin * frequency)), int(round(end * frequency))
        return tuple(ImuMeasurement(index / frequency,
                                    self.acceleration(index / frequency) + [0., 0., 9.81], np.zeros(3))
                     for index in range(first, last + 1))

    def frame(self, sensor, time, previous_time=None, baseline=.1):
        sensor = Sensor(sensor)
        image, depth = self.render(time)
        right = self.render(time, baseline)[0] if sensor.stereo else None
        imu = self.imu(time if previous_time is None else previous_time, time) if sensor.inertial else ()
        return FrameInput(time, image, right, depth if not sensor.monocular and not sensor.stereo else None, imu)

    def camera_pose(self, time):
        pose = np.eye(4)
        pose[:3, 3] = -self.position(time)
        return pose


class FisheyePlanes(TexturedPlanes):
    """World-fixed planes rendered through a nonrectified KB8 stereo rig."""

    def __init__(self, width=320, height=240, seed=2718):
        camera = KannalaBrandt8(210., 205., width / 2., height / 2., .025, -.004, .0006, -.00004)
        super().__init__(width, height, seed, camera)
        # The right optical axis differs by .025 rad about y, so the pipeline
        # must use the rigid-camera triangulation path rather than row matching.
        angle = .025
        self.T_right_left = np.array([[np.cos(angle), 0., np.sin(angle), -.1],
                                      [0., 1., 0., 0.],
                                      [-np.sin(angle), 0., np.cos(angle), 0.],
                                      [0., 0., 0., 1.]])
        pixels = np.stack((self.pixel_x, self.pixel_y), axis=-1)
        self.bearings = self.camera.unproject(pixels)

    def render_pose(self, camera_pose):
        rotation_cw = camera_pose[:3, :3]
        center = -rotation_cw.T @ camera_pose[:3, 3]
        world_rays = self.bearings @ rotation_cw
        image = np.zeros((self.height, self.width), np.uint8)
        depth = np.full((self.height, self.width), np.inf)
        for world_z, bounds, texture in zip(self.depths, self.patch_bounds, self.textures):
            parameter = (world_z - center[2]) / world_rays[..., 2]
            world_points = center + parameter[..., None] * world_rays
            # Bearings have camera z=1, so parameter is also camera depth.
            visible = ((world_points[..., 0] >= bounds[0]) & (world_points[..., 0] < bounds[1])
                       & (parameter > 0.) & (parameter < depth))
            texture_x = (self.camera.fx * world_points[..., 0] / world_z
                         + self.camera.cx + self.margin).astype(np.float32)
            texture_y = (self.camera.fy * world_points[..., 1] / world_z
                         + self.camera.cy + self.margin).astype(np.float32)
            sampled = cv2.remap(texture, texture_x, texture_y, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            image[visible], depth[visible] = sampled[visible], parameter[visible]
        depth[~np.isfinite(depth)] = 5.
        return image, depth

    def render(self, time, camera_shift=0.):
        pose = self.camera_pose(time)
        pose[0, 3] -= camera_shift
        return self.render_pose(pose)

    def frame(self, sensor, time, previous_time=None, baseline=.1):
        sensor = Sensor(sensor)
        left_pose = self.camera_pose(time)
        left, depth = self.render_pose(left_pose)
        right = self.render_pose(self.T_right_left @ left_pose)[0] if sensor.stereo else None
        imu = self.imu(time if previous_time is None else previous_time, time) if sensor.inertial else ()
        return FrameInput(time, left, right, depth if not sensor.monocular and not sensor.stereo else None, imu)
