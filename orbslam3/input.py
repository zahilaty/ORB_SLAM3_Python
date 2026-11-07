"""General user-data input and frame preparation; no dataset conventions.

Frame depth/stereo preprocessing follows Frame.cc. All timestamps are seconds,
depths become metres, T_right_left maps left camera into right camera, and
IMU acceleration/angular velocity use m/s² and rad/s respectively.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import cv2
import numpy as np
from .camera import Pinhole, KannalaBrandt8, triangulate_match
from .features import ORBExtractor, FeatureSet, hamming
from .map import Frame


class Sensor(Enum):
    MONOCULAR = "monocular"
    STEREO = "stereo"
    RGBD = "rgbd"
    MONOCULAR_IMU = "monocular_imu"
    STEREO_IMU = "stereo_imu"
    RGBD_IMU = "rgbd_imu"

    @property
    def inertial(self):
        return self.value.endswith("_imu")

    @property
    def monocular(self):
        return self in (Sensor.MONOCULAR, Sensor.MONOCULAR_IMU)

    @property
    def stereo(self):
        return self in (Sensor.STEREO, Sensor.STEREO_IMU)


@dataclass
class Calibration:
    camera: Pinhole
    baseline_fx: float = 0.
    depth_threshold: float = 0.
    distortion: tuple = ()
    depth_units_per_meter: float = 1.
    frames_per_second: float = 30.
    color_order: str = "RGB"
    right_camera: Pinhole | None = None
    T_right_left: np.ndarray | None = None
    overlap_left: tuple | None = None
    overlap_right: tuple | None = None
    imu: object | None = None

    def __post_init__(self):
        self.distortion = tuple(self.distortion)
        if self.frames_per_second <= 0 or self.depth_units_per_meter <= 0:
            raise ValueError("Frame rate and depth units per metre must be positive")
        if self.color_order not in ("RGB", "BGR"):
            raise ValueError("color_order must be RGB or BGR")
        if self.right_camera is not None:
            if self.T_right_left is None:
                raise ValueError("A second camera requires T_right_left")
            self.T_right_left = np.asarray(self.T_right_left, dtype=float)
            if self.T_right_left.shape != (4, 4):
                raise ValueError("T_right_left must be 4x4")
        if self.distortion and isinstance(self.camera, KannalaBrandt8):
            raise ValueError("KB8 distortion is already included in the camera model")


@dataclass
class FrameInput:
    timestamp: float
    image: np.ndarray
    right_image: np.ndarray | None = None
    depth: np.ndarray | None = None
    imu: tuple = ()


def grayscale(image, color_order="RGB"):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError("Images must be uint8")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = {("RGB", 3): cv2.COLOR_RGB2GRAY, ("BGR", 3): cv2.COLOR_BGR2GRAY,
                ("RGB", 4): cv2.COLOR_RGBA2GRAY, ("BGR", 4): cv2.COLOR_BGRA2GRAY}[color_order, image.shape[2]]
        return cv2.cvtColor(image, code)
    raise ValueError("Image must be grayscale, RGB/BGR, or RGBA/BGRA")


def undistort_features(features, camera, distortion):
    # Source short circuits on k1, even if higher coefficients are nonzero.
    if not distortion or distortion[0] == 0 or not len(features):
        return features
    points = cv2.undistortPoints(features.points[:, None].astype(np.float32), camera.K, np.asarray(distortion), P=camera.K)[:, 0]
    return replace(features, points=points)


def image_bounds(width, height, camera, distortion):
    if not distortion or distortion[0] == 0:
        return 0., float(width), 0., float(height)
    corners = np.array([[[0., 0.]], [[width, 0.]], [[0., height]], [[width, height]]], np.float32)
    corners = cv2.undistortPoints(corners, camera.K, np.asarray(distortion), P=camera.K)[:, 0]
    return min(corners[0, 0], corners[2, 0]), max(corners[1, 0], corners[3, 0]), min(corners[0, 1], corners[1, 1]), max(corners[2, 1], corners[3, 1])


def concatenate_features(left, right):
    return FeatureSet(*(np.concatenate((getattr(left, name), getattr(right, name)))
                        for name in ("points", "descriptors", "octaves", "angles", "responses", "sizes")), mono_count=left.mono_count)


def rectified_stereo(left, right, pyramid_left, pyramid_right, scales, baseline_fx, fx):
    """Frame::ComputeStereoMatches: descriptor/row search then L1 patches.

    Pyramids are the unblurred extraction images. The active source uses raw
    patch intensities, not zero-mean patches from older ORB-SLAM versions.
    """
    n = len(left)
    right_u, depth = np.full(n, -1.), np.full(n, -1.)
    rows = [[] for _ in range(pyramid_left[0].shape[0])]
    for i, ((_, y), octave) in enumerate(zip(right.points, right.octaves)):
        radius = 2*scales[octave]
        for row in range(max(0, int(np.floor(y-radius))), min(len(rows), int(np.ceil(y+radius))+1)):
            rows[row].append(i)
    correlations = []
    for i, ((u, v), level) in enumerate(zip(left.points, left.octaves)):
        candidates = np.array(rows[int(v)], dtype=int)
        if not len(candidates):
            continue
        candidates = candidates[(np.abs(right.octaves[candidates]-int(level)) <= 1)
                                & (right.points[candidates, 0] >= u-fx) & (right.points[candidates, 0] <= u)]
        if not len(candidates):
            continue
        distances = hamming(left.descriptors[i], right.descriptors[candidates])
        best = np.argmin(distances)
        if distances[best] >= 75:
            continue
        j = candidates[best]
        ul, vl, ur = (int(np.floor(float(value/scales[level])+.5)) for value in (u, v, right.points[j, 0]))
        il, ir = pyramid_left[level], pyramid_right[level]
        if ul < 5 or vl < 5 or ul+6 > il.shape[1] or vl+6 > il.shape[0] or ur-10 < 0 or ur+11 >= ir.shape[1]:
            continue
        patch = il[vl-5:vl+6, ul-5:ul+6].astype(np.int16)
        costs = np.array([np.abs(patch-ir[vl-5:vl+6, ur+offset-5:ur+offset+6].astype(np.int16)).sum() for offset in range(-5, 6)])
        peak = int(np.argmin(costs))
        if peak in (0, 10):
            continue
        a, b, c = costs[peak-1:peak+2]
        denominator = 2.*(a+c-2*b)
        if denominator == 0:
            continue
        offset = (a-c)/denominator
        if offset < -1 or offset > 1:
            continue
        refined = scales[level]*(ur+peak-5+offset)
        disparity = u-refined
        if 0 <= disparity < fx:
            if disparity <= 0:
                disparity, refined = .01, u-.01
            depth[i], right_u[i] = baseline_fx/disparity, refined
            correlations.append((int(costs[peak]), i))
    if correlations:
        correlations.sort()
        threshold = 1.5*1.4*correlations[len(correlations)//2][0]
        for distance, i in reversed(correlations):
            if distance < threshold:
                break
            right_u[i] = depth[i] = -1
    # Original indexes the median of an empty vector. Empty matches are valid
    # input here; avoiding that undefined behavior is an explicit deviation.
    return right_u, depth


class FrameBuilder:
    def __init__(self, sensor, calibration, nfeatures=1000, scale_factor=1.2, nlevels=8, initial_fast=20, minimum_fast=7):
        self.sensor, self.calibration = Sensor(sensor), calibration
        options = dict(nfeatures=nfeatures, scale_factor=scale_factor, nlevels=nlevels,
                       initial_fast=initial_fast, minimum_fast=minimum_fast)
        self.extractor = ORBExtractor(**options)
        self.right_extractor = ORBExtractor(**options) if self.sensor.stereo else None
        self.initial_extractor = ORBExtractor(**(options | {"nfeatures": 5*nfeatures})) if self.sensor.monocular else None
        self.next_id = 0
        if self.sensor.inertial and calibration.imu is None:
            raise ValueError("Inertial sensor configurations require IMU calibration")
        if not self.sensor.monocular and calibration.right_camera is None and calibration.baseline_fx <= 0:
            raise ValueError("Stereo and RGB-D require a positive baseline_fx")

    def build(self, data, initializing=False):
        calibration = self.calibration
        if not np.isfinite(data.timestamp):
            raise ValueError("Frame timestamp must be finite")
        gray = grayscale(data.image, calibration.color_order)
        extractor = self.initial_extractor if initializing and self.initial_extractor is not None else self.extractor
        overlap = calibration.overlap_left
        if calibration.right_camera is not None and overlap is None:
            overlap = (0, gray.shape[1]-1)
        raw = extractor.extract(gray, overlap=overlap)
        features = undistort_features(raw, calibration.camera, calibration.distortion)
        frame = Frame(self.next_id, float(data.timestamp), features, calibration.camera, gray.shape[1], gray.shape[0],
                      extractor.scales, extractor.inverse_variances, baseline_fx=calibration.baseline_fx,
                      depth_threshold=calibration.depth_threshold, raw_features=raw,
                      bounds=image_bounds(gray.shape[1], gray.shape[0], calibration.camera, calibration.distortion))
        if self.sensor.stereo:
            if data.right_image is None:
                raise ValueError("Stereo frame requires right_image")
            right_gray = grayscale(data.right_image, calibration.color_order)
            if right_gray.shape != gray.shape:
                raise ValueError("Left and right image dimensions must agree")
            overlap = calibration.overlap_right
            if calibration.right_camera is not None and overlap is None:
                overlap = (0, right_gray.shape[1]-1)
            right = self.right_extractor.extract(right_gray, overlap=overlap)
            if calibration.right_camera is None:
                frame.right_u, frame.depth = rectified_stereo(raw, right, extractor.pyramid, self.right_extractor.pyramid,
                                                             frame.scales, calibration.baseline_fx, calibration.camera.fx)
            else:
                frame = self._unrectified(frame, right, calibration)
        elif not self.sensor.monocular:
            if data.depth is None or np.shape(data.depth) != gray.shape:
                raise ValueError("RGB-D frame needs an image-aligned depth map")
            depths = np.asarray(data.depth, dtype=float)/calibration.depth_units_per_meter
            indices = raw.points.astype(int)
            values = depths[indices[:, 1], indices[:, 0]]
            valid = np.isfinite(values) & (values > 0)
            frame.depth[valid] = values[valid]
            frame.right_u[valid] = features.points[valid, 0]-calibration.baseline_fx/values[valid]
        self.next_id += 1
        return frame

    def _unrectified(self, frame, right, calibration):
        left = frame.raw_features
        nleft, n = len(left), len(left)+len(right)
        combined = concatenate_features(left, right)
        matches, depth = np.full(n, -1, dtype=int), np.full(n, -1.)
        points = np.full((nleft, 3), np.nan)
        # This knn matcher is exactly the one used in ComputeStereoFishEyeMatches.
        if len(right)-right.mono_count >= 2 and len(left)-left.mono_count:
            knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left.descriptors[left.mono_count:], right.descriptors[right.mono_count:], k=2)
            for neighbors in knn:
                if len(neighbors) < 2 or neighbors[0].distance >= .7*neighbors[1].distance:
                    continue
                first = neighbors[0]
                i, j = first.queryIdx+left.mono_count, first.trainIdx+right.mono_count
                point = triangulate_match(calibration.camera, calibration.right_camera, left.points[i], right.points[j],
                                          np.eye(4), calibration.T_right_left, frame.scales[left.octaves[i]]**2,
                                          frame.scales[right.octaves[j]]**2)
                if point is not None and point[2] > .0001:
                    points[i], depth[i] = point, point[2]
                    matches[i], matches[nleft+j] = nleft+j, i
        return replace(frame, features=combined, raw_features=combined, n_left=nleft,
                       right_camera=calibration.right_camera, T_right_left=calibration.T_right_left,
                       stereo_points=points, stereo_matches=matches, depth=depth,
                       right_u=np.full(n, -1.), map_points=[None]*n, outliers=np.zeros(n, bool))


def video_frames(path, start_time=0.):
    """Yield BGR frames with file presentation timestamps; no playback pacing."""
    capture = cv2.VideoCapture(str(Path(path)))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        index, previous = 0, -np.inf
        while True:
            ok, image = capture.read()
            if not ok:
                return
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if index and timestamp <= previous:
                if fps <= 0:
                    raise ValueError("Video has neither usable timestamps nor a frame rate")
                timestamp = index/fps
            yield FrameInput(start_time+timestamp, image)
            previous, index = timestamp, index+1
    finally:
        capture.release()
