"""ORB-SLAM3's pyramid FAST/octree/IC-angle/rotated-BRIEF extraction.

Algorithm translated from src/ORBextractor.cc, GPL-3.0-or-later. Original
copyright 2014-2021 Raúl Mur-Artal, Carlos Campos and University of Zaragoza
contributors. The retained ORB pattern derives from Willow Garage/OpenCV
(BSD-3-Clause); see THIRD_PARTY_NOTICES.md.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np


# Exact 512 sample points, in the source's comparison and byte/bit order.
_PATTERN = np.array([
    8, -3, 9, 5, 4, 2, 7, -12, -11, 9, -8, 2, 7, -12, 12, -13,
    2, -13, 2, 12, 1, -7, 1, 6, -2, -10, -2, -4, -13, -13, -11, -8,
    -13, -3, -12, -9, 10, 4, 11, 9, -13, -8, -8, -9, -11, 7, -9, 12,
    7, 7, 12, 6, -4, -5, -3, 0, -13, 2, -12, -3, -9, 0, -7, 5,
    12, -6, 12, -1, -3, 6, -2, 12, -6, -13, -4, -8, 11, -13, 12, -8,
    4, 7, 5, 1, 5, -3, 10, -3, 3, -7, 6, 12, -8, -7, -6, -2,
    -2, 11, -1, -10, -13, 12, -8, 10, -7, 3, -5, -3, -4, 2, -3, 7,
    -10, -12, -6, 11, 5, -12, 6, -7, 5, -6, 7, -1, 1, 0, 4, -5,
    9, 11, 11, -13, 4, 7, 4, 12, 2, -1, 4, 4, -4, -12, -2, 7,
    -8, -5, -7, -10, 4, 11, 9, 12, 0, -8, 1, -13, -13, -2, -8, 2,
    -3, -2, -2, 3, -6, 9, -4, -9, 8, 12, 10, 7, 0, 9, 1, 3,
    7, -5, 11, -10, -13, -6, -11, 0, 10, 7, 12, 1, -6, -3, -6, 12,
    10, -9, 12, -4, -13, 8, -8, -12, -13, 0, -8, -4, 3, 3, 7, 8,
    5, 7, 10, -7, -1, 7, 1, -12, 3, -10, 5, 6, 2, -4, 3, -10,
    -13, 0, -13, 5, -13, -7, -12, 12, -13, 3, -11, 8, -7, 12, -4, 7,
    6, -10, 12, 8, -9, -1, -7, -6, -2, -5, 0, 12, -12, 5, -7, 5,
    3, -10, 8, -13, -7, -7, -4, 5, -3, -2, -1, -7, 2, 9, 5, -11,
    -11, -13, -5, -13, -1, 6, 0, -1, 5, -3, 5, 2, -4, -13, -4, 12,
    -9, -6, -9, 6, -12, -10, -8, -4, 10, 2, 12, -3, 7, 12, 12, 12,
    -7, -13, -6, 5, -4, 9, -3, 4, 7, -1, 12, 2, -7, 6, -5, 1,
    -13, 11, -12, 5, -3, 7, -2, -6, 7, -8, 12, -7, -13, -7, -11, -12,
    1, -3, 12, 12, 2, -6, 3, 0, -4, 3, -2, -13, -1, -13, 1, 9,
    7, 1, 8, -6, 1, -1, 3, 12, 9, 1, 12, 6, -1, -9, -1, 3,
    -13, -13, -10, 5, 7, 7, 10, 12, 12, -5, 12, 9, 6, 3, 7, 11,
    5, -13, 6, 10, 2, -12, 2, 3, 3, 8, 4, -6, 2, 6, 12, -13,
    9, -12, 10, 3, -8, 4, -7, 9, -11, 12, -4, -6, 1, 12, 2, -8,
    6, -9, 7, -4, 2, 3, 3, -2, 6, 3, 11, 0, 3, -3, 8, -8,
    7, 8, 9, 3, -11, -5, -6, -4, -10, 11, -5, 10, -5, -8, -3, 12,
    -10, 5, -9, 0, 8, -1, 12, -6, 4, -6, 6, -11, -10, 12, -8, 7,
    4, -2, 6, 7, -2, 0, -2, 12, -5, -8, -5, 2, 7, -6, 10, 12,
    -9, -13, -8, -8, -5, -13, -5, -2, 8, -8, 9, -13, -9, -11, -9, 0,
    1, -8, 1, -2, 7, -4, 9, 1, -2, 1, -1, -4, 11, -6, 12, -11,
    -12, -9, -6, 4, 3, 7, 7, 12, 5, 5, 10, 8, 0, -4, 2, 8,
    -9, 12, -5, -13, 0, 7, 2, 12, -1, 2, 1, 7, 5, 11, 7, -9,
    3, 5, 6, -8, -13, -4, -8, 9, -5, 9, -3, -3, -4, -7, -3, -12,
    6, 5, 8, 0, -7, 6, -6, 12, -13, 6, -5, -2, 1, -10, 3, 10,
    4, 1, 8, -4, -2, -2, 2, -13, 2, -12, 12, 12, -2, -13, 0, -6,
    4, 1, 9, 3, -6, -10, -3, -5, -3, -13, -1, 1, 7, 5, 12, -11,
    4, -2, 5, -7, -13, 9, -9, -5, 7, 1, 8, 6, 7, -8, 7, 6,
    -7, -4, -7, 1, -8, 11, -7, -8, -13, 6, -12, -8, 2, 4, 3, 9,
    10, -5, 12, 3, -6, -5, -6, 7, 8, -3, 9, -8, 2, -12, 2, 8,
    -11, -2, -10, 3, -12, -13, -7, -9, -11, 0, -10, -5, 5, -3, 11, 8,
    -2, -13, -1, 12, -1, -8, 0, 9, -13, -11, -12, -5, -10, -2, -10, 11,
    -3, 9, -2, -13, 2, -3, 3, 2, -9, -13, -4, 0, -4, 6, -3, -10,
    -4, 12, -2, -7, -6, -11, -4, 9, 6, -3, 6, 11, -13, 11, -5, 5,
    11, 11, 12, 6, 7, -5, 12, -2, -1, 12, 0, 7, -4, -8, -3, -2,
    -7, 1, -6, 7, -13, -12, -8, -13, -7, -2, -6, -8, -8, 5, -6, -9,
    -5, -1, -4, 5, -13, 7, -8, 10, 1, 5, 5, -13, 1, 0, 10, -13,
    9, 12, 10, -1, 5, -8, 10, -9, -1, 11, 1, -13, -9, -3, -6, 2,
    -1, -10, 1, 12, -13, 1, -8, -10, 8, -11, 10, -6, 2, -13, 3, -6,
    7, -13, 12, -9, -10, -10, -5, -7, -10, -8, -8, -13, 4, -6, 8, 5,
    3, 12, 8, -13, -4, 2, -3, -3, 5, -13, 10, -12, 4, -13, 5, -1,
    -9, 9, -4, 3, 0, 3, 3, -9, -12, 1, -6, 1, 3, 2, 4, -8,
    -10, -10, -10, 9, 8, -13, 12, 12, -8, -12, -6, -5, 2, 2, 3, 7,
    10, 6, 11, -8, 6, 8, 8, -12, -7, 10, -6, 5, -3, -9, -3, 9,
    -1, -13, -1, 5, -3, -7, -3, 4, -8, -2, -8, 3, 4, 2, 12, 12,
    2, -5, 3, 11, 6, -9, 11, -13, 3, -1, 7, 12, 11, -1, 12, 4,
    -3, 0, -3, 6, 4, -11, 4, 12, 2, -4, 2, 1, -10, -6, -8, 1,
    -13, 7, -11, 1, -13, 12, -11, -13, 6, 0, 11, -13, 0, -1, 1, 4,
    -13, 3, -9, -2, -9, 8, -6, -3, -13, -6, -8, -2, 5, -9, 8, 10,
    2, 7, 3, -9, -1, -6, -1, -1, 9, 5, 11, -2, 11, -3, 12, -8,
    3, 0, 3, 5, -1, 4, 0, 10, 3, -6, 4, 5, -13, 0, -10, 5,
    5, 8, 12, 11, 8, 9, 9, -6, 7, -4, 8, -12, -10, 4, -10, 9,
    7, 3, 12, 4, 9, -7, 10, -2, 7, 0, 12, -2, -1, -6, 0, -11,
], dtype=np.int32).reshape(512, 2)
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def hamming(a, b):
    """256-bit Hamming distance, broadcasting over leading dimensions."""
    return _POPCOUNT[np.bitwise_xor(a, b)].sum(axis=-1, dtype=np.int32)


@dataclass
class FeatureSet:
    points: np.ndarray
    descriptors: np.ndarray
    octaves: np.ndarray
    angles: np.ndarray
    responses: np.ndarray
    sizes: np.ndarray
    mono_count: int

    def __len__(self):
        return len(self.points)


@dataclass(eq=False)
class _Node:
    x0: int
    y0: int
    x1: int
    y1: int
    indices: np.ndarray

    def divide(self, points):
        xm = self.x0 + math.ceil((self.x1-self.x0)/2)
        ym = self.y0 + math.ceil((self.y1-self.y0)/2)
        xy = points[self.indices]
        right, bottom = xy[:, 0] >= xm, xy[:, 1] >= ym
        children = []
        for mask, bounds in (
            (~right & ~bottom, (self.x0, self.y0, xm, ym)),
            (right & ~bottom, (xm, self.y0, self.x1, ym)),
            (~right & bottom, (self.x0, ym, xm, self.y1)),
            (right & bottom, (xm, ym, self.x1, self.y1)),
        ):
            if np.any(mask):
                children.append(_Node(*bounds, self.indices[mask]))
        return children


def distribute_octree(points, responses, width, height, count):
    """Retain the strongest FAST response per spatial leaf (not a top-N sort).

    Like the C++ routine, splitting a leaf can exceed the requested count by
    up to three. Equal population/x-position ties use Python's stable sort;
    the source's std::sort leaves those ties unspecified.
    """
    if not len(points) or count <= 0:
        return np.empty(0, dtype=np.int32)
    n_initial = max(1, int(math.floor(width/height + 0.5)))
    strip_width = np.float32(width/n_initial)
    strips = np.minimum((points[:, 0]/strip_width).astype(int), n_initial-1)
    nodes = []
    for strip in range(n_initial):
        ids = np.flatnonzero(strips == strip)
        if len(ids):
            nodes.append(_Node(int(strip_width*strip), 0,
                               int(strip_width*(strip+1)), height, ids))
    while True:
        previous_count = len(nodes)
        expandable = []
        # C++ push_front inserts children before the not-yet-visited parents.
        for node in nodes.copy():
            if len(node.indices) == 1:
                continue
            children = node.divide(points)
            nodes.remove(node)
            nodes[0:0] = children[::-1]
            expandable.extend(child for child in children if len(child.indices)>1)
        if len(nodes) >= count or len(nodes) == previous_count:
            break
        if len(nodes) + len(expandable)*3 > count:
            while True:
                previous_count = len(nodes)
                previous_expandable = sorted(expandable, key=lambda n: (len(n.indices), n.x0))
                expandable = []
                for node in reversed(previous_expandable):
                    children = node.divide(points)
                    nodes.remove(node)
                    nodes[0:0] = children[::-1]
                    expandable.extend(child for child in children if len(child.indices)>1)
                    if len(nodes) >= count:
                        break
                if len(nodes) >= count or len(nodes) == previous_count:
                    break
            break
    return np.asarray([n.indices[np.argmax(responses[n.indices])] for n in nodes], dtype=np.int32)


class ORBExtractor:
    def __init__(self, nfeatures=1000, scale_factor=1.2, nlevels=8,
                 initial_fast=20, minimum_fast=7):
        if nfeatures < 0 or nlevels < 1 or scale_factor <= 1:
            raise ValueError("nfeatures >= 0, nlevels >= 1 and scale_factor > 1 required")
        if not 0 <= minimum_fast <= initial_fast <= 255:
            raise ValueError("FAST thresholds must satisfy 0 <= minimum <= initial <= 255")
        self.nfeatures, self.nlevels = int(nfeatures), int(nlevels)
        self.scale_factor = np.float32(scale_factor)
        self.scales = np.ones(nlevels, dtype=np.float32)
        for level in range(1, nlevels):
            self.scales[level] = self.scales[level-1]*self.scale_factor
        self.inverse_scales = 1/self.scales
        self.variances = self.scales*self.scales
        self.inverse_variances = 1/self.variances
        factor = np.float32(1/self.scale_factor)
        desired = np.float32(nfeatures*(1-factor)/(1-float(factor)**nlevels))
        self.features_per_level = []
        for _ in range(nlevels-1):
            self.features_per_level.append(int(np.rint(desired)))
            desired *= factor
        self.features_per_level.append(max(nfeatures-sum(self.features_per_level), 0))
        self._fast = cv2.FastFeatureDetector_create(int(initial_fast), True)
        self._fast_low = cv2.FastFeatureDetector_create(int(minimum_fast), True)
        self.pyramid = []
        umax = np.zeros(16, dtype=int)
        vmax, vmin = int(np.floor(15*np.sqrt(2)/2+1)), int(np.ceil(15*np.sqrt(2)/2))
        for v in range(vmax+1):
            umax[v] = int(np.rint(np.sqrt(225-v*v)))
        v0 = 0
        for v in range(15, vmin-1, -1):
            while umax[v0] == umax[v0+1]:
                v0 += 1
            umax[v] = v0
            v0 += 1
        # The mask includes the source's symmetric discrete circular support.
        offsets = [(u, v) for v in range(-15,16) for u in range(-umax[abs(v)],umax[abs(v)]+1)]
        self._orientation_offsets = np.asarray(offsets, dtype=np.int32)

    def compute_pyramid(self, image):
        self.pyramid = [image.copy()]
        for level in range(1, self.nlevels):
            size = tuple(np.rint(np.array(image.shape[::-1])*self.inverse_scales[level]).astype(int))
            if min(size) < 1:
                raise ValueError("image too small for requested pyramid levels")
            self.pyramid.append(cv2.resize(self.pyramid[-1], size, interpolation=cv2.INTER_LINEAR))
        return self.pyramid

    def orientation(self, image, points):
        if not len(points):
            return np.empty(0, dtype=np.float32)
        centers = np.rint(points).astype(int)
        offset = self._orientation_offsets
        pixels = image[centers[:, None, 1]+offset[None, :, 1],
                       centers[:, None, 0]+offset[None, :, 0]].astype(np.int32)
        moments = pixels @ offset
        return np.array([cv2.fastAtan2(float(y), float(x)) for x,y in moments], dtype=np.float32)

    @staticmethod
    def descriptors(image, points, angles):
        if not len(points):
            return np.empty((0,32), dtype=np.uint8)
        radians = angles*np.float32(np.pi/180)
        cosine, sine = np.cos(radians), np.sin(radians)
        x, y = _PATTERN.astype(np.float32).T
        dx = np.rint(cosine[:, None]*x - sine[:, None]*y).astype(int)
        dy = np.rint(sine[:, None]*x + cosine[:, None]*y).astype(int)
        centers = np.rint(points).astype(int)
        samples = image[centers[:,None,1]+dy, centers[:,None,0]+dx]
        bits = samples[:,0::2] < samples[:,1::2]
        return np.packbits(bits, axis=1, bitorder="little")

    def extract(self, image, overlap=None):
        """Extract from uint8 grayscale; optional inclusive stereo overlap [x0,x1].

        Images are deliberately not color-converted or undistorted in this
        core routine. The caller supplies the calibrated image domain.
        """
        image = np.asarray(image)
        if image.ndim != 2 or image.dtype != np.uint8 or not image.size:
            raise ValueError("nonempty uint8 grayscale image required")
        self.compute_pyramid(image)
        all_points, all_desc, all_levels = [], [], []
        all_angles, all_responses, all_sizes = [], [], []
        for level, layer in enumerate(self.pyramid):
            height, width = layer.shape
            border = 16
            inner_width, inner_height = width-2*border, height-2*border
            # The source divides by zero below 35 interior pixels. Return no
            # features at such levels instead of accessing an invalid grid.
            if min(inner_width, inner_height) < 35:
                continue
            columns, rows = inner_width//35, inner_height//35
            cell_width, cell_height = math.ceil(inner_width/columns), math.ceil(inner_height/rows)
            points, responses = [], []
            for row in range(rows):
                y0 = border+row*cell_height
                y1 = min(y0+cell_height+6, height-border)
                if y0 >= height-border-3:
                    continue
                for col in range(columns):
                    x0 = border+col*cell_width
                    x1 = min(x0+cell_width+6, width-border)
                    if x0 >= width-border-6:
                        continue
                    cell = layer[y0:y1,x0:x1]
                    keypoints = self._fast.detect(cell)
                    if not keypoints:
                        keypoints = self._fast_low.detect(cell)
                    for keypoint in keypoints:
                        points.append((keypoint.pt[0]+col*cell_width,
                                       keypoint.pt[1]+row*cell_height))
                        responses.append(keypoint.response)
            points = np.asarray(points, dtype=np.float32).reshape(-1,2)
            responses = np.asarray(responses, dtype=np.float32)
            selected = distribute_octree(points, responses, inner_width, inner_height,
                                         self.features_per_level[level])
            points, responses = points[selected]+border, responses[selected]
            angles = self.orientation(layer, points)
            blurred = cv2.GaussianBlur(layer, (7,7), 2, 2, borderType=cv2.BORDER_REFLECT_101)
            descriptors = self.descriptors(blurred, points, angles)
            all_points.append(points*self.scales[level])
            all_desc.append(descriptors)
            all_levels.append(np.full(len(points), level, dtype=np.int32))
            all_angles.append(angles)
            all_responses.append(responses)
            all_sizes.append(np.full(len(points), int(31*self.scales[level]), dtype=np.float32))
        if not all_points:
            return FeatureSet(np.empty((0,2), np.float32), np.empty((0,32), np.uint8),
                              np.empty(0,np.int32), np.empty(0,np.float32),
                              np.empty(0,np.float32), np.empty(0,np.float32), 0)
        points, descriptors = np.concatenate(all_points), np.concatenate(all_desc)
        octaves, angles = np.concatenate(all_levels), np.concatenate(all_angles)
        responses, sizes = np.concatenate(all_responses), np.concatenate(all_sizes)
        order = np.arange(len(points))
        mono_count = len(points)
        if overlap is not None:
            inside = (points[:,0] >= overlap[0]) & (points[:,0] <= overlap[1])
            order = np.r_[order[~inside],order[inside][::-1]]
            mono_count = int((~inside).sum())
        return FeatureSet(points[order], descriptors[order], octaves[order],
                          angles[order], responses[order], sizes[order], mono_count)
