"""Optional feature and map inspection, independent of SLAM execution."""

import cv2
import numpy as np
from .geometry import inverse_pose


def draw_features(image, frame):
    image = np.asarray(image)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[..., :3].copy()
    points = frame.raw_features.points
    for i, pixel in enumerate(points):
        mapped = frame.map_points[i] is not None
        color = (50, 210, 50) if mapped and not frame.outliers[i] else (50, 50, 230) if frame.outliers[i] else (210, 150, 40)
        cv2.circle(canvas, tuple(np.rint(pixel).astype(int)), 2, color, 1, cv2.LINE_AA)
    return canvas


def draw_matches(image1, frame1, image2, frame2, matches):
    def bgr(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[..., :3].copy()
    left, right = bgr(np.asarray(image1)), bgr(np.asarray(image2))
    canvas = np.zeros((max(len(left), len(right)), left.shape[1]+right.shape[1], 3), np.uint8)
    canvas[:len(left), :left.shape[1]], canvas[:len(right), left.shape[1]:] = left, right
    for i, j in enumerate(matches):
        if j < 0:
            continue
        a = np.rint(frame1.raw_features.points[i]).astype(int)
        b = np.rint(frame2.raw_features.points[j]).astype(int)+[left.shape[1], 0]
        cv2.line(canvas, tuple(a), tuple(b), (50, 200, 50), 1, cv2.LINE_AA)
    return canvas


def annotate_tracking(image, result):
    canvas = np.asarray(image).copy()
    text = f"{result.state.name} | {result.inliers} inliers | map {result.map_id} | keyframe {result.keyframe_id}"
    cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def plot_map(slam, axes=None, maximum_points=10000):
    """Return a matplotlib figure/axes; importing SLAM never imports matplotlib."""
    import matplotlib.pyplot as plt
    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(projection="3d")
    for map_ in slam.atlas.maps.values():
        points = np.array([point.position for point in map_.points.values() if not point.bad])
        if len(points):
            stride = max(1, int(np.ceil(len(points)/maximum_points)))
            axes.scatter(*points[::stride].T, s=1, alpha=.4)
        keyframes = np.array([keyframe.center for keyframe in map_.keyframes.values() if not keyframe.bad])
        if len(keyframes):
            axes.scatter(*keyframes.T, marker="^", s=18, label=f"map {map_.id} keyframes")
        trajectory = np.array([inverse_pose(pose)[:3, 3] for _, pose, map_id, _ in slam.trajectory() if map_id == map_.id])
        if len(trajectory):
            axes.plot(*trajectory.T, label=f"map {map_.id} trajectory")
    axes.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    return axes.figure, axes


def plot_diagnostics(results, axes=None):
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots()
    axes.plot([r.timestamp for r in results], [r.inliers for r in results], label="tracking inliers")
    axes.set(xlabel="time (s)", ylabel="inliers")
    axes.legend()
    return axes.figure, axes
