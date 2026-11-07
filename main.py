"""Minimal monocular video example with approximate camera intrinsics."""

from pathlib import Path

import cv2

from orbslam3 import SLAM, Calibration, FrameInput, Pinhole, Sensor


def main():
    video = cv2.VideoCapture(str(Path("~/Data/DJI_0690.MP4").expanduser()))
    if not video.isOpened():
        raise RuntimeError("Cannot open video")

    try:
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = video.get(cv2.CAP_PROP_FPS)
        scale = min(1.0, 640 / max(width, height))
        size = (round(width * scale), round(height * scale))
        sx, sy = size[0] / width, size[1] / height

        focal = max(width, height)  # Rough guess; resolution cannot determine focal length.
        calibration = Calibration(
            camera=Pinhole(fx=focal * sx, fy=focal * sy,
                           cx=width / 2 * sx, cy=height / 2 * sy),
            frames_per_second=fps,
            color_order="BGR",
        )
        slam = SLAM(Sensor.MONOCULAR, calibration)

        index = 0
        while True:
            ok, image = video.read()
            if not ok:
                break
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            result = slam.process(FrameInput(timestamp=index / fps, image=image))
            index += 1

        trajectory = slam.trajectory()
        print(f"Processed {index} frames; trajectory has {len(trajectory)} poses.")
        return trajectory
    finally:
        video.release()


if __name__ == "__main__":
    trajectory = main()
