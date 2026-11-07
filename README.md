# ORB-SLAM3 in readable Python

A source-based reimplementation of ORB-SLAM3 at revision `4452a3c4`, with
explicit tracking, mapping, visual/inertial factors, loop closing and map merging.
Read [COVERAGE.md](COVERAGE.md) before treating it as equivalent to the original.
Whole-system 99% algorithmic fidelity has **not** been established.

This directory runs independently with its own environment, unchanged trained ORB
vocabulary, source-derived reference fixtures and tests. It does
not import, link to, or discover the original C++ checkout.

Git excludes `.venv/` and the large local asset `assets/ORBvoc.txt`. For a fresh
checkout, supply the original ORB-SLAM3 vocabulary at that path before installing
or running the package and tests. Its SHA-256 must be
`f8dd027f7a6cb88129821341194d7f2c75b77b3394257ddd0d2229863d1a3570`.

## Run

The environment is at `.venv/` (a dedicated Conda environment containing Python;
it does not depend on the original project's environment).

```bash
.venv/bin/python -m pytest -q
```

To recreate it, run `bash tools/setup_environment.sh` with Conda available, or
create a Python environment there yourself and install `.[test,visualization]`.
Exact tested Python packages are listed in `requirements.lock.txt`.

## Supply your own data

```python
import numpy as np
from orbslam3 import SLAM, Sensor, Calibration, Pinhole, FrameInput

calibration = Calibration(
    camera=Pinhole(fx=500, fy=500, cx=320, cy=240),
    baseline_fx=40,            # baseline in metres * fx; also used by RGB-D factors
    depth_threshold=3.2,       # close-point threshold in metres
    color_order="RGB",
)
slam = SLAM(Sensor.RGBD, calibration)

# image: uint8 grayscale/RGB; depth: image-aligned camera-z depth in metres
result = slam.process(FrameInput(timestamp=0.0, image=image, depth=depth))
if result.pose is not None:
    point_in_camera = result.pose[:3, :3] @ point_in_world + result.pose[:3, 3]
trajectory = slam.trajectory()  # corrected poses, timestamps, map IDs, states
```

Use `MONOCULAR`, `STEREO`, `RGBD`, or the corresponding `*_IMU` sensor value.
Stereo accepts `right_image=`. Rectified stereo needs `baseline_fx`; unrectified
stereo additionally needs `right_camera`, `T_right_left` and overlap intervals
in calibration. `KannalaBrandt8(fx,fy,cx,cy,k1,k2,k3,k4)` supplies the fisheye
model. Pinhole radial/tangential coefficients go in `Calibration.distortion`.
Rectified pinhole stereo images must already be rectified and horizontally aligned.
The core does not guess calibration or image geometry.

For IMU modes, set `Calibration.imu` using
`ImuCalibration.from_noise_density(T_body_camera, gyro_noise, accel_noise,
gyro_walk, accel_walk, frequency)`. `T_body_camera` maps camera to IMU body.
Pass `imu=(ImuMeasurement(timestamp, acceleration, angular_velocity), ...)`
in each `FrameInput`; units are seconds, m/s² and rad/s. Include samples that
bracket consecutive image times. Adjacent packets may repeat their shared
endpoint sample. Camera/IMU timestamps must be synchronized.

`video_frames("your_video.mp4")` yields BGR image inputs; set
`color_order="BGR"` to use them. Pair or augment these inputs with your own
depth/stereo/IMU data. There are no published-dataset adapters.

## Read the algorithms

Start with `tracking.py`, `mapping.py`, `optimization.py`, `imu.py`,
`inertial_optimization.py`, `loop_closing.py` and `merging.py` in `orbslam3/`.
`input.py`, `system.py`, `visualization.py` and `export.py` are separate support
code. NumPy/SciPy solve visible numerical problems; no external SLAM framework
or original C++ binding supplies the algorithms.

`slam.set_localization_mode(True)` freezes mapping for visual sensors and enables
the original localization/temporary-point VO branch. The source itself does
not implement inertial localization-only mode.

Visualization is optional and creates no windows during tracking.
`draw_features`, `draw_matches`, `annotate_tracking`, `plot_map`, and
`plot_diagnostics` live in `orbslam3.visualization`; generic CSV writers live
in `orbslam3.export`.

GPL-3.0-or-later; retain [NOTICE](NOTICE), [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md) with redistribution.
