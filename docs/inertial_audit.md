# Inertial source audit

Baseline: `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4` of the local
ORB-SLAM3 checkout. Source, rather than a replacement inertial framework,
defines the algorithms. Component tests and rendered pinhole image sequences
verify initialization and tracking; whole-system 99% fidelity is not established.

| Component | Python coverage | Authoritative source |
| --- | --- | --- |
| IMU calibration and noise-density conversion | Implemented | `ImuTypes.cc:Calib::Set`, `Tracking.cc:613` |
| Discrete IMU preintegration, covariance, five bias Jacobians | Implemented | `ImuTypes.cc:Preintegrated::IntegrateNewMeasurement` |
| Bias correction, raw-measurement reintegration, preceding-interval merge | Implemented | `ImuTypes.cc` |
| Frame interval endpoint interpolation | Implemented as a pure function; queue selection is caller's responsibility | `Tracking.cc:PreintegrateIMU` |
| Body-state prediction | Implemented; selection of previous frame/keyframe remains tracking logic | `Tracking.cc:PredictStateIMU` |
| Metric and gravity/scale inertial edge residuals, analytic Jacobians | Implemented | `G2oTypes.cc:EdgeInertial`, `EdgeInertialGS` |
| Combined gyro/accelerometer random-walk residual | Implemented; factor information comes from the appropriate preintegration block | `G2oTypes.h:EdgeGyroRW`, `EdgeAccRW` |
| Hessian Schur marginalization and tracking prior generation | Implemented | `Optimizer.cc:Marginalize` |
| Fixed-pose gravity, scale, velocities and shared-bias optimization | Three graph cores implemented | Three `Optimizer::InertialOptimization` overloads |
| Full and local visual-inertial bundle adjustment | Graphs, robust objectives, variable selection, boundary weighting and outlier policies implemented | `Optimizer::FullInertialBA`, `LocalInertialBA` |
| Last-frame and last-keyframe visual-inertial tracking optimization | Four GN stages, visual rejection/recovery, state priors and marginalization implemented | `Optimizer::PoseInertialOptimizationLastFrame/LastKeyFrame` |
| Gravity/scale initialization, BA1/BA2 refinement and map transformation | Implemented, including motion/time gates and insufficient-motion reset request | `LocalMapping::InitializeIMU`, `ScaleRefinement` |
| Inertial map merge and four-DoF essential graph | Merging implemented in `merging.py`; four-DoF graph in `loop_geometry.py`; see their source audits | `LoopClosing.cc`, `Optimizer.cc` |
| End-to-end VI tracking, non-keyframe correction, Atlas reset and recovery | Integrated; basic initialization/post-initialization tracking verified in all three IMU modes; long loss/recovery remains unvalidated | `Tracking.cc`, `LocalMapping.cc`, `LoopClosing.cc` |

`imu.py` supplies mathematical factors. `inertial_optimization.py` constructs
source graphs and runs bundled-g2o-style LM or GN as appropriate.
`inertial_mapping.py` implements the local-mapping initialization schedule.
Component tests do not establish complete VI behavior on image sequences.

## Conventions and preserved decisions

`T_body_camera` maps camera coordinates into the body frame (`Tbc` in C++).
`rotation1/2` and positions describe body-to-world poses. Local body-pose
increments follow `ImuCamPose::Update`: `Rwb <- Rwb Exp(dr)` and
`twb <- twb + Rwb dt`, using the previous rotation for the position update.
Velocities live in world coordinates. Bias vectors are gyroscope followed by
accelerometer for optimization; `Bias` constructor arguments are named to avoid
confusion with the C++ accelerometer-first constructor.

The preintegrated covariance is ordered rotation, velocity, position, gyroscope
bias, accelerometer bias. The leading 9x9 covariance is propagated with the
source's `A C A^T + B Nga B^T`. Its bias-walk block adds discrete `NgaWalk`
each sample, without another `dt` factor. Noise-density conversion is
`noise * sqrt(frequency)` and `walk / sqrt(frequency)`.

Position, velocity and their bias Jacobians use the old rotation and the old
rotation-bias Jacobian. Rotation is updated last and projected using SVD
`U V^T`; the source does not apply an extra determinant correction. Integrated
rotation uses `I + hat(w dt)` and an identity right Jacobian when the rotation
increment norm is below `1e-4`, rather than replacing that branch with a
different midpoint integration rule.

The inertial edge information matrix is the inverse preintegration covariance,
symmetrized and eigendecomposed, with eigenvalues below `1e-12` set to zero.
Hessian marginalization uses an **absolute** singular-value cutoff of `1e-6`.
The implementation does not use the relative cutoff of a default pseudoinverse.

Frame endpoint interpolation intentionally follows all four `Tracking.cc`
branches. With only two selected samples, the original uses the first sample
for the entire frame interval. The IMU queue selection tolerance is the
source's hard-coded `mImuPer = 0.001` seconds, despite a comment suggesting
`1/frequency`.

## Source issues and numerical substitutions

- Arrays are NumPy float64; the original preintegration uses Eigen float32.
  NumPy SVD replaces Eigen JacobiSVD. No bitwise equivalence is claimed.
- Input shape, finite-value, rigid-transform and positive-duration checks raise
  exceptions instead of allowing undefined/nonfinite operations.
- Singular inertial covariance raises `LinAlgError` instead of silently using
  an unchecked inverse. No pseudo-inverse is substituted for a valid inverse.
- `EdgeInertialGS::linearizeOplus` gives a derivative with respect to additive
  scale `s`, while `VertexScale::oplusImpl` changes `s` multiplicatively by
  `exp(update)`. The analytic Jacobian port retains the source formula and
  documents that convention. The optimizer also retains this inherited
  inconsistency; multiplying the scale Jacobian by `s` would be a correction
  rather than a literal translation.
- `G2oTypes.cc::LogSO3` returns its skew-derived vector if `abs(sin(theta))`
  is below `1e-5`, including near pi. This behavior is retained for inertial
  factors. A separate general Lie implementation need not have this limitation.
- `Tracking::ResetFrameIMU()` is a TODO in this exact original revision. A
  port must distinguish preserving that original limitation from omitting
  other recovery paths that the source actually implements.
- Initial gravity alignment explicitly handles exactly parallel/antiparallel
  vectors; the source divides by a zero cross-product norm. The parallel case
  uses identity and the antiparallel case pi about x. Unobservable zero gravity
  increments raise an exception.

## Implemented optimization and scheduling policies

Initial inertial estimation requires ten temporally linked keyframes and at
least two seconds for monocular or one second for depth-supported modes.
Gravity is initially estimated by accumulating negative rotated preintegrated
delta velocities; visual velocities come from consecutive body-position
differences. The initial shared-bias/gravity/scale optimization fixes all poses,
optimizes velocities and two bias vertices, and optimizes two gravity rotation
DoFs; stereo scale is fixed. It runs 200 LM iterations, with zero-mean bias
priors and initial lambda `1e3` when the gyro prior is nonzero. A scale below
`0.1` rejects initialization before changing the map. Corrections update poses,
landmarks, velocities, and bias/preintegration state.

Initial priors are gyro `1e2` and accelerometer `1e10` for monocular or `1e5`
for other inertial modes. Local mapping reruns initialization after motion time
five seconds with priors `1` and `1e5`, then after fifteen seconds with zero
priors. These stages request full inertial BA. Motion time accumulates only
when the sum of two successive keyframe translations exceeds `0.05`. Before
BA2, insufficient motion (`dist < 0.02`, motion time below ten seconds) sets
the bad-IMU flag and requests active-map reset. Scale-refinement windows and
their outer `mTinit < 50` guard should be copied literally rather than assuming
all listed windows up to 75 seconds are reachable. The port preserves that guard.

Scale refinement fixes poses, velocities and biases, optimizes gravity and
scale with ten Gauss-Newton iterations and Huber delta `1`. The pose-inertial
tracking optimizers include visual factors, inertial links, bias random walks,
and (for last-frame tracking) a prior on the previous 15-dimensional state.
The next prior comes from a 30x30 Hessian with the previous 15 states
marginalized. `optimize_inertial_tracking` implements these graph procedures;
the caller selects last-frame versus last-keyframe and commits results.

Full VI BA has per-keyframe biases/random walks, or shared biases/priors during
initialization. Initial LM lambda is `1e-5`; inertial Huber delta is
`sqrt(16.92)` and visual deltas are `sqrt(5.991)`/`sqrt(7.815)`. Points seen only
by fixed poses are excluded. Like the source, no implicit first-pose constraint
is added. `fix_local` uses BA generation markers and returns early with fewer
than three free poses. Optional `inertial_ids` retains per-keyframe source
`bImu` readiness for mixed maps; visual-only states keep fixed velocity/bias
values and contribute no IMU links. Map filtering belongs to mapping.

Local VI BA uses ten temporal states and ten LM iterations, or 25/four in
the large-window branch. Initial lambda is `1` or `.01`. The fixed-boundary
inertial link has information scaled by `.01` and a Huber kernel; all links
are robust during reinitialization. Mono rejection uses `5.991`, relaxed by
`1.5` for tracked depths below ten metres, and checks positive depth. Stereo
uses `7.815`. Non-large BA rejects nonfinite or more-than-doubled cost.

Tracking retains its state through four ten-iteration GN stages. Last-keyframe
mono thresholds are `[12, 7.5, 5.991, 5.991]`; last-frame mono thresholds are
all `5.991`; stereo thresholds are `[15.6, 9.8, 7.815, 7.815]`. Visual kernels
are removed after stage two. Low-inlier recovery uses thresholds 18/24 below
thirty inliers, except during reinitialization. Last-frame state priors use
Huber delta five. Next priors use the unrobustified retained-factor Hessian.

Normal equations are assembled from visible analytic Jacobians and block
information matrices. Huber weights apply to whole residual blocks, not each
coordinate. Independent 3D landmarks are Schur-marginalized after LM damping.
SciPy sparse LU replaces Eigen's sparse solver. Source state retraction and
bundled-g2o LM acceptance remain explicit.

## Verification

`tests/test_imu.py` covers analytic constant-acceleration integration, rotation
and all bias Jacobians against numerical differentiation or reintegration,
covariance symmetry/positive semidefiniteness, calibration conversion,
reintegration/merging, endpoint interpolation, zero inertial residual after
prediction, every vertex of metric and gravity/scale edge Jacobians, and
Hessian Schur marginalization including the absolute singular-value threshold.

The default differential test reads `tests/fixtures/imu_reference.npz`,
generated by compiling the **unmodified original** `src/ImuTypes.cc` and
integrating sixty synthetic intervals. The fixture retains inputs and outputs;
`imu_reference.json` records SHA-256, source/header hashes, commit, compiler,
flags and generator hash. Normal tests require no original checkout or native
compiler. No C++ library is used by the Python algorithm. Measured absolute
differences on this environment were: rotation `4.34e-7`, velocity `3.55e-7`,
position `1.82e-8`, bias Jacobians at most `1.73e-7`, covariance `2.75e-9`,
average acceleration `1.91e-6`, and average angular velocity `2.22e-8`. The
largest discrepancy was investigated as accumulated single-precision SVD
normalization differences; the average-acceleration test has a separate
`3e-6` absolute tolerance while covariance is checked at `1e-8`. Source formula
or test-input changes were not used to reduce these differences.

Set `ORBSLAM3_REFERENCE` explicitly to request optional oracle recompilation
against that checkout. It never auto-discovers a sibling checkout and skips by
default. The checked-in oracle and all Python tests still run standalone.

`test_inertial_optimization.py` adds eleven synthetic tests for body/point
projection Jacobians under six pinhole/fisheye observation combinations, state
prior Jacobians, sparse Schur versus direct normal solves, gravity/scale/bias/
velocity recovery, fixed variables, all three initialization/refinement graphs,
full/shared-bias/local VI BA, temporal window selection, and both tracking-prior
paths. `test_inertial_mapping.py` adds four tests for real keyframe initialization
through full VI BA, motion/reset/refinement scheduling, and nonzero camera-IMU
lever-arm correction. Body states are recomputed from transformed camera
centers, so the rigid lever arm is not scaled with world coordinates.

## Image pipeline verification

`tests/test_system_inertial.py` processes fifteen physically rendered pinhole
frames in each of monocular+IMU, stereo+IMU and RGB-D+IMU, with the retained
vocabulary and default loop closer. A 200 Hz analytic IMU trajectory drives
actual ten-keyframe initialization, full VI BA and subsequent state-prior
tracking. With first-camera alignment and no scale fitting, metric position
error stays below 8 cm over approximately one metre, and rotation below
0.03 radians. All keyframe camera/body poses remain consistent. This caught
and repaired stale body-pose caches following visual local BA. Long sequences,
image-triggered inertial loops/merges and loss/recovery still need validation.

The rendered VI tests enable the original `fast_imu_init=True` option,
which bypasses the initial stereo/RGB-D acceleration-difference gate.
They do not establish initialization under that default excitation gate;
the normal ten-keyframe/time requirements and graph formulations remain active.
