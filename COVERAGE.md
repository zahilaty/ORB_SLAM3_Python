# Algorithm coverage and deviations

Baseline: exact local revision `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`.
`BASELINE.json` records source hashes and local modifications. The source is
authoritative; this port does not substitute a generic SLAM pipeline.

**Status: principal subsystems implemented; whole-system fidelity remains under
validation.** Passing component tests or short generated sequences does not
establish 99% equivalence, long-sequence robustness or original thread timing.

| Component | Implementation and evidence |
| --- | --- |
| ORB extraction | Pyramid, cell FAST/fallback, octree distribution, circular IC orientation, exact rotated BRIEF pattern, blur and fisheye overlap ordering. C++ feature/descriptor goldens. |
| Feature association | Initialization, BoW, local/motion/relocalization/Sim3 projection, triangulation, mutual Sim3 and fusion variants; original thresholds, scale/view gates and orientation histogram. Decision tests. |
| Cameras and Lie geometry | Pinhole and KB8 projection/unprojection/Jacobians, SE3/Sim3, epipolar tests, DLT triangulation; numerical derivatives, original C++ goldens and optical-axis/pi limits. |
| Monocular initialization | Shared eight-point RANSAC H/F, anisotropic normalization, symmetric scores, source 0.50 model gate, essential/four and homography/eight hypotheses, ambiguity/cheirality/parallax gates, initial BA/scale normalization. C++ decision goldens and rendered-image tracking. |
| Stereo/RGB-D preparation and initialization | Source row/descriptor/L1/subpixel stereo and median rejection; fisheye overlap/triangulation; raw-pixel depth lookup and synthetic stereo coordinate; strict 500-feature gate. Input tests and image sequences. |
| Tracking | Motion/reference/local-map paths, pose optimization, local voting, visibility/statistics, source keyframe criteria, recently-lost/lost recovery, Atlas reset and MLPnP relocalization. Decision tests and generated images. |
| Visual localization-only | Mapping freeze, temporary depth landmarks, VO/map-match decisions, motion snapshot/relocalization priority, freeze/resume. Unit and real-image integration tests. |
| Map/Atlas | Observation accounting including rigs, representative descriptors, normal/range prediction, covisibility, spanning tree and reparenting, protected loop/merge edges, multiple maps, corrected trajectory references. Mutation and geometry tests. |
| Local mapping | Triangulation/depth fallback, parallax/baseline/reprojection/scale gates, recent-point culling, neighbor fusion, local visual/VI graph selection and commit, redundant-KF/temporal culling. Real geometry/optimizer tests. |
| Visual optimization | Analytic mono/stereo/rig factors, block Huber, g2o-style LM acceptance, landmark Schur elimination, pose rejection stages, global/local/welding BA. Original C++ optimizer goldens and noisy geometric recovery. |
| Place recognition | Retained ORBvoc tree, TF-IDF/L1 normalization/scoring, level-up feature groups, inverted index, independent loop/merge ranking and covisibility accumulation. Original C++ vocabulary goldens. |
| Loop closing | Temporal/common-region hypotheses, Horn Sim3 RANSAC, projection and bilateral refinement, covisible verification, duplicate fusion, correction, Sim3/4DoF graphs, global BA and descendant propagation. Actual synthetic-map loop pipeline. |
| Map merging | Visual and inertial welding windows, transfer/ID/tree changes, fusion, welding BA, noncritical propagation and VI readiness. Direct merging tests and full cross-map recognition/welding pipeline. |
| IMU mathematics | Source noise conversion, discrete preintegration, covariance/bias Jacobians, correction/reintegration/merge, endpoint interpolation and body prediction; inertial/prior/random-walk factors and marginalization. C++ preintegration goldens and numerical Jacobians. |
| VI optimization and initialization | Gravity/two-DoF rotation, positive scale, bias/velocity initialization/refinements, full/local/tracking graphs, priors and outlier schedules; motion/time gates, BA1/BA2, map/frame correction and low-motion reset. Graph tests and three rendered-image VI modes through actual initialization. |

All six sensor modes have frame-by-frame input and tracking implementations.
Generated pinhole image tests cover monocular, stereo, RGB-D and all three IMU
combinations. These VI image tests enable the source option `fast_imu_init=True`
(bypassing the initial stereo/RGB-D acceleration-difference gate); they still
execute the normal ten-keyframe/time gates and inertial optimizations.
Five unrectified KB8 stereo image pairs, including right-camera
yaw and nonzero distortion, verify triangulation, initialization and tracking
with metric position error below 2.5 cm. Their complete inertial and
image-to-loop/merge/recovery trajectories remain unvalidated.
Real BoW/MLPnP/projection relocalization, temporary/persistent loss, Atlas
thresholds, all inertial timestamp-jump cases and five-second IMU-only prediction
have recovery tests. Relocalization/loop/merging components have geometric and decision tests;
long image sequences that trigger each recovery/merge combination remain work.

## Deliberate differences

- The sequential order is tracking, queued keyframe mapping, then loop closing.
  Map changes become visible at image boundaries; global BA commits synchronously.
  Original timing, BA interruption/cancellation and opportunistic queue behavior
  are not reproduced. These differences can affect later keyframes/trajectories.
- NumPy float64/SVD/eigh and SciPy sparse solves/matrix exponentials replace
  Eigen/Sophus/g2o numerical operations. Residuals, information, robust block
  losses, retractions, fixed/free variables and staged rejection remain explicit.
  Reduced visual camera/pose-graph systems are dense and can be slow on large maps.
- NumPy sampling replaces DUtils randomness; stable IDs replace C++ pointer/tie
  ordering. On the extractor fixture, 1001/1004 locations coincide; descriptors
  at all shared locations match byte-for-byte. Random-model outputs may differ.
- Input validation and stable optical-axis/half-turn Lie limits avoid undefined
  source behavior. Tiny pyramid interiors and empty stereo match sets are safe.
- Explicit source-defect repairs: `ReconstructH` returns its computed 3D points;
  vocabulary parsing excludes an uninitialized EOF phantom node; combined right
  feature indices avoid a keyframe-culling out-of-bounds access; duplicate point
  observations are idempotent; pose-only VI merge covisibles retain velocity/bias
  instead of dereferencing nonexistent optimizer vertices. Stale hypotheses on
  discarded maps are released. Frames with no estimated pose do not append the
source exporter’s duplicate previous timestamp/pose record. Repairs are identified
where they occur.

Many unusual source decisions are intentionally retained, including the
orientation histogram's `1/30` factor, triangulation's many-to-one matching,
source-specific stereo thresholds, the monocular initial-extractor gate's mixed
frame/keyframe ID subtraction, MLPnP's OR trial loop/discarded refinement,
planar MLPnP scale limitation, inertial scale-Jacobian inconsistency, and the
exact revision's empty `ResetFrameIMU` TODO. Detailed mappings and numerical
evidence are in `docs/inertial_audit.md`, `docs/optimization_audit.md` and
`docs/merging_audit.md`.

## Remaining verification

- The root `main.py` example processes `~/Data/DJI_0690.MP4`, expanding `~` to
  the current user's home directory, in monocular mode. It resizes to at most
  640 pixels on the longer side and scales
  a centered pinhole guess (`fx = fy = max(width, height)`, zero distortion).
  A 12-frame smoke run completed resizing, processing and trajectory extraction;
  no poses were initialized within that prefix.
  Resolution alone does not determine calibration; this clip's calibration,
  complete trajectory and accuracy remain unvalidated. Monocular scale is arbitrary.
- Broader identical-input C++ comparisons, especially image-based VI tracking,
  fisheye rigs, long loss/relocalization sequences and inertial map merges.
- Long-sequence accuracy, initialization observability in difficult motion, and
  practical memory/time limits for large maps.
- Continued source-branch review; the source hash ledger is provenance, not proof
  that every branch has been exercised. No numerical percentage is claimed.

Normal tests use checked-in C++ output fixtures and the local `assets/ORBvoc.txt`
vocabulary, which is excluded from Git and must be supplied separately in a fresh
checkout. With that asset and the environment installed, they need no original
checkout, compiler, network, ROS, datasets, display or Pangolin. Optional live
oracles run only when `ORBSLAM3_REFERENCE` is explicitly supplied.

An identical-input five-frame RGB-D `System` comparison uses the original C++
tracker and retained vocabulary: feature counts agree exactly and camera-center
differences stay below 0.13 mm. An 806-landmark local-BA graph captured from
the image pipeline agrees with original g2o poses within 1.7e-8. Both optimizers
produce the same approximately 25 mm deformation on that noisy short-baseline
fixture; it was investigated rather than hidden by changing the objective.
