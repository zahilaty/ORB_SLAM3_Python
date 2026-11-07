# Optimization source audit

Baseline: local ORB-SLAM3 commit `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`.
Authoritative files: `src/Optimizer.cc`, `src/OptimizableTypes.cpp`,
`include/OptimizableTypes.h`, `src/G2oTypes.cc`, `include/G2oTypes.h`, and bundled
g2o `core/optimization_algorithm_levenberg.cpp`, `core/base_edge.h`,
`types/types_six_dof_expmap.*`, `types/se3quat.h`, `types/sim3.*`.

## Implemented numerical operations

| Source procedure | Python operation | Preserved behavior |
| --- | --- | --- |
| `EdgeSE3ProjectXYZ*` | `projection_factor` | Observation minus camera projection; world-to-camera T_cw; left increments [rotation, translation]; analytic pose and landmark Jacobians. |
| `EdgeStereoSE3ProjectXYZ*` | Three-coordinate `projection_factor` | [u, v, u - bf/z], including its separate third-row Jacobian; identical pyramid information for all coordinates. |
| `*ToBody` | `camera_from_reference` factor | Project through fixed T_right_left after optimized T_left_world, including the extrinsic rotation in Jacobians. |
| `PoseOptimization` | `optimize_pose` | Fewer than 3 correspondences returns zero; four 10-iteration stages; reset original pose each stage; retain edge levels; re-evaluate omitted edges; mono 5.991/stereo 7.815 chi²; remove Huber after stage 2; one stage for total edges <10; no added cheirality test. |
| `BundleAdjustment` | `bundle_adjustment` numerical graph | Fixed vertices supplied explicitly; optimized XYZ landmarks; source's global mono Huber 5.99 and stereo 7.815; one solve; no automatic observation erasure. |
| Primary visual `LocalBundleAdjustment` | `local_bundle_adjustment` numerical graph | Requires fixed keyframes; 10 robust iterations; mono Huber 5.991; chi² and positive-depth diagnostics; starting lambda 100 for an inertial map. |
| Merge visual `LocalBundleAdjustment` overload | `merge_bundle_adjustment` numerical graph | Five robust iterations, omit outliers/remove Huber, ten least-squares iterations, check all original observations again. |
| Bundled g2o LM | `_levenberg_marquardt` | rho' block weighting; Hessian damping before marginalization; gain ratio denominator +1e-3; lambda initialization 1e-5 max diagonal; factor clipped to [1/3, 2/3]; retry cap 10; stop after three iterations with <0.1% improvement. |

The local-map selection in `mapping.py` supplies local/nonlocal keyframes
and points, applies rejected observation erasure, updates normal/depth estimates,
and increments map change indices. These mutations stay separate from the
array-based BA functions. Global BA's delayed loop-generation commit and propagation through
the spanning tree also belong to map/loop code. Their existence must not be
inferred from the numerical solver's availability.

## Numerical substitutions and limits

- NumPy float64 dense 6x6 pose solve and landmark-block Schur elimination replace
  Eigen/g2o linear algebra. The reduced camera system is currently dense; large
  global BA needs a sparse reduced solve to remain practical. No dense matrix
  is built over all landmarks.
- SE(3) exponential uses its mathematical small-angle series. Bundled
  `SE3Quat::exp` uses `R = I + Omega + Omega²; V = R` below angle 1e-5. The port
  uses correct coefficients and normal manifold rotations; this is a documented
  numerical difference, rather than a different state or objective.
- Nonfinite projection errors are treated as unusable observations. C++'s plain
  `chi2 > threshold` may leave a NaN unclassified; no finite projection or
  negative depth is clamped to a synthetic value.
- The functions are sequential; there is no concurrent stop flag or mid-solve
  cancellation yet. The numerical essential graph and Sim3 refinement live in
  `loop_geometry.py`; `loop_closing.py` now supplies standard loop detection,
  correction, fusion, graph construction, and global BA map commits.
  Visual-inertial optimization is a separate module.

## Source optimization families and graph responsibilities

These are distinct algorithms in the source, not aliases of visual BA:

1. `FullInertialBA`: body-frame pose retraction; velocities; gyro/accel biases;
   temporal preintegration factors; bias random walks or shared initialization
   biases; initialization priors; visual mono/stereo/body factors; selected local
   variables fixed; singular values for initialization observability.
2. `LocalInertialBA`: temporal window with oldest/boundary fixed frames; separate
   visual covisibility additions; boundary IMU robustification and information
   downweighting; divergence checks; visual outlier erasure with close-point
   rules; optional recovery initialization behavior.
3. Three `InertialOptimization` overloads: joint gravity direction/scale/bias/
   velocity initialization; bias-only refinement; scale-and-gravity-only
   refinement. Gravity direction has two degrees of freedom; scale is updated
   multiplicatively and cannot be substituted with unconstrained Euclidean
   scale fitting. All use source-specific priors and fixed-variable choices.
4. `PoseInertialOptimizationLastKeyFrame` and `...LastFrame`: multi-stage visual
   outlier policy; previous-state prior; inertial and bias random-walk factors;
   final Hessian marginalization; retained 15D pose/velocity/bias prior.
5. `Marginalize`: permute requested variable block out, SVD pseudoinverse with
   singular-value cutoff 1e-6, Schur complement, and restore original layout.
6. `OptimizeSim3` (numerical stages implemented in `loop_geometry.py`): paired forward/inverse camera reprojection, fixed matched XYZ
   vertices, five robust iterations, bilateral rejection, at least ten surviving
   correspondences, five/ten nonrobust refinement iterations; optional scale
   lock and matches not explicitly observed in KF2. Despite its name the output
   `mAcumHessian` is assigned zero and NEVER accumulated in this baseline; Python
   preserves that zero output. Under ten surviving pairs leaves input Sim3
   unchanged while still removing first-stage outlier matches.
7. `OptimizeEssentialGraph` (standard graph construction and numerical solve
   implemented in `loop_geometry.py`): graph constraints from loop connections, spanning
   tree, historical loop edges, covisibility with weight >=100, and inertial
   temporal links; fixed map-origin keyframe; optional scale lock; identity 7x7
   information; 20 iterations, initial lambda 1e-16; convert [sR,t] into [R,t/s];
   transform landmarks through their nonoptimized and optimized references.
8. Merge `OptimizeEssentialGraph` overload: both corrected fixed and uncorrected
   vertices, selective constraints, and transfer of map correction through
   nonfixed keyframes and points.
9. `OptimizeEssentialGraph4DoF` (standard graph construction and numerical solve
   implemented in `loop_geometry.py`): inertial loop correction with yaw and position
   variables, six-dimensional rotation/translation edge residual, information
   emphasizing rotation, fixed gravity roll/pitch; cannot be replaced with a
   free SE(3) or Sim(3) graph. This baseline fixes the loop keyframe, disables
   spanning-tree edges via an explicitly null parent, uses temporal edges, and
   sets information diag[1000,1000,1,1,1,1] (the repeated assignment to (0,0)
   leaves yaw weight 1). These quirks are preserved. Body/camera extrinsics
   enter the world-yaw/body-position retraction explicitly.
10. `MergeInertialBA`: explicit temporal welding windows from both maps, fixed
    boundary variables and external covisible frames, full inertial and visual
    factors, and propagation/velocity/bias/map correction outside the window.

## Verification

`tests/test_optimization.py`: 11 passing tests at initial implementation,
including centered numerical derivatives for mono, stereo, and rigid-right
pose/landmark factors; synthetic pose recovery with bad correspondences;
fixed camera and metric structure recovery in joint BA; primary/merge LBA;
and compiled exact baseline g2o pose optimization on identical noisy inputs.
The differential test links the original C++ only as an independent test oracle;
runtime Python has no C++ ORB-SLAM3 binding. It matched pose entries within 2e-8
and produced identical rejected correspondence flags.

The default tests consume shipped golden fixtures and require no C++ checkout.
Run the optional live differential test with `ORBSLAM3_REFERENCE` pointing at
the built C++ checkout. It requires matching Eigen ABI compilation flags
(`-march=native`, as in this local baseline build). Golden fixture JSON records
the exact source-file SHA256 values and input/output arrays.

## Similarity estimation and MLPnP additions

`sim3_solver.py` ports the source Horn quaternion eigenproblem, three-point
sampling, scale formula and bilateral reprojection check. It uses projected
camera-frame 3D points as its reference pixels, not observed keypoint pixels.
The source's `vector<size_t>` max-error thresholds truncate 9.210*octave_variance
to integers. Convergence requires support strictly greater than the configured
minimum. Random sets come from NumPy's RNG rather than DUtils; degenerate
three-point models are skipped instead of allowing source NaNs/zero-axis
division. Identity rotation is represented directly by the quaternion rather
than normalized through a zero axis.

`pnp.py` ports `src/MLPnPsolver.cpp`: six-point bearing nullspaces, uncentered
scatter rank test, planar eigenframe and four pose candidates, general 12-column
design matrix, optional projected bearing covariance, Frobenius rotation fit,
direction selection from the first six points, and five-step additive
Rodrigues Gauss-Newton. Its compact analytic Jacobian is the same chain-rule
derivative as the source's expanded symbolic expressions. NumPy SVD/eigh and
normal solves replace Eigen; stable SO(3) limits handle zero/pi angles.

Two baseline RANSAC quirks remain visible: its trial loop uses OR, so
`iterate(5)` can consume the remaining global budget; `Refine` computes a new
local pose but never assigns it back before checking/returning the current
sampled pose. These are preserved. The special planar branch's translation
scale is wrong for the shipped rank-2 synthetic fixtures, and its GN guard
stops before fixing that scale. A direct call into the original C++
`MLPnPsolver::computePose` reproduces those outputs; the Python tests preserve
them and document that limitation. Off-origin planes may not enter the source's
planar branch because its rank test is uncentered.

`tests/reference/mlpnp.cpp` and `sim3.cpp` are test-only original-library
oracles; runtime does not compile or bind to C++. Recorded general/noisy/planar
MLPnP outputs agree within 2e-8 and Sim3 refinement outputs within 3e-8, with
identical bilateral inlier masks. The optional harness's private-access macro
only exposes the original numerical function in a test process.

MLPnP code additionally retains Steffen Urban/ANU's 2016 BSD-3-Clause copyright,
conditions and disclaimer in `pnp.py`; this notice must travel with the port.

## Loop coordinator

`loop_closing.py` implements `NewDetectCommonRegions`, BoW common-region
verification, temporal verification, `FindMatchesByProjection`, `CorrectLoop`,
`SearchAndFuse`, and `RunGlobalBundleAdjustment` coordination with real map
records and the array-based numerical routines above. `merging.py` owns the
two sensor-dependent map-welding paths. No place-recognition method is claimed
merely from vocabulary scoring: the coordinator performs the geometric stages.

Preserved policies include the 12-keyframe gate; inertial BA2 gate; three
coincidences and two misses; independent loop/merge hypothesis states; three
best candidates per category; candidate-neighborhood rejection near current
connections; BoW >=20; Horn support strictly >15; coarse projection >=50;
optimized paired matches >=20; tight projection >=80; up to three covisible
projection verifications >=30 each; temporal projection >=30, optimized support
>50, repeat projection >=100. Mono-inertial scale-lock overrides follow each
specific source call site, including the BoW refinement call that ignores its
computed override. Inertial merging checks scale [0.9,1.1] and BA1 yaw-only
correction; inertial loops require roll/pitch <0.008 and yaw <0.349 radians,
then enforce yaw-only/unit-scale after BA2.

Landmark replacement/fusion, normal/distance refresh, covisibility regeneration,
new inter-region graph edges, full Sim3 versus gravity-preserving 4DoF graph
selection, reference-based landmark correction, neighbor velocity rotation,
loop-edge protection from culling, map change counters, and global BA commit
through later spanning-tree descendants are explicit. Global BA uses ten
nonrobust visual iterations or seven full inertial iterations. It follows the
source launch gate: always before IMU initialization, otherwise only fewer
than 200 keyframes and a single Atlas map.

Additional exact-baseline quirks are visible in code: temporal verification
discards its optimized camera-to-camera estimate before reprojecting the
original propagated transform forced to unit scale; the BoW best-neighbor
index is never assigned to the candidate pointer; Sim3Solver ignores the
nonempty per-match keyframe provenance vector; the second-neighbor projection
search appends full neighbor vectors despite its preceding filtering loop;
merge-success does not reset its previous miss counter. These are not silently
repaired. The all-points Sim3 path also preserves its source's synthetic
normalized observation, even though that is subsequently compared with camera
pixels and can reject useful matches.

Architectural/numerical limits: worker queues, stop flags, background GBA
generation cancellation, and thread timing are replaced by sequential keyframe
boundaries. GBA commits immediately after a correction, so map updates can
become available earlier than in a real concurrent C++ run. Active-map reset
retains valid archived-map hypotheses/database entries; hypotheses on actually
discarded maps are released as a documented robustness difference. C++ pointer
ordering is replaced by deterministic keyframe IDs when selecting correction
references. These scheduling/order changes can alter later keyframe choices
and numerical trajectories, despite preserving the factor formulations and
major processing order.

Ten coordinator tests include a full 12-keyframe/240-landmark generated scene
through real BoW retrieval, Horn sampling, projection, paired Sim3 refinement,
three-covisible verification, duplicate fusion to 120 points, essential graph
correction and global BA. A second generated scene uses six archived and
twelve current-map keyframes and 200 landmarks, verifies the cross-map region,
and performs real visual map welding/fusion to 100 landmarks with consistent
map/database identities. Other tests isolate temporal decisions, inertial
gates, preserved-map reset behavior, discarded temporal refinement, simultaneous
bad-scale merge/loop handling, and delayed global-BA propagation.
