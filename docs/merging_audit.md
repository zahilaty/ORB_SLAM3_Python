# Map merging source audit

Baseline: `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`. The Python entry point is
`merging.merge_maps(atlas, current, matched, similarity, sensor, ...)`.
`similarity` maps **matched-map world into current camera**. The derived
world correction is `similarity.inverse() @ Sim3(current.pose)`, mapping
current-map world into matched-map world. The loop coordinator can supply its
already gravity-restricted correction as `world_transform`.

| Source operation | Python implementation |
| --- | --- |
| `LoopClosing::Run` inertial initialization/scale/yaw merge gates | Also enforced by `merge_maps` for direct callers |
| `MergeLocal` visual welding windows | Current/matched KF, best 25 covisibles, five expansions by best 12 |
| Correct local keyframes and points through reference cameras | Relative Sim3 corrections, translation/scale conversion, reference-window filtering |
| Local transfer and Atlas active-map change | Explicit map membership transfer, source Map ID replacement, database map-ID synchronization |
| Reverse donor spanning-tree path at the welding pair | Explicit parent/child updates without introducing a cycle |
| `SearchAndFuse` Sim3 overload | Existing projection/descriptor/scale matching, threshold four; matched-side points survive replacements |
| Visual welding BA overload | Five robust and ten plain iterations, source chi-square/depth rejection, observation erasure |
| Noncritical monocular geometry | Same relative Sim3/reference-point correction as `MergeLocal` |
| Noncritical depth-supported geometry | Literal merging essential-graph overload, fixed/corrected/remaining pose groups, twenty LM iterations |
| `MergeLocal2` inertial map correction and transfer | Scale/rotation/translation update, unfinished-BA2 bias refinement, donor transfer, tree reversal |
| Inertial fusion neighborhoods | Up to six current and six matched covisibles; collect points until count exceeds 1000 |
| `MergeInertialBA` two-map window | Six current temporal frames, matched predecessor/next window up to twelve vector entries, fixed older-map boundary, extra covisibles from first thirty ranked local points |
| Inertial welding graph | Temporal pose/velocity/bias variables, pose-only covisibles, fixed boundary, IMU/bias-walk/visual factors, eight LM iterations at initial lambda 1000 |

All geometry and graph construction is readable Python. Numerical sparse
linear solves use SciPy; graph LM acceptance, block Huber weights, pose and bias
parameterizations come from the source port. Tracking consumes returned
corrections to update non-keyframe states; merging does not retain a hidden
second tracking state. A sequential caller has no concurrent mapping queue or
running global BA to interrupt/relaunch.

## Preserved source behavior and repaired defects

- Visual merging retains the **matched Map object**, then sets its map ID to
  the former current-map ID. Atlas dictionary keys and database entries are
  updated consistently with that ID change. The former current map becomes
  bad and is removed from active Atlas maps. Both welding KFs receive merge
  edges and are protected from culling.
- Inertial `MergeLocal2` retains the **current Map object**, transforms its
  geometry into the matched world, then transfers all matched-map objects.
  This exact source branch leaves the donor map empty in Atlas and does not
  add merge edges or run the visual branch's bad-map cleanup. The Python
  result explicitly reports `retained_empty_map=True`.
- If the original current inertial map has fewer than ten KFs, the source
  returns after transfer/tree rebuilding/fusion and skips welding BA. This
  early return is preserved and `welding_ba` is `None`.
- If the current inertial map has not completed BA2, source bias refinement
  runs before transfer and sets IMU initialized, BA1 and BA2 flags. Gyroscope
  changes above `.01` trigger raw-measurement reintegration.
- Inertial scale must be in `[.9, 1.1]`. After BA1, roll/pitch of the world
  correction are suppressed and scale is set to one. This preserves the
  source's log-rotation yaw extraction rather than replacing it with a
  different Euler-angle projection.
- The merging essential-graph overload uses its **old inverse first pose**
  even in the good/good pose branch; a target-only fixed pose uses default
  identity for that inverse. The Python edge builder retains and documents
  this literal source formula. The source's duplicate `addEdge(en)` call for
  the same pointer has no extra weighting effect because the graph owns a set.
- `sortByVal` orders local-point observation counts **ascending**, despite the
  apparent intent to select well-observed points. Python preserves that order;
  exact ties use stable point IDs instead of C++ pointer addresses.
- The inertial temporal expansion can revisit already selected predecessor
  frames when the matched KF has no next frame. The bounded source vector then
  repeats vertices and edges. Python keeps dictionary vertices unique and
  preserves the repeated temporal edges/weights.
- Source `MergeInertialBA` creates only pose vertices for extra covisibles, but
  its recovery code then dereferences absent velocity/bias vertices when those
  KFs are inertial. Python keeps their velocity/bias unchanged and recovers
  their optimized poses. This is an explicit crash repair. An inertial link
  whose predecessor is a pose-only covisible is skipped, matching the source's
  missing-vertex test.
- Per-keyframe inertial readiness is retained during transfer. Keyframes from
  an uninitialized matched map remain visual-only even though the survivor's
  map-wide IMU flag is true. Their velocities/biases stay fixed and their IMU
  links are excluded, following source `bImu` selection.
- Source visual and inertial welding BA overloads construct left-camera
  observations only. These merging wrappers preserve that selection; right
  observations are still supported in ordinary local/full VI BA. This is an
  inherited source limitation for unrectified stereo welding.
- Camera/IMU lever arms stay rigid during map scaling. Body poses are
  recomputed from corrected camera poses and their fixed extrinsics; velocities
  receive the source rotation and, when nonunit, scale correction.
- Covisibility and representative descriptors are refreshed only at their
  source merge boundaries. Cross-map observation helpers retain access to
  not-yet-transferred keyframes without a whole-map refresh that would make
  additional graph connections available earlier than in C++.

## Validation and remaining limits

`tests/test_merging.py` exercises two-map monocular similarity and RGB-D metric
welding, projection preservation, duplicate fusion, parent-tree integrity,
Atlas/database ID changes, cross-map observation transfer, literal good/bad
essential-graph measurements, noncritical monocular and depth-supported
propagation, inertial pre-mutation gates, unfinished-map bias refinement,
small-map early return/empty donor, and twelve-plus-twelve-frame inertial
welding with the eight-iteration graph.

The tests use synthetic map geometry and retained descriptors. They validate
merging operations given a verified place match; end-to-end image-based place
recognition, merge detection frequency, and long-term recovery need separate
integration validation. No claim of 99% fidelity is inferred from these tests.
