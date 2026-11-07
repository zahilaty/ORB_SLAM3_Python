# Local Python setup

Verified: **221 passed, 5 optional live C++ oracles skipped**,
with the original checkout hidden and networking disabled.

Use `.venv/bin/python -m pytest -q` from this directory. The dedicated Conda
environment, Python dependencies, trained vocabulary and reference fixtures
are local. Runtime and default tests require no original checkout or network.

Recreate with `bash tools/setup_environment.sh`; see `README.md` for own-data
inputs and `COVERAGE.md` for implementation limits. Conda records the environment
path in its user registry; no system packages or shell startup files were changed.

Git excludes `.venv/` and `assets/ORBvoc.txt`; both remain local. A fresh checkout
requires the vocabulary supplied separately (SHA-256 in `AI_CONTEXT.yaml`) before
package installation or tests, plus environment recreation as above.

## Changes: one line per file

| File | Status | What and why |
| --- | --- | --- |
| `.gitignore` | New | Exclude the local environment, trained vocabulary, caches and generated results. |
| `AGENTS.md` | New | Preserve standalone and source-fidelity requirements for future work. |
| `AI_CONTEXT.yaml` | New | Record implementation facts, evidence, decisions and remaining validation. |
| `BASELINE.json` | New | Pin the original revision, local changes and source/asset hashes. |
| `COVERAGE.md` | New | State implemented algorithms, numerical substitutions, repairs and limits. |
| `LICENSE` | New | Retain the original GPL version 3 license text. |
| `LOCAL_SETUP.md` | New | Keep setup instructions and this per-file change inventory short. |
| `NOTICE` | New | Preserve original authors and GPL attribution. |
| `README.md` | New | Explain own-data API, setup including the separately supplied vocabulary, and algorithm reading order. |
| `TASK.md` | New | Retain the complete authorized implementation brief. |
| `THIRD_PARTY_NOTICES.md` | New | Identify retained assets, translated components and license notices. |
| `assets/ORBvoc.txt` | Local, ignored | Retain the unchanged trained vocabulary required by tracking and place recognition outside Git. |
| `docs/inertial_audit.md` | New | Record source mapping, preserved quirks, numerical choices and verification. |
| `docs/merging_audit.md` | New | Record source mapping, preserved quirks, numerical choices and verification. |
| `docs/optimization_audit.md` | New | Record source mapping, preserved quirks, numerical choices and verification. |
| `environment.yml` | New | Recreate this directory’s independent Python environment. |
| `licenses/DBoW2-original-README.txt` | New | Retain this component’s original copyright, conditions and disclaimer. |
| `licenses/DBoW2.txt` | New | Retain this component’s original copyright, conditions and disclaimer. |
| `licenses/MLPnP-ANU-BSD.txt` | New | Retain this component’s original copyright, conditions and disclaimer. |
| `licenses/ORB-WillowGarage-BSD.txt` | New | Retain this component’s original copyright, conditions and disclaimer. |
| `licenses/Sophus-MIT.txt` | New | Retain this component’s original copyright, conditions and disclaimer. |
| `licenses/g2o-BSD.txt` | New | Retain this component’s original copyright, conditions and disclaimer. |
| `main.py` | New | Expand `~/Data/DJI_0690.MP4` for the current user, run monocular video at low resolution with scaled guessed intrinsics and collect the final trajectory. |
| `orbslam3/__init__.py` | New | Expose the small frame-by-frame public API. |
| `orbslam3/camera.py` | New | Implement pinhole/KB8 projection, Jacobians and geometric match checks. |
| `orbslam3/export.py` | New | Export corrected trajectories and map points as generic CSV. |
| `orbslam3/features.py` | New | Implement source pyramid FAST, octree, orientation and rotated descriptors. |
| `orbslam3/geometry.py` | New | Provide explicit SE3/Sim3 and triangulation/epipolar mathematics. |
| `orbslam3/imu.py` | New | Port discrete preintegration, covariance, bias Jacobians and inertial factors. |
| `orbslam3/inertial_mapping.py` | New | Preserve inertial initialization/refinement timing and map corrections. |
| `orbslam3/inertial_optimization.py` | New | Construct and solve visible VI initialization, BA and tracking graphs. |
| `orbslam3/initialization.py` | New | Port source two-view RANSAC, model scoring and reconstruction decisions. |
| `orbslam3/input.py` | New | Prepare own-data frames, stereo/depth measurements and generic videos. |
| `orbslam3/loop_closing.py` | New | Verify places, maintain hypotheses, correct/fuse loops and commit global BA. |
| `orbslam3/loop_geometry.py` | New | Optimize bilateral Sim3 and full/gravity-preserving essential graphs. |
| `orbslam3/map.py` | New | Manage frames, points, observations, covisibility, spanning trees and Atlas. |
| `orbslam3/mapping.py` | New | Triangulate/cull/fuse points, select local BA graphs and cull keyframes. |
| `orbslam3/matching.py` | New | Port the source’s BoW, projection, initialization, triangulation and fusion rules. |
| `orbslam3/merging.py` | New | Weld visual/inertial maps, fuse points and propagate graph/state corrections. |
| `orbslam3/optimization.py` | New | Show visual residuals/Jacobians, block Huber, staged rejection and Schur BA. |
| `orbslam3/pnp.py` | New | Port six-point maximum-likelihood PnP and source RANSAC/refinement decisions. |
| `orbslam3/sim3_solver.py` | New | Port Horn similarity fitting, sampling and symmetric geometric verification. |
| `orbslam3/system.py` | New | Coordinate user inputs and deterministic tracker/mapper/loop boundaries. |
| `orbslam3/tracking.py` | New | Port initialization, motion/reference/local tracking, recovery and localization. |
| `orbslam3/visualization.py` | New | Provide optional feature, match, map, trajectory and diagnostic views. |
| `orbslam3/vocabulary.py` | New | Load the retained tree and implement DBoW2 weighting, scoring and retrieval. |
| `pyproject.toml` | New | Declare the Python package, dependencies, assets and test options. |
| `requirements.lock.txt` | New | Record exact tested Python dependency versions. |
| `tests/fixtures/extractor_cpp.npz` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/geometry_initialization_reference.json` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/imu_reference.json` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/imu_reference.npz` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/mlpnp_reference.json` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/pose_optimization_reference.json` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/provenance.json` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/rgbd_system_reference.json` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/sim3_optimization_reference.json` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/visual_lba_reference.json` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/visual_lba_reference.npz` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/fixtures/vocabulary_cpp.npz` | New | Retain original-C++ reference outputs or their provenance for standalone tests. |
| `tests/reference/mlpnp.cpp` | New | Allow explicit opt-in regeneration using the original C++ as a test oracle. |
| `tests/reference/rgbd_system.cpp` | New | Allow explicit opt-in regeneration using the original C++ as a test oracle. |
| `tests/reference/sim3.cpp` | New | Allow explicit opt-in regeneration using the original C++ as a test oracle. |
| `tests/reference/visual_ba.cpp` | New | Allow explicit opt-in regeneration using the original C++ as a test oracle. |
| `tests/synthetic.py` | New | Render own physical multi-depth images and synchronized analytic IMU inputs. |
| `tests/test_features.py` | New | Check features behavior with numerical/source/integration evidence. |
| `tests/test_geometry_camera.py` | New | Check geometry camera behavior with numerical/source/integration evidence. |
| `tests/test_imu.py` | New | Check imu behavior with numerical/source/integration evidence. |
| `tests/test_inertial_mapping.py` | New | Check inertial mapping behavior with numerical/source/integration evidence. |
| `tests/test_inertial_optimization.py` | New | Check inertial optimization behavior with numerical/source/integration evidence. |
| `tests/test_initialization.py` | New | Check initialization behavior with numerical/source/integration evidence. |
| `tests/test_input.py` | New | Check input behavior with numerical/source/integration evidence. |
| `tests/test_localization.py` | New | Check localization behavior with numerical/source/integration evidence. |
| `tests/test_loop_closing.py` | New | Check loop closing behavior with numerical/source/integration evidence. |
| `tests/test_loop_geometry.py` | New | Check loop geometry behavior with numerical/source/integration evidence. |
| `tests/test_map.py` | New | Check map behavior with numerical/source/integration evidence. |
| `tests/test_mapping.py` | New | Check mapping behavior with numerical/source/integration evidence. |
| `tests/test_matching.py` | New | Check matching behavior with numerical/source/integration evidence. |
| `tests/test_merging.py` | New | Check merging behavior with numerical/source/integration evidence. |
| `tests/test_optimization.py` | New | Check optimization behavior with numerical/source/integration evidence. |
| `tests/test_pnp.py` | New | Check pnp behavior with numerical/source/integration evidence. |
| `tests/test_recovery.py` | New | Check recovery behavior with numerical/source/integration evidence. |
| `tests/test_reference_geometry.py` | New | Check reference geometry behavior with numerical/source/integration evidence. |
| `tests/test_system_fisheye.py` | New | Check system fisheye behavior with numerical/source/integration evidence. |
| `tests/test_system_inertial.py` | New | Check system inertial behavior with numerical/source/integration evidence. |
| `tests/test_system_visual.py` | New | Check system visual behavior with numerical/source/integration evidence. |
| `tests/test_tracking_decisions.py` | New | Check tracking decisions behavior with numerical/source/integration evidence. |
| `tests/test_vocabulary.py` | New | Check vocabulary behavior with numerical/source/integration evidence. |
| `tools/setup_environment.sh` | New | Create the local environment and install locked dependencies/package. |
| `VERIFICATION.json` | New | Record the final original-hidden, network-disabled test result and environment. |
