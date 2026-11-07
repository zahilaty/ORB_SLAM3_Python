# Task: Reimplement ORB-SLAM3 in Python

## 1. Objective

We have cloned the original ORB-SLAM3 repository here and successfully executed its C++ examples.

Your task is to create a new  **Python reimplementation of the complete ORB-SLAM3 algorithm** at ~/Projects/ORB_SLAM3_Python

The primary purpose is educational: I am familiar with SLAM theory and want to study the **actual algorithmic implementation of ORB-SLAM3** through readable Python code.

I am not asking you to implement a generic SLAM system, a simplified educational SLAM framework, or an approximate equivalent.

**The Python implementation must reproduce the algorithmic logic of the original ORB-SLAM3 as faithfully as possible, while allowing substantial simplification of its software architecture.**

My priorities, in order, are:

1. Algorithmic fidelity to ORB-SLAM3.
2. Readability, simplicity, and conciseness of the Python code.
3. Functional correctness.
4. Computational efficiency.

Real-time performance is not required. However, the implementation should be reasonably efficient and capable of processing real image sequences.

## 2. Algorithmic fidelity — critical requirement

Aim for approximately 99% algorithmic fidelity.

By this, I mean preserving the original algorithmic steps, including:

- Mathematical formulations and objective functions.
- Initialization procedures.
- Feature extraction and matching strategies.
- Tracking logic and state transitions.
- Geometric estimation and optimization procedures.
- Keyframe insertion and removal criteria.
- Map point creation, management, and culling.
- Outlier rejection and failure-recovery mechanisms.
- Loop detection, loop correction, and map merging.
- Visual-inertial estimation and initialization.
- Relevant thresholds, heuristics, and algorithmic parameters.

Do not replace an original ORB-SLAM3 algorithm with an easier or more modern alternative simply because it is shorter to implement.

For example:

- Do not replace ORB-SLAM3's feature extraction and distribution strategy with ordinary `cv2.ORB_create()` unless you establish that the relevant behavior is equivalent.
- Do not replace its feature matching strategy with a generic matcher if that changes its matching logic.
- Do not replace its initialization procedure with a generic OpenCV pose estimation pipeline.
- Do not replace its bundle adjustment formulation with a mathematically different optimization problem.
- Do not replace its place recognition algorithm with an unrelated modern method.

You may use different implementations of numerical operations when they preserve the mathematical operation and relevant algorithmic behavior.

Exact floating-point results, numerical trajectories, and bit-level compatibility are **not required**.

When exact equivalence cannot reasonably be achieved, preserve the original mathematical formulation and decision logic, and explicitly document the deviation.

If necessary, prefer slightly longer code over an algorithmic shortcut.

## 3. Implement the complete algorithm

All principal ORB-SLAM3 capabilities must be included.

This includes:

### Sensor configurations
- Monocular
- Stereo
- RGB-D
- Monocular + IMU
- Stereo + IMU
- RGB-D + IMU

Support the relevant camera models, including pinhole and fisheye models implemented in the original repository.

### Tracking
- Feature extraction and feature matching.
- Monocular and depth-based initialization.
- Visual-inertial initialization.
- Motion-model tracking.
- Reference-keyframe tracking.
- Local-map tracking.
- Pose optimization.
- Relocalization.
- Tracking failure detection and recovery.
- Keyframe insertion decisions.

### Local mapping
- Map point triangulation.
- Map point and keyframe management.
- Covisibility relationships.
- Local bundle adjustment.
- Keyframe culling.
- Map point culling and fusion.
- Visual-inertial local optimization.

### Loop closing and map merging
- Place recognition.
- Loop candidate detection and geometric verification.
- Loop correction.
- Pose graph optimization.
- Map merging.
- Global optimization when applicable.
- ORB-SLAM3 Atlas and multiple-map functionality.

### Inertial estimation
- IMU preintegration.
- Bias estimation.
- Gravity and scale initialization.
- Visual-inertial optimization.
- Inertial map initialization and refinement.
- Relevant inertial failure-recovery procedures.

This is a description of the expected scope, not an exhaustive algorithm checklist.

**Inspect the original implementation and identify additional algorithms or important processing steps that are missing from this list.**

Do not omit them merely because I did not explicitly name them.

## 4. Python implementation philosophy

You are free to completely redesign the software architecture to maximize readability.

You do not need to preserve:

- Original C++ classes or inheritance hierarchies.
- Original directory structure.
- Original multithreaded implementation.
- C++ memory-management patterns.
- Original third-party dependencies.
- Complicated configuration and initialization infrastructure.

Use Pythonic structures and an organization that makes the algorithm easy to follow.

Prefer straightforward functions, small classes when appropriate, NumPy arrays, and simple data structures.

Avoid unnecessary abstraction, wrappers, factories, callbacks, and inheritance.

Do not create a large software framework merely to reproduce the original architecture.

A sequential implementation is acceptable and probably preferable, provided the original algorithmic operations and their logical execution order are preserved.

Take special care when converting concurrent processing to sequential processing: changing when map updates or optimizations become available can affect algorithmic behavior.

The objective is to reduce **software-engineering complexity**, not algorithmic complexity.

## 5. Libraries and numerical implementation

Use modern, well-maintained Python libraries.

Prefer:

- NumPy for numerical operations.
- OpenCV for image processing and established computer vision primitives.
- SciPy for numerical optimization, sparse matrices, and linear algebra.
- Other mature libraries when they materially simplify implementation without hiding or changing important algorithms.

There is no need to manually implement standard operations such as matrix multiplication, SVD, or basic image filtering.

However, preserve algorithmic transparency.

For instance, in bundle adjustment:

- The construction of the optimization problem must remain understandable.
- State variables and their parameterization must be identifiable.
- Residual functions and their mathematical meaning must be visible.
- The optimization objective, information matrices, and robust losses must be apparent.
- The selection of optimized and fixed variables must follow ORB-SLAM3.
- The optimization stages and outlier rejection must remain faithful to the original.

For optimization, prefer a Pythonic implementation using NumPy and SciPy, but you may also use Python bindings for g2o (the optimizer used by the original ORB-SLAM3) if they provide a clean, maintainable interface.

Choose whichever approach provides the best combination of algorithmic fidelity, readability, and reasonable computational efficiency.

The following must remain transparent and readable in Python:

Construction of the optimization problem and factor graph.

Definition of state variables, residuals, and Jacobians.

Robust kernels, information matrices, and weighting.

State parameterization, including SE(3) and Sim(3) operations.

Optimization stages, outlier rejection, and convergence criteria.

You may delegate generic numerical optimization operations to an external library.

Do not replace ORB-SLAM3's optimization formulations with those of another SLAM framework. In particular, do not introduce GTSAM/iSAM2 simply for convenience.

Note that ORB-SLAM3 uses custom g2o vertex and edge types. If available Python bindings do not adequately support these, prefer a readable NumPy/SciPy implementation rather than introducing complicated wrappers or modifying the original C++ code.

You may use library implementations for the numerical solution of the optimization problem, but do not hide the entire SLAM algorithm behind an opaque library.

Do not simply create Python bindings around the original ORB-SLAM3 C++ implementation.

The important algorithms must be implemented in readable Python.

## 6. Code organization

Choose a clean, compact Python package structure.

Organize the code according to algorithmic responsibilities rather than the original C++ architecture.

Most importantly, keep the **core SLAM algorithms separate from supporting code**.

Examples of supporting code:

- Input adapters.
- Video readers.
- Configuration parsing.
- Calibration loading.
- Visualization.
- Logging.
- Debugging.
- Exporting results.

The core algorithm files should contain as little of this supporting code as possible.

I want to open the implementation of tracking, mapping, bundle adjustment, or IMU preintegration and read the actual algorithm without navigating through unrelated infrastructure.

Use descriptive variable and function names. Keep functions focused and avoid unnecessary indirection.

Comments should explain non-obvious algorithmic decisions, mathematical conventions, and important steps.

Do not add extensive comments explaining elementary Python syntax or basic mathematical operations.

Do not over-engineer the documentation or code structure.

## 7. Input interface — my own data

I do not want support for the published ORB-SLAM datasets.

Remove all dataset-specific loaders, wrappers, configuration files, and scripts from the Python implementation.

Instead, design a simple, general input interface through which I can provide my own data.

The interface should support the six sensor configurations listed above.

It should accept the necessary inputs, including:

- Images or stereo image pairs.
- Depth maps when applicable.
- Timestamps.
- IMU measurements with timestamps.
- Camera intrinsics and distortion parameters.
- Camera-to-camera and camera-to-IMU extrinsics when applicable.
- Relevant camera and IMU calibration parameters.

Provide a simple way to process a sequence frame by frame.

For convenience, include a minimal adapter for reading my own video files, without making video handling part of the algorithmic core.

Keep data loading and preprocessing separate from the SLAM algorithms.

You may choose the exact Python API.

Keep it simple, consistent, and easy for me to modify.

## 8. Visualization and debugging

Include basic tools for visualizing:

- Extracted and matched features.
- Current tracking status.
- Camera trajectory.
- Reconstructed 3D map points.
- Keyframes and their poses.
- Relevant optimization or tracking diagnostics.

Keep visualization code completely separate from the SLAM core.

Visualization must not be required for running the algorithms.

I should be able to disable it without changing the tracking or mapping code.

## 9. Validation

The implementation should be functionally correct, not merely a readable translation.

Use appropriate unit tests and integration tests.

Where practical, compare the Python implementation against the original C++ version using identical inputs and parameters.

Pay particular attention to verifying:

- Geometric conventions.
- Camera projection and unprojection.
- Pose transformations.
- Feature extraction and matching.
- Triangulation.
- Optimization residuals and Jacobians.
- Bundle adjustment.
- IMU preintegration and its Jacobians.
- Initialization and tracking decisions.

Numerical outputs need not be identical, but mathematical and algorithmic discrepancies should be investigated.

Do not silently change an algorithm to make tests pass.

No published SLAM dataset integrations are required.

You may use synthetic inputs, small generated sequences, or minimal test fixtures.

## 10. Algorithm coverage and deviations

Maintain a concise document describing implementation completeness.

It should identify:

1. Major ORB-SLAM3 algorithmic components.
2. Whether each has been implemented.
3. Any algorithmic deviations from the original.
4. Any numerical library substitutions.
5. Any known limitations.

This document is important because I want to know that I am studying the actual ORB-SLAM3 algorithms rather than approximations.

Do not describe an algorithm as fully implemented if important steps have been omitted.

A component is not complete merely because its Python interface exists.

## 11. Exclusions

Do not implement unnecessary infrastructure.

In particular, exclude:

- ROS integration.
- Dataset-specific implementations and loaders.
- Original example applications.
- Pangolin and elaborate graphical interfaces.
- C++ bindings or wrappers around the original system.
- Complex configuration frameworks.
- Benchmarking infrastructure unrelated to algorithm correctness.
- Unnecessary compatibility layers.
- Unnecessary deployment infrastructure.

An important exception: if an asset or component is needed by the actual algorithm, retain it.

For example, the visual vocabulary used for place recognition is an algorithmic dependency, not merely a dataset accessory. Preserve the necessary vocabulary and retrieval behavior even if its Python implementation differs.

## 12. Implementation strategy

First inspect the original source code thoroughly.

You may look atORB-SLAM3 paper , but the source code in this repo is the authoritative reference for this task.

Use the exact local repository revision as the algorithmic baseline.

Identify the major algorithms, their dependencies, and their relationships.

Then plan an implementation strategy that prioritizes correctness and readability.

You may implement the system incrementally, with tests along the way.

The exact development order is your decision.

However:

- Do not sacrifice algorithm coverage to reduce the project size.
- Do not substitute placeholder algorithms for difficult components.
- Do not declare the implementation complete while major algorithms are missing.
- Do not silently skip difficult cases, sensor modes, or recovery procedures.

If a component cannot be completed, explicitly mark it as incomplete.

Do not report intended functionality as implemented functionality.

## 13. Licensing

Respect the original ORB-SLAM3 license, copyright notices, and attribution requirements. Identify any additional licensing obligations associated with retained assets or reused source components.

## 14. Final expectations

The final product should be a **complete, readable Python reimplementation of ORB-SLAM3**.

Someone familiar with SLAM theory should be able to read the Python source and understand precisely how ORB-SLAM3 performs tracking, mapping, optimization, loop closure, and visual-inertial estimation.

The implementation should preserve the original algorithmic behavior as closely as practical, while eliminating unnecessary architectural and infrastructure complexity.

**Remember: simplify the code, not the algorithm.**
