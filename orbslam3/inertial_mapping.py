"""LocalMapping.cc inertial initialization and refinement scheduling.

This coordinator owns the source's motion-time heuristic and calls explicit
inertial graphs. LocalMapper calls its three phases around local BA/culling;
tracking consumes returned corrections to update its non-keyframe states.
SPDX-License-Identifier: GPL-3.0-or-later
Derived from ORB-SLAM3, Copyright (C) 2017-2021 University of Zaragoza and
Copyright (C) 2014-2016 the ORB-SLAM2 authors; see NOTICE and LICENSE.
"""

from dataclasses import dataclass, field
import copy
import numpy as np

from .geometry import inverse_pose
from .inertial_optimization import (
    BodyState, InertialLink, initial_gravity_velocities,
    optimize_inertial_initialization, refine_gravity_scale,
    full_inertial_bundle_adjustment,
)


def camera_pose_from_body(state, calibration):
    world_body = np.eye(4)
    world_body[:3, :3], world_body[:3, 3] = state.rotation, state.position
    return calibration.T_camera_body @ inverse_pose(world_body)


def body_state_from_camera(pose, calibration, velocity=None, bias=None):
    world_body = inverse_pose(pose) @ calibration.T_camera_body
    state = BodyState(world_body[:3, :3], world_body[:3, 3],
                      np.zeros(3) if velocity is None else velocity)
    if bias is not None:
        state.bias = copy.deepcopy(bias)
    return state


def apply_inertial_scaled_rotation(map_, rotation, scale):
    """Map::ApplyScaledRotation followed by consistent body-state updates.

    Camera centers are scaled, matching C++. Body centers cannot simply be
    multiplied by scale when the camera/IMU lever arm is nonzero.
    """
    previous_states = {identifier: copy.deepcopy(keyframe.body_state)
                       for identifier, keyframe in map_.keyframes.items()
                       if getattr(keyframe, "body_state", None) is not None}
    transform = np.eye(4)
    transform[:3, :3] = rotation
    map_.apply_scaled_rotation(transform, scale, scale_velocity=True)
    for identifier, state in previous_states.items():
        keyframe = map_.keyframes[identifier]
        keyframe.body_state = body_state_from_camera(
            keyframe.pose, keyframe.imu_calibration, rotation @ state.velocity * scale, state.bias)


@dataclass
class InertialMappingUpdate:
    reset_required: bool = False
    initialized: bool = False
    changed: bool = False
    scale: float = 1.
    gravity_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    updated_keyframes: tuple = ()
    event: str | None = None
    bias: object | None = None


class InertialMapper:
    """Sequential LocalMapping::InitializeIMU/ScaleRefinement decisions.

    Call before_local_ba, initialize, and refine in that order, with keyframe
    culling between initialize and refine. ``step`` is a convenience for
    callers without a separate local-mapping coordinator.
    """

    def __init__(self, monocular):
        self.monocular = bool(monocular)
        self.map_id = None
        self.motion_time = 0.
        self.initialization_count = 0
        self.bad_imu = False
        self._motion_keyframe = None

    def _ensure_map(self, current):
        if self.map_id != current.map.id:
            self.map_id = current.map.id
            self.motion_time = 0.
            self.initialization_count = 0
            self.bad_imu = False
            self._motion_keyframe = None

    def _empty(self, current):
        return InertialMappingUpdate(reset_required=self.bad_imu,
                                     initialized=current.map.imu_initialized)

    def before_local_ba(self, current):
        self._ensure_map(current)
        map_ = current.map
        result = self._empty(current)
        if (not map_.imu_initialized or len(map_.keyframes) <= 2
                or self._motion_keyframe == current.id):
            return result
        previous = map_.keyframes.get(current.previous_id)
        earlier = map_.keyframes.get(previous.previous_id) if previous is not None else None
        if previous is None or earlier is None:
            # A broken temporal link has no valid source motion test.
            raise ValueError("initialized inertial mapping requires two preceding temporal keyframes")
        distance = np.linalg.norm(previous.center - current.center) + np.linalg.norm(earlier.center - previous.center)
        if distance > .05:
            self.motion_time += current.timestamp - previous.timestamp
        if not map_.inertial_ba2 and self.motion_time < 10. and distance < .02:
            self.bad_imu = True
            result.reset_required = True
            result.event = "insufficient_motion"
        self._motion_keyframe = current.id
        return result

    @staticmethod
    def _temporal_chain(current):
        keyframes, chain, seen = current.map.keyframes, [], set()
        keyframe = current
        while keyframe is not None:
            if keyframe.id in seen:
                raise ValueError("cycle in inertial temporal keyframe chain")
            seen.add(keyframe.id)
            chain.append(keyframe)
            keyframe = keyframes.get(keyframe.previous_id)
        return chain[::-1]

    @staticmethod
    def _states_links(keyframes):
        states = {keyframe.id: keyframe.body_state for keyframe in keyframes
                  if not keyframe.bad and getattr(keyframe, "body_state", None) is not None}
        links = [InertialLink(keyframe.previous_id, keyframe.id, keyframe.preintegrated)
                 for keyframe in keyframes if keyframe.id in states and keyframe.previous_id in states
                 and keyframe.preintegrated is not None]
        return states, links

    @staticmethod
    def _commit_states(map_, states, reintegrate_ids=()):
        for identifier, state in states.items():
            keyframe = map_.keyframes[identifier]
            keyframe.body_state = copy.deepcopy(state)
            keyframe.pose = camera_pose_from_body(state, keyframe.imu_calibration)
            if keyframe.preintegrated is not None:
                keyframe.preintegrated.set_new_bias(state.bias)
                if identifier in reintegrate_ids:
                    keyframe.preintegrated.reintegrate()

    def _initialize_with_priors(self, current, prior_gyro, prior_accel):
        result = self._empty(current)
        map_ = current.map
        if self.bad_imu or len(map_.keyframes) < 10:
            return result
        chain = self._temporal_chain(current)
        if len(chain) < 10 or current.timestamp - chain[0].timestamp < (2. if self.monocular else 1.):
            return result
        states, links = self._states_links(chain)
        if len(states) != len(chain) or len(links) != len(chain) - 1:
            raise ValueError("IMU initialization needs calibrated body states and every temporal preintegration")
        if not map_.imu_initialized:
            states, gravity_rotation = initial_gravity_velocities(states, links)
            self.motion_time = current.timestamp - chain[0].timestamp
            # Like C++, velocity seeds are written before the scale validity test.
            self._commit_states(map_, states)
        else:
            gravity_rotation = np.eye(3)
        estimate = optimize_inertial_initialization(
            states, links, monocular=self.monocular, gravity_rotation=gravity_rotation,
            prior_gyro=prior_gyro, prior_accel=prior_accel)
        self._commit_states(map_, estimate.states, estimate.reintegrate_ids)
        if estimate.scale < .1:
            result.event = "scale_too_small"
            return result
        applied_scale, applied_rotation = 1., np.eye(3)
        if abs(estimate.scale - 1.) > 1e-5 or not self.monocular:
            applied_scale, applied_rotation = estimate.scale, estimate.gravity_rotation.T
            apply_inertial_scaled_rotation(map_, applied_rotation, applied_scale)
        was_initialized = map_.imu_initialized
        map_.imu_initialized = True
        # The original bImu flags enable inertial graph vertices for these KFs.
        for keyframe in chain:
            keyframe.inertial = True
        self._full_bundle_adjustment(map_, prior_gyro, prior_accel)
        self.initialization_count += 1
        map_.change_index += 1
        result.initialized, result.changed = True, True
        result.scale, result.gravity_rotation = applied_scale, applied_rotation
        result.updated_keyframes = tuple(map_.keyframes)
        result.bias = copy.deepcopy(current.body_state.bias)
        result.event = "initialized" if not was_initialized else "inertial_refinement"
        return result

    def _full_bundle_adjustment(self, map_, prior_gyro, prior_accel):
        from .mapping import reprojection_observations
        keyframes = [keyframe for keyframe in map_.keyframes.values() if not keyframe.bad]
        states, links = self._states_links(keyframes)
        observations = [observation for observation in reprojection_observations(keyframes)
                        if observation.pose_id in states]
        points = {identifier: point.position for identifier, point in map_.points.items() if not point.bad}
        calibrations = {keyframe.id: keyframe.imu_calibration for keyframe in keyframes if keyframe.id in states}
        result = full_inertial_bundle_adjustment(
            states, points, observations, links, calibrations, iterations=100,
            initialize_bias=prior_accel != 0., prior_gyro=prior_gyro, prior_accel=prior_accel,
            inertial_ids={keyframe.id for keyframe in keyframes
                          if getattr(keyframe, "inertial", map_.imu_initialized)})
        self._commit_states(map_, result.states)
        for identifier, position in result.points.items():
            map_.points[identifier].position = position.copy()
        for point in map_.points.values():
            if not point.bad:
                point.update_normal_and_depth()
        map_.change_index += 1

    def initialize(self, current):
        self._ensure_map(current)
        if current.map.imu_initialized:
            return self._empty(current)
        return self._initialize_with_priors(current, 1e2, 1e10 if self.monocular else 1e5)

    def refine(self, current, tracking_ok=True):
        self._ensure_map(current)
        map_ = current.map
        if self.bad_imu or not tracking_ok or not map_.imu_initialized or self.motion_time >= 50.:
            return self._empty(current)
        result = self._empty(current)
        if not map_.inertial_ba1 and self.motion_time > 5.:
            map_.inertial_ba1 = True
            result = self._initialize_with_priors(current, 1., 1e5)
            result.event = "inertial_ba1"
        elif map_.inertial_ba1 and not map_.inertial_ba2 and self.motion_time > 15.:
            map_.inertial_ba2 = True
            result = self._initialize_with_priors(current, 0., 0.)
            result.event = "inertial_ba2"
        # Preserve both the source windows and its outer motion_time < 50 guard.
        in_window = any(start < self.motion_time < start + .5 for start in (25., 35., 45., 55., 65., 75.))
        if self.monocular and len(map_.keyframes) <= 200 and in_window:
            chain = self._temporal_chain(current)
            states, links = self._states_links(chain)
            estimate = refine_gravity_scale(states, links)
            if estimate.scale < .1:
                return result
            if abs(estimate.scale - 1.) > .002:
                apply_inertial_scaled_rotation(map_, estimate.gravity_rotation.T, estimate.scale)
                result.changed, result.scale = True, estimate.scale
                result.gravity_rotation = estimate.gravity_rotation.T
                result.updated_keyframes = tuple(map_.keyframes)
                result.bias = copy.deepcopy(current.body_state.bias)
            map_.change_index += 1
            result.event = "scale_refinement"
        return result

    def step(self, current, keyframes=None, mapper_diagnostics=None, tracking_state="OK", inlier_count=0):
        """Convenience complete schedule; LocalMapper uses individual phases.

        Extra context arguments mirror the tracking coordinator API. Exact
        local BA and keyframe-culling interleaving is the caller's operation.
        """
        motion = self.before_local_ba(current)
        if motion.reset_required:
            return motion
        initialized = self.initialize(current)
        refined = self.refine(current, tracking_ok=(tracking_state == "OK" or tracking_state == 2))
        return refined if refined.changed or refined.event else initialized
