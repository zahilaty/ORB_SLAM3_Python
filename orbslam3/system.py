"""User-facing frame-by-frame API and deterministic processing boundary.

Algorithm implementations live in tracking/mapping/loop_closing. This small
coordinator prepares inputs and commits their sequential map updates.
SPDX-License-Identifier: GPL-3.0-or-later
"""

from pathlib import Path
import sys
from .input import Sensor, Calibration, FrameInput, FrameBuilder
from .map import Atlas
from .vocabulary import Vocabulary, InvertedDatabase
from .tracking import Tracker
from .mapping import LocalMapper


class SLAM:
    def __init__(self, sensor, calibration, *, vocabulary=None, nfeatures=1000,
                 scale_factor=1.2, nlevels=8, initial_fast=20, minimum_fast=7,
                 loop_closing=True, fast_imu_init=False):
        self.sensor, self.calibration = Sensor(sensor), calibration
        if vocabulary is None:
            vocabulary = Path(__file__).resolve().parent.parent / "assets" / "ORBvoc.txt"
            if not vocabulary.is_file():
                vocabulary = Path(sys.prefix) / "share" / "orbslam3" / "ORBvoc.txt"
        self.vocabulary = Vocabulary.from_text(vocabulary) if isinstance(vocabulary, (str, Path)) else vocabulary
        self.database = InvertedDatabase(self.vocabulary)
        self.atlas = Atlas(inertial=self.sensor.inertial)
        self.builder = FrameBuilder(self.sensor, calibration, nfeatures, scale_factor, nlevels, initial_fast, minimum_fast)
        self.tracker = Tracker(self.atlas, self.sensor, calibration, self.vocabulary, self.database,
                               fast_imu_init=fast_imu_init)
        self.mapper = LocalMapper(self.atlas, self.sensor, self.vocabulary, self.database)
        self.loop_closer = None
        if loop_closing:
            from .loop_closing import LoopCloser
            self.loop_closer = LoopCloser(self.atlas, self.database, self.vocabulary, self.sensor)

    def process(self, data):
        if not isinstance(data, FrameInput):
            raise TypeError("process expects FrameInput(timestamp, image, right_image=None, depth=None, imu=())")
        self.tracker.far_points = self.mapper.far_points
        self.tracker.far_threshold = self.mapper.far_threshold
        frame = self.builder.build(data, initializing=self.tracker.initializing_features)
        old_map = self.atlas.current.id
        result = self.tracker.track(frame, data.imu)
        if self.atlas.current.id != old_map:
            self.mapper.reset()
            if self.loop_closer is not None:
                self.loop_closer.reset_active_map(old_map)
        pending = self.tracker.pending_keyframes.copy()
        self.tracker.pending_keyframes.clear()
        updates = []
        for index, keyframe in enumerate(pending):
            self.mapper.pending_keyframes = len(pending)-index-1
            self.mapper.matches_inliers = self.tracker.inliers
            diagnostics = self.mapper.process(keyframe, self.tracker.state)
            updates.append(diagnostics)
            for name in ("inertial_initialization", "inertial_refinement"):
                update = diagnostics.get(name)
                if update is not None and update.changed:
                    # UpdateFrameIMU runs before loop detection and donor map
                    # transfer; scale only the history in this map at this time.
                    self.tracker.apply_inertial_corrections(
                        [(keyframe.map.id, update.scale, update.bias, keyframe.id)])
            if diagnostics["inertial_reset_required"]:
                self.tracker._new_map(discard=True)
                self.mapper.reset()
                if self.loop_closer is not None:
                    self.loop_closer.reset_active_map(keyframe.map.id)
                break
            if self.loop_closer is not None:
                previous_current_ids = set(keyframe.map.keyframes)
                loop_update = self.loop_closer.process(keyframe)
                diagnostics["loop_closing"] = loop_update
                merge = loop_update.get("diagnostics", {}).get("merge")
                if self.sensor.inertial and merge and loop_update["merge"]:
                    # The merge may reparent temporal history and change the
                    # active map. The tracker retains the current temporal KF.
                    self.tracker.apply_merge_correction(merge, history_keyframe_ids=previous_current_ids)
        result.diagnostics["mapping"] = updates
        # TrackingResult keeps the frame's pose at the tracking boundary, as
        # TrackMonocular/Stereo/RGBD do; trajectory() resolves map corrections.
        return result

    @property
    def frame(self):
        return self.tracker.current

    def trajectory(self):
        """List of (seconds, T_cw, map_id, state), including map/BA corrections."""
        return [(entry.timestamp, entry.pose(self.atlas.all_keyframes), entry.reference.map.id, entry.state)
                for entry in self.tracker.history]

    def set_localization_mode(self, enabled=True):
        """Pause mapping and use the source visual localization/VO branch."""
        self.tracker.set_localization_mode(enabled)
        self.mapper.accepts_keyframes = not enabled
