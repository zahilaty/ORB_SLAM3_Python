import copy
import numpy as np
import pytest
from orbslam3.camera import Pinhole
from orbslam3.features import FeatureSet
from orbslam3.input import Sensor, Calibration
from orbslam3.geometry import make_pose, inverse_pose
from orbslam3.map import Atlas, Frame
from orbslam3.tracking import (TrackingState, need_new_keyframe, Tracker,
                               synchronize_body_pose, synchronize_camera_pose, TrajectoryEntry)
from orbslam3.imu import ImuCalibration
from orbslam3.vocabulary import InvertedDatabase


def need(sensor=Sensor.MONOCULAR,**kwargs):
    args = dict(frame_id=50,timestamp=2.,last_id=30,last_timestamp=1.9,last_reloc_id=0,
                max_frames=30,map_keyframes=5,reference_matches=100,inliers=80,
                tracked_close=200,untracked_close=0,state=TrackingState.OK,
                imu_initialized=True)
    args.update(kwargs)
    return need_new_keyframe(sensor,**args)


def test_visual_keyframe_ratio_strict_boundary_and_software_gates():
    assert need(inliers=89)
    assert not need(inliers=90)
    assert not need(inliers=15)
    assert not need(only_tracking=True)
    assert not need(mapping_stopped=True)
    assert not need(dual_camera=True)
    assert not need(frame_id=29,last_id=20,map_keyframes=31)
    assert need(frame_id=30,last_id=20,map_keyframes=31)


def test_close_depth_trigger_and_queued_keyframe_limit():
    assert need(Sensor.RGBD,inliers=100,tracked_close=99,untracked_close=71)
    assert not need(Sensor.RGBD,inliers=100,tracked_close=100,untracked_close=71)
    assert not need(Sensor.RGBD,inliers=100,tracked_close=99,untracked_close=70)
    assert need(Sensor.RGBD,inliers=20,frame_id=60,mapping_idle=False,queued=2)
    assert not need(Sensor.RGBD,inliers=20,frame_id=60,mapping_idle=False,queued=3)
    assert not need(Sensor.MONOCULAR,inliers=20,frame_id=60,mapping_idle=False)
    assert need(Sensor.MONOCULAR,inliers=20,frame_id=60,mapping_idle=False,mapping_initializing=True)


@pytest.mark.parametrize("sensor",[Sensor.MONOCULAR_IMU,Sensor.STEREO_IMU,Sensor.RGBD_IMU])
def test_uninitialized_imu_keyframe_quarter_second_rule_precedes_other_gates(sensor):
    assert not need(sensor,timestamp=2.,last_timestamp=1.751,imu_initialized=False)
    assert need(sensor,timestamp=2.,last_timestamp=1.75,imu_initialized=False,
                only_tracking=True,mapping_stopped=True)


def test_initialized_imu_temporal_and_weak_monocular_conditions():
    assert need(Sensor.STEREO_IMU,inliers=100,timestamp=2.,last_timestamp=1.5)
    assert not need(Sensor.STEREO_IMU,inliers=100,timestamp=2.,last_timestamp=1.501)
    assert need(Sensor.MONOCULAR_IMU,inliers=74,reference_matches=70)
    assert not need(Sensor.MONOCULAR_IMU,inliers=75,reference_matches=70)
    assert need(Sensor.MONOCULAR_IMU,inliers=100,reference_matches=70,state=TrackingState.RECENTLY_LOST)


def make_frame(index=0,pose=None):
    features = FeatureSet(np.array([[320,240]],np.float32),np.zeros((1,32),np.uint8),np.zeros(1,np.int32),
                          np.zeros(1,np.float32),np.ones(1,np.float32),np.full(1,31,np.float32),1)
    return Frame(index,.1*index,features,Pinhole(500,500,320,240),640,480,
                 np.array([1.,1.2]),np.array([1.,1/1.44]),pose=np.eye(4) if pose is None else pose)


def test_body_camera_pose_synchronization_preserves_nonzero_lever_arm():
    frame = make_frame(pose=make_pose(translation=[-.3,.2,.1]))
    relative = make_pose(translation=[.05,-.01,.02])
    frame.imu_calibration = ImuCalibration(relative,.01,.05,.001,.005)
    original = frame.pose.copy()
    synchronize_body_pose(frame)
    np.testing.assert_allclose(frame.body_state.position,[.25,-.19,-.12],atol=1e-12)
    synchronize_camera_pose(frame)
    np.testing.assert_allclose(frame.pose,original,atol=1e-12)


def test_trajectory_resolves_removed_reference_through_spanning_parent():
    atlas = Atlas()
    parent = atlas.add_keyframe(make_frame())
    reference = atlas.add_keyframe(make_frame(1,make_pose(translation=[-.2,0,0])))
    reference.change_parent(parent)
    frame_pose = make_pose(translation=[-.3,0,0])
    entry = TrajectoryEntry(1.,frame_pose @ inverse_pose(reference.pose),reference,TrackingState.OK)
    reference.set_bad()
    parent.pose = make_pose(translation=[-.1,0,0])
    np.testing.assert_allclose(entry.pose(atlas.all_keyframes)[:3,3],[-.4,0,0])


class TinyVocabulary:
    def transform(self,descriptors,levelsup=4):
        return {0:1.},{0:list(range(len(descriptors)))}

    def score(self,a,b):
        return 1.


def test_monocular_initial_extractor_uses_source_mixed_frame_and_keyframe_ids():
    atlas = Atlas()
    vocabulary = TinyVocabulary()
    tracker = Tracker(atlas, Sensor.MONOCULAR, Calibration(Pinhole(500,500,320,240)),
                      vocabulary, InvertedDatabase(vocabulary))
    tracker.state = TrackingState.OK
    tracker.current = make_frame(40)
    tracker.initial_extractor_keyframe_id = 5
    assert not tracker.initializing_features
    tracker.initial_extractor_keyframe_id = 11
    assert tracker.initializing_features
    tracker.current.id = 41
    assert not tracker.initializing_features
    tracker.state = TrackingState.NOT_INITIALIZED
    assert tracker.initializing_features


def test_backward_image_timestamp_starts_an_atlas_map_without_reusing_last_frame():
    atlas = Atlas()
    vocabulary = TinyVocabulary()
    tracker = Tracker(atlas,Sensor.MONOCULAR,Calibration(Pinhole(500,500,320,240)),vocabulary,
                      InvertedDatabase(vocabulary))
    tracker.last = make_frame(3)
    tracker.state = TrackingState.OK
    before = atlas.current.id
    result = tracker.track(make_frame(2))
    assert result.map_id != before
    assert tracker.last is None and tracker.state == TrackingState.NO_IMAGES_YET
    assert result.diagnostics["reset"] == "backward timestamp"


def test_depth_initialization_preserves_strict_500_feature_threshold():
    atlas = Atlas()
    vocabulary = TinyVocabulary()
    tracker = Tracker(atlas,Sensor.RGBD,Calibration(Pinhole(500,500,320,240),baseline_fx=50.),vocabulary,
                      InvertedDatabase(vocabulary))
    frame = make_frame()
    # Deliberately isolate the source decision; exact feature positions do not
    # matter until the threshold permits depth initialization.
    feature = frame.features
    frame.features = FeatureSet(*(np.repeat(getattr(feature,name),500,axis=0)
                                  for name in ("points","descriptors","octaves","angles","responses","sizes")),500)
    frame.map_points = [None]*500
    frame.right_u = np.full(500,310.)
    frame.depth = np.full(500,5.)
    frame.outliers = np.zeros(500,bool)
    tracker._initialize_depth(frame)
    assert tracker.state == TrackingState.NO_IMAGES_YET
    assert not atlas.current.keyframes
