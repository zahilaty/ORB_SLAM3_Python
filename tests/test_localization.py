"""Visual-only localization/VO decisions and per-frame IMU map corrections."""
import copy
from unittest.mock import patch
import numpy as np
import pytest
from orbslam3.camera import Pinhole
from orbslam3.features import FeatureSet
from orbslam3.geometry import Sim3, make_pose, inverse_pose
from orbslam3.imu import Bias, ImuCalibration, Preintegrated, predict_body_state
from orbslam3.inertial_mapping import body_state_from_camera, camera_pose_from_body
from orbslam3.inertial_optimization import BodyState
from orbslam3.input import Sensor, Calibration
from orbslam3.map import Atlas, Frame
from orbslam3.tracking import Tracker, TrackingState, TrajectoryEntry
from orbslam3.vocabulary import InvertedDatabase


class VocabularyForTest:
    def transform(self,descriptors,levelsup=4):
        return {0:1.},{0:list(range(len(descriptors)))}

    def score(self,a,b):
        return 1.


def frame_from_scene(index, world, descriptors, translation=(0,0,0), calibration=None):
    camera = Pinhole(400,400,320,240)
    local = np.asarray(world)+translation
    n = len(world)
    features = FeatureSet(camera.project(local).astype(np.float32),descriptors.copy(),np.zeros(n,int),
                          np.zeros(n,np.float32),np.ones(n,np.float32),np.full(n,31,np.float32),n)
    frame = Frame(index,index/30,features,camera,640,480,np.array([1.,1.2,1.44]),
                  1/np.array([1.,1.44,2.0736]),baseline_fx=40.,depth_threshold=4.,
                  pose=make_pose(translation=translation),right_u=features.points[:,0]-40/local[:,2],depth=local[:,2].copy())
    if calibration is not None:
        frame.imu_calibration = calibration
        frame.body_state = body_state_from_camera(frame.pose,calibration)
    return frame


def scene(count=30):
    rng = np.random.default_rng(49)
    world = rng.uniform([-.8,-.5,2.],[.8,.5,4.],(count,3))
    descriptors = rng.integers(0,256,(count,32),dtype=np.uint8)
    return world,descriptors


def localization_scene(count=30, persistent=0, sensor=Sensor.RGBD):
    atlas,vocabulary = Atlas(),VocabularyForTest()
    world,descriptors = scene(count)
    reference = atlas.add_keyframe(frame_from_scene(0,world,descriptors))
    tracker = Tracker(atlas,sensor,Calibration(reference.camera,baseline_fx=40.,depth_threshold=4.),
                      vocabulary,InvertedDatabase(vocabulary))
    tracker.reference = tracker.last_keyframe = reference
    tracker.last = frame_from_scene(40,world,descriptors)
    tracker.last.reference_id = reference.id
    tracker.velocity = make_pose(translation=[-.03,0,0])
    tracker.state = TrackingState.OK
    tracker.history = [TrajectoryEntry(tracker.last.timestamp,tracker.last.pose.copy(),reference,TrackingState.OK)]
    tracker.set_localization_mode(True)
    current = frame_from_scene(41,world,descriptors,translation=[-.03,0,0])
    for index in range(persistent):
        point = atlas.add_point(world[index],reference)
        reference.map_points[index] = tracker.last.map_points[index] = point
        point.add_observation(reference,index)
        point.compute_descriptor()
        point.update_normal_and_depth()
    return tracker,current,world


def test_temporary_depth_points_have_original_geometry_but_do_not_enter_atlas():
    tracker,current,world = localization_scene()
    tracker._update_last_frame()
    assert len(tracker.temporary_points) == len(world)
    assert not tracker.atlas.current.points
    for index,point in enumerate(tracker.last.map_points):
        assert point.observation_count == 0 and not point.observations
        np.testing.assert_allclose(point.position,world[index],atol=3e-7)
        np.testing.assert_array_equal(point.descriptor,tracker.last.features.descriptors[index])
        np.testing.assert_allclose(np.linalg.norm(point.normal),1.)
        assert point.max_distance > point.min_distance > 0
    assert tracker.atlas.next_point_id == len(world)


def test_temporary_depth_policy_includes_first_101_far_points_and_skips_keyframe():
    tracker,current,world = localization_scene(130)
    tracker.last.depth_threshold = 1.
    tracker._update_last_frame()
    assert len(tracker.temporary_points) == 101
    tracker,current,world = localization_scene()
    tracker.last_keyframe_frame_id = tracker.last.id
    tracker._update_last_frame()
    assert not tracker.temporary_points
    tracker,current,world = localization_scene(sensor=Sensor.MONOCULAR)
    tracker._update_last_frame()
    assert not tracker.temporary_points


@pytest.mark.parametrize("count,accepted",[(20,False),(21,True)])
def test_localization_motion_requires_strictly_more_than_20_total_inliers(count,accepted):
    tracker,current,_ = localization_scene(count)
    assert tracker._track_motion(current,False) == accepted
    assert tracker.visual_odometry
    assert tracker.inliers == count
    assert not tracker.atlas.current.points


@pytest.mark.parametrize("persistent,vo",[(9,True),(10,False)])
def test_vo_fallback_depends_on_persistent_map_matches_not_depth_matches(persistent,vo):
    tracker,current,_ = localization_scene(30,persistent)
    assert tracker._track_motion(current,False)
    assert tracker.visual_odometry == vo
    assert len(tracker.atlas.current.points) == persistent


def test_vo_frame_skips_local_map_cleans_temporaries_and_never_adds_keyframes():
    tracker,current,_ = localization_scene()
    map_ = tracker.atlas.current
    keyframe_ids = set(map_.keyframes)
    with patch.object(tracker,"_track_local_map",side_effect=AssertionError("VO must bypass local map")):
        result = tracker.track(current)
    assert result.state == TrackingState.OK and result.inliers == 30
    assert result.diagnostics["localization_only"] and result.diagnostics["visual_odometry"]
    assert set(map_.keyframes) == keyframe_ids and not map_.points
    assert not tracker.pending_keyframes and not tracker.temporary_points
    assert all(point is None for point in current.map_points)


def test_failed_relocalization_restores_motion_pose_associations_and_outliers():
    tracker,current,_ = localization_scene(3,persistent=3)
    tracker.visual_odometry = True
    motion_points = tracker.last.map_points.copy()
    motion_pose = make_pose(translation=[-.1,.02,0])
    motion_outliers = np.array([False,True,False])
    counts = [point.found for point in motion_points]
    def motion(frame,map_updated):
        frame.pose,frame.map_points,frame.outliers = motion_pose.copy(),motion_points.copy(),motion_outliers.copy()
        return True
    def relocalize(frame):
        frame.pose = make_pose(translation=[10,20,30])
        frame.map_points,frame.outliers = [None]*3,np.zeros(3,bool)
        return False
    with patch.object(tracker,"_track_motion",side_effect=motion),patch.object(tracker,"_relocalize",side_effect=relocalize):
        assert tracker._track_localization(current,False)
    np.testing.assert_array_equal(current.pose,motion_pose)
    assert current.map_points == motion_points
    np.testing.assert_array_equal(current.outliers,motion_outliers)
    assert [point.found for point in motion_points] == [counts[0]+1,counts[1],counts[2]+1]


def test_successful_relocalization_overrides_motion_and_reenables_local_map():
    tracker,current,_ = localization_scene(3,persistent=3)
    tracker.visual_odometry = True
    relocalized_pose = make_pose(translation=[-.2,0,0])
    def relocalize(frame):
        frame.pose = relocalized_pose.copy()
        return True
    with patch.object(tracker,"_track_motion",return_value=True),patch.object(tracker,"_relocalize",side_effect=relocalize):
        assert tracker._track_localization(current,False)
    assert not tracker.visual_odometry
    np.testing.assert_array_equal(current.pose,relocalized_pose)


def test_localization_does_not_add_normal_mode_reference_fallback_after_failed_motion():
    tracker,current,_ = localization_scene()
    with patch.object(tracker,"_track_motion",return_value=False),patch.object(tracker,"_track_reference") as reference:
        assert not tracker._track_localization(current,False)
        reference.assert_not_called()
    tracker.state = TrackingState.LOST
    with patch.object(tracker,"_relocalize",return_value=True) as relocalize:
        assert tracker._track_localization(current,False)
        relocalize.assert_called_once_with(current)


@pytest.mark.parametrize("sensor",[Sensor.MONOCULAR_IMU,Sensor.STEREO_IMU,Sensor.RGBD_IMU])
def test_original_inertial_localization_only_limitation_is_explicit(sensor):
    calibration = ImuCalibration(np.eye(4),.01,.05,.001,.005)
    vocabulary = VocabularyForTest()
    tracker = Tracker(Atlas(inertial=True),sensor,Calibration(Pinhole(400,400,320,240),imu=calibration),
                      vocabulary,InvertedDatabase(vocabulary))
    with pytest.raises(ValueError,match="inertial localization-only"):
        tracker.set_localization_mode(True)
    assert not tracker.only_tracking


def test_disabling_localization_mode_resumes_keyframe_decisions():
    tracker,current,_ = localization_scene(3,persistent=3)
    tracker.inliers = 20
    tracker.reference.tracked_points = lambda count: 100
    tracker.last_keyframe_frame_id = 0
    current.id = 60
    assert not tracker._needs_keyframe(current)
    tracker.set_localization_mode(False)
    assert tracker._needs_keyframe(current)


def integrated(calibration,dt=.1):
    preintegrated = Preintegrated(Bias(),calibration)
    for _ in range(5):
        preintegrated.integrate([0,0,9.81],[0,0,0],dt/5)
    return preintegrated


def inertial_tracker_scene():
    atlas,vocabulary = Atlas(inertial=True),VocabularyForTest()
    calibration = ImuCalibration(make_pose(translation=[.03,-.01,.02]),.01,.05,.001,.005)
    world,descriptors = scene(3)
    keyframes = []
    for index,position in enumerate(([0,0,0],[10,0,0],[100,0,0])):
        frame = frame_from_scene(index,world,descriptors,calibration=calibration)
        frame.body_state = BodyState(np.eye(3),np.array(position,float),np.array([index+1.,0,0]))
        frame.pose = camera_pose_from_body(frame.body_state,calibration)
        keyframes.append(atlas.add_keyframe(frame))
    tracker = Tracker(atlas,Sensor.RGBD_IMU,Calibration(Pinhole(400,400,320,240),imu=calibration),
                      vocabulary,InvertedDatabase(vocabulary))
    tracker.last_keyframe,tracker.reference = keyframes[-1],keyframes[-1]
    last = frame_from_scene(15,world,descriptors,calibration=calibration)
    last.last_keyframe_id,last.preintegrated = keyframes[0].id,integrated(calibration,.1)
    current = frame_from_scene(16,world,descriptors,calibration=calibration)
    current.last_keyframe_id,current.preintegrated = keyframes[1].id,integrated(calibration,.2)
    tracker.last,tracker.current = last,current
    return tracker,keyframes,calibration


def test_update_frame_imu_uses_each_frames_preceding_keyframe_and_explicit_bias():
    tracker,keyframes,calibration = inertial_tracker_scene()
    donor_map = tracker.atlas.create_map()
    donor_frame = copy.deepcopy(tracker.last)
    donor_reference = tracker.atlas.add_keyframe(donor_frame)
    tracker.atlas.change_map(keyframes[0].map.id)
    old_relative = make_pose(translation=[.2,0,0])
    donor_relative = make_pose(translation=[.3,0,0])
    tracker.history = [TrajectoryEntry(0,old_relative,keyframes[0],TrackingState.OK),
                       TrajectoryEntry(0,donor_relative,donor_reference,TrackingState.OK)]
    bias = Bias([.01,0,0],[0,0,.001])
    expected = []
    for frame,reference in ((tracker.last,keyframes[0]),(tracker.current,keyframes[1])):
        state = reference.body_state
        expected.append(predict_body_state(state.rotation,state.position,state.velocity,frame.preintegrated,bias))
    tracker.apply_inertial_corrections([(keyframes[0].map.id,2.,bias,keyframes[-1].id)])
    for frame,(rotation,position,velocity) in zip((tracker.last,tracker.current),expected):
        np.testing.assert_allclose(frame.body_state.rotation,rotation,atol=1e-12)
        np.testing.assert_allclose(frame.body_state.position,position,atol=1e-12)
        np.testing.assert_allclose(frame.body_state.velocity,velocity,atol=1e-12)
        np.testing.assert_array_equal(frame.body_state.bias.vector(),bias.vector())
        np.testing.assert_allclose(frame.pose,camera_pose_from_body(frame.body_state,calibration),atol=1e-12)
    assert tracker.last.body_state.position[0] < 1
    assert 10 < tracker.current.body_state.position[0] < 11
    assert old_relative[0,3] == pytest.approx(.4)
    assert donor_relative[0,3] == pytest.approx(.3)


def test_new_keyframe_does_not_relabel_its_incoming_preintegration_anchor():
    tracker,keyframes,_ = inertial_tracker_scene()
    frame = tracker.current
    previous = tracker.last_keyframe
    frame.last_keyframe_id = previous.id
    keyframe = tracker._create_keyframe(frame)
    assert keyframe.previous_id == previous.id
    assert frame.last_keyframe_id == keyframe.last_keyframe_id == previous.id
    assert tracker.last_keyframe is keyframe


def test_merge_correction_scales_only_premerge_current_history_and_keeps_temporal_anchor():
    tracker,keyframes,calibration = inertial_tracker_scene()
    old_map = tracker.atlas.current
    current_ids = set(old_map.keyframes)
    tracker.last_keyframe = keyframes[1]
    bias = Bias()
    # Correct only the source map keyframes, as MergeLocal2 has already done.
    angle = .2
    rotation = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    transform = Sim3(rotation,np.array([1.,2.,0]),1.5)
    for keyframe in keyframes:
        old_state = copy.deepcopy(keyframe.body_state)
        world_camera = inverse_pose(keyframe.pose)
        world_camera[:3,:3] = rotation @ world_camera[:3,:3]
        world_camera[:3,3] = transform.map(world_camera[:3,3])
        keyframe.pose = inverse_pose(world_camera)
        keyframe.body_state = body_state_from_camera(keyframe.pose,calibration,
                                                     1.5*rotation @ old_state.velocity,old_state.bias)
    donor_map = tracker.atlas.create_map()
    donor = tracker.atlas.add_keyframe(frame_from_scene(99,*scene(3),calibration=calibration))
    # Transfer is complete: donor shares current's map, so checking map.id here
    # would incorrectly scale donor history too.
    donor.map = old_map
    old_map.keyframes[donor.id] = donor
    tracker.atlas.change_map(old_map.id)
    current_relative,donor_relative = make_pose(translation=[.2,0,0]),make_pose(translation=[.3,0,0])
    tracker.history = [TrajectoryEntry(0,current_relative,keyframes[0],TrackingState.OK),
                       TrajectoryEntry(0,donor_relative,donor,TrackingState.OK)]
    expected = []
    for frame,reference in ((tracker.last,keyframes[0]),(tracker.current,keyframes[1])):
        base = reference.body_state
        expected.append(predict_body_state(base.rotation,base.position,base.velocity,frame.preintegrated,bias))
    tracker.apply_merge_correction({"scale":1.5,"world_transform":transform,"bias":bias},history_keyframe_ids=current_ids)
    assert tracker.last_keyframe is keyframes[1]
    assert current_relative[0,3] == pytest.approx(.3)
    assert donor_relative[0,3] == pytest.approx(.3)
    for frame,(r,p,v) in zip((tracker.last,tracker.current),expected):
        np.testing.assert_allclose(frame.body_state.rotation,r,atol=1e-12)
        np.testing.assert_allclose(frame.body_state.position,p,atol=1e-12)
        np.testing.assert_allclose(frame.body_state.velocity,v,atol=1e-12)


def test_frame_by_frame_visual_localization_freezes_map_geometry_and_resumes():
    from pathlib import Path
    from orbslam3.system import SLAM
    from orbslam3.vocabulary import Vocabulary
    from synthetic import TexturedPlanes
    world = TexturedPlanes()
    vocabulary = Vocabulary.from_text(Path(__file__).resolve().parents[1]/"assets/ORBvoc.txt")
    slam = SLAM(Sensor.RGBD,Calibration(world.camera,baseline_fx=world.camera.fx*.1,
                       depth_threshold=4.,frames_per_second=10/3.),vocabulary=vocabulary,
                nfeatures=750,nlevels=4)
    for index in range(2):
        result = slam.process(world.frame(Sensor.RGBD,index*.3))
        assert result.state == TrackingState.OK
    slam.set_localization_mode(True)
    map_ = slam.atlas.current
    poses = {identifier:keyframe.pose.copy() for identifier,keyframe in map_.keyframes.items()}
    points = {identifier:point.position.copy() for identifier,point in map_.points.items()}
    for index in range(2,6):
        result = slam.process(world.frame(Sensor.RGBD,index*.3))
        assert result.state == TrackingState.OK and result.inliers >= 30
        assert result.diagnostics["localization_only"]
        assert not result.diagnostics["mapping"]
    assert set(map_.keyframes) == set(poses) and set(map_.points) == set(points)
    for identifier,pose in poses.items():
        np.testing.assert_array_equal(map_.keyframes[identifier].pose,pose)
    for identifier,position in points.items():
        np.testing.assert_array_equal(map_.points[identifier].position,position)
    assert not slam.mapper.accepts_keyframes
    slam.set_localization_mode(False)
    assert slam.mapper.accepts_keyframes
    result = slam.process(world.frame(Sensor.RGBD,1.8))
    assert result.state == TrackingState.OK and not result.diagnostics["localization_only"]
