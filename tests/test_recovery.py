"""Actual BoW/MLPnP/pose/projection recovery and Atlas failure transitions.

Fixtures provide calibrated observations of a synthetic nonplanar map. The
trained ORB vocabulary and all matching, geometric and optimization routines
execute normally; no tracking or optimizer algorithm is mocked.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from orbslam3.camera import Pinhole
from orbslam3.features import FeatureSet, hamming
from orbslam3.geometry import make_pose,inverse_pose
from orbslam3.input import Sensor,Calibration
from orbslam3.imu import Bias,ImuCalibration,ImuMeasurement,Preintegrated
from orbslam3.inertial_mapping import body_state_from_camera
from orbslam3.map import Atlas,Frame
from orbslam3.matching import search_by_bow_frame
from orbslam3.tracking import Tracker,TrackingState,TrajectoryEntry
from orbslam3.vocabulary import Vocabulary,InvertedDatabase


@pytest.fixture(scope="module")
def trained_vocabulary():
    return Vocabulary.from_text(Path(__file__).resolve().parents[1]/"assets/ORBvoc.txt")


def observation_frame(identifier,world,descriptors,pose=None,timestamp=None):
    camera = Pinhole(400.,400.,320.,240.)
    pose = np.eye(4) if pose is None else pose
    local = world @ pose[:3,:3].T+pose[:3,3]
    n = len(world)
    features = FeatureSet(camera.project(local).astype(np.float32),descriptors.copy(),np.zeros(n,np.int32),
                          np.zeros(n,np.float32),np.ones(n,np.float32),np.full(n,31,np.float32),n)
    return Frame(identifier,identifier/30. if timestamp is None else timestamp,features,camera,640,480,
                 np.array([1.,1.2,1.44,1.728]),1/np.array([1.,1.44,2.0736,2.985984]),pose=pose.copy())


def existing_map(vocabulary,keyframe_count=11,point_count=200,sensor=Sensor.MONOCULAR,
                 insert_keyframes_when_lost=True):
    rng = np.random.default_rng(814)
    world = rng.uniform([-1.2,-.8,4.],[1.2,.8,7.],(point_count,3))
    descriptors = rng.integers(0,256,(point_count,32),dtype=np.uint8)
    atlas,database = Atlas(inertial=sensor.inertial),InvertedDatabase(vocabulary)
    imu_calibration = ImuCalibration(np.eye(4),.01,.05,.001,.005) if sensor.inertial else None
    points,keyframes = [],[]
    # Connections are established chronologically. This matches the source's
    # first-connection parent choice and prevents a synthetic spanning cycle.
    for identifier in range(keyframe_count):
        pose = make_pose(translation=[-.02*identifier,0,0])
        frame = observation_frame(identifier,world,descriptors,pose)
        if sensor.inertial:
            frame.imu_calibration = imu_calibration
            frame.body_state = body_state_from_camera(pose,imu_calibration,velocity=np.array([.6,0,0]))
        keyframe = atlas.add_keyframe(frame)
        if keyframes:
            keyframe.previous_id,keyframes[-1].next_id = keyframes[-1].id,keyframe.id
        keyframe.compute_bow(vocabulary)
        if not points:
            points = [atlas.add_point(position,keyframe) for position in world]
        for index,point in enumerate(points):
            keyframe.map_points[index] = point
            point.add_observation(keyframe,index)
        keyframe.update_connections()
        keyframes.append(keyframe)
        database.add(keyframe.id,keyframe.bow,keyframe.map.id,keyframe.ordered_covisibility)
    for point in points:
        point.compute_descriptor()
        point.update_normal_and_depth()
    for keyframe in keyframes:
        database.entries[keyframe.id].neighbors = keyframe.ordered_covisibility.copy()
    tracker = Tracker(atlas,sensor,Calibration(keyframes[0].camera,imu=imu_calibration),vocabulary,database,
                      insert_keyframes_when_lost=insert_keyframes_when_lost)
    tracker.reference = tracker.last_keyframe = keyframes[-1]
    tracker.last_keyframe_frame_id = keyframes[-1].source_frame_id
    tracker.last = observation_frame(100,world,descriptors,keyframes[-1].pose)
    tracker.last.map_points = points.copy()
    tracker.last.reference_id = keyframes[-1].id
    tracker.last.last_keyframe_id = keyframes[-1].id
    if sensor.inertial:
        tracker.last.imu_calibration = imu_calibration
        tracker.last.body_state = body_state_from_camera(tracker.last.pose,imu_calibration,
                                                         velocity=np.array([.6,0,0]))
    tracker.history = [TrajectoryEntry(tracker.last.timestamp,np.eye(4),keyframes[-1],TrackingState.OK)]
    tracker.state = TrackingState.OK
    tracker.last_map_change = atlas.current.change_index
    tracker.local_keyframes,tracker.local_points = keyframes.copy(),points.copy()
    return tracker,world,descriptors,keyframes,points


def empty_frame(identifier,timestamp=None):
    frame = observation_frame(identifier,np.empty((0,3)),np.empty((0,32),np.uint8),timestamp=timestamp)
    frame.pose = None
    return frame


def partial_bow_descriptors(descriptors,exact_count=25):
    result = descriptors.copy()
    rng = np.random.default_rng(196)
    for index in range(exact_count,len(result)):
        bits = np.zeros(256,np.uint8)
        bits[rng.choice(256,size=70,replace=False)] = 1
        result[index] ^= np.packbits(bits,bitorder="little")
    np.testing.assert_array_equal(hamming(result[exact_count:],descriptors[exact_count:]),70)
    return result


def assert_pose(actual,expected,tolerance=2e-5):
    np.testing.assert_allclose(inverse_pose(actual)[:3,3],inverse_pose(expected)[:3,3],atol=tolerance)
    angle = np.linalg.norm(Rotation.from_matrix(actual[:3,:3] @ expected[:3,:3].T).as_rotvec())
    assert angle < tolerance


def test_real_relocalization_uses_pnp_pose_stages_and_projection_expansion(trained_vocabulary):
    tracker,world,descriptors,keyframes,points = existing_map(trained_vocabulary)
    actual_pose = make_pose(Rotation.from_rotvec([.008,-.012,.006]).as_matrix(),[-.27,.025,-.015])
    frame = observation_frame(101,world,partial_bow_descriptors(descriptors),actual_pose)
    frame.pose = None
    frame.compute_bow(trained_vocabulary)
    initial = search_by_bow_frame(keyframes[-1],frame,nnratio=.75)
    # Only the first25 descriptors can pass BoW; trained zero-weight words
    # stop some of those too. Fewer than50 correspondences cannot satisfy
    # the source relocalization gate without projection expansion.
    assert 15 <= sum(point is not None for point in initial) <= 25
    assert tracker._relocalize(frame)
    assert tracker.last_reloc_id == frame.id
    recovered = sum(point is not None and not frame.outliers[index]
                    for index,point in enumerate(frame.map_points))
    assert recovered >= 190
    # The remaining descriptors have distance70: they cannot have passed
    # SearchByBoW'sTH_LOW=50, but the real100-bit projection expansion can.
    assert sum(point is not None for point in frame.map_points[25:]) >= 165
    assert_pose(frame.pose,actual_pose)
    assert len(tracker.diagnostics["pose_chi2"]) >= 190


def test_temporary_loss_relocalizes_real_observations_without_replacing_map(trained_vocabulary):
    tracker,world,descriptors,keyframes,points = existing_map(trained_vocabulary)
    old_map = tracker.atlas.current
    lost = tracker.track(empty_frame(101))
    assert lost.state == TrackingState.RECENTLY_LOST
    assert tracker.lost_timestamp == pytest.approx(101/30.)
    assert tracker.atlas.current is old_map and len(tracker.atlas.maps) == 1
    expected = make_pose(Rotation.from_rotvec([.005,-.01,.004]).as_matrix(),[-.24,.015,.01])
    frame = observation_frame(102,world,partial_bow_descriptors(descriptors),expected)
    frame.pose = None
    recovered = tracker.track(frame)
    assert recovered.state == TrackingState.OK and recovered.inliers >= 190
    assert recovered.map_id == old_map.id and tracker.atlas.current is old_map
    assert tracker.last_reloc_id == frame.id
    assert not tracker.pending_keyframes  # recent relocalization suppresses insertion
    assert len(old_map.points) == len(points) and len(old_map.keyframes) == len(keyframes)
    assert_pose(recovered.pose,expected)
    assert tracker.history[-1].timestamp == frame.timestamp
    assert_pose(tracker.history[-1].pose(tracker.atlas.all_keyframes),expected)


def test_persistent_visual_loss_retains_old_map_and_starts_fresh_initialization(trained_vocabulary):
    tracker,world,descriptors,keyframes,points = existing_map(trained_vocabulary)
    old_map = tracker.atlas.current
    first = tracker.track(empty_frame(101))
    assert first.state == TrackingState.RECENTLY_LOST
    result = tracker.track(empty_frame(102,timestamp=tracker.lost_timestamp+3.01))
    assert result.state == TrackingState.NO_IMAGES_YET
    assert tracker.atlas.current is not old_map
    assert tracker.atlas.maps[old_map.id] is old_map and not old_map.bad
    assert len(tracker.atlas.maps) == 2
    assert not tracker.atlas.current.points and not tracker.atlas.current.keyframes
    assert tracker.last is tracker.last_keyframe is tracker.reference is None
    assert not tracker.pending_keyframes
    assert len(tracker.database.entries) == len(keyframes)
    # These old words remain available for Atlas map merging; relocalization
    # itself filters them out when asked for the fresh active map.
    bow,_ = trained_vocabulary.transform(descriptors)
    assert tracker.database.relocalization_candidates(bow,tracker.atlas.current.id) == []
    frame = observation_frame(103,world,descriptors,timestamp=6.5)
    frame.pose = None
    next_result = tracker.track(frame)
    assert next_result.state == TrackingState.NOT_INITIALIZED
    assert tracker.initial is frame and not tracker.atlas.current.keyframes


@pytest.mark.parametrize("keyframe_count",[1,10])
def test_small_map_tracking_failure_resets_map_and_clears_retrieval_index(trained_vocabulary,keyframe_count):
    tracker,_,_,keyframes,points = existing_map(trained_vocabulary,keyframe_count)
    old_map = tracker.atlas.current
    result = tracker.track(empty_frame(101))
    assert result.state == TrackingState.NO_IMAGES_YET
    assert old_map.bad and tracker.atlas.bad_maps[old_map.id] is old_map
    assert old_map.id not in tracker.atlas.maps and len(tracker.atlas.maps) == 1
    assert result.map_id != old_map.id
    assert not tracker.atlas.current.points and not tracker.atlas.current.keyframes
    assert not tracker.database.entries and not tracker.database.inverted
    assert tracker.last is None and not tracker.pending_keyframes


def test_already_lost_state_retains_exact_ten_keyframe_boundary(trained_vocabulary):
    tracker,_,_,_,_ = existing_map(trained_vocabulary,10)
    old_map = tracker.atlas.current
    tracker.state = TrackingState.LOST
    result = tracker.track(empty_frame(101))
    # Source has two distinct tests: the already-LOST branch uses '<10',
    # whereas a fresh tracking failure near the end of Track uses '<=10'.
    assert result.state == TrackingState.NO_IMAGES_YET
    assert old_map.id in tracker.atlas.maps and not old_map.bad
    assert len(tracker.atlas.maps) == 2 and len(tracker.database.entries) == 10


@pytest.mark.parametrize("sensor",[Sensor.MONOCULAR_IMU,Sensor.STEREO_IMU,Sensor.RGBD_IMU])
@pytest.mark.parametrize("initialized,ba2,preserve",[(False,False,False),(True,False,False),(True,True,True)])
def test_inertial_timestamp_jump_resets_before_ba2_but_preserves_mature_map(
        trained_vocabulary,sensor,initialized,ba2,preserve):
    tracker,_,_,_,_ = existing_map(trained_vocabulary,sensor=sensor)
    old_map = tracker.atlas.current
    old_map.imu_initialized,old_map.inertial_ba2 = initialized,ba2
    result = tracker.track(empty_frame(101,timestamp=tracker.last.timestamp+1.001))
    assert result.state == TrackingState.NO_IMAGES_YET
    assert result.diagnostics["reset"] == "inertial timestamp jump"
    assert tracker.atlas.current is not old_map and tracker.last is None
    assert not tracker.atlas.current.points and not tracker.atlas.current.keyframes
    if preserve:
        assert tracker.atlas.maps[old_map.id] is old_map and not old_map.bad
        assert len(tracker.atlas.maps) == 2 and len(tracker.database.entries) == 11
    else:
        assert old_map.bad and old_map.id in tracker.atlas.bad_maps
        assert len(tracker.atlas.maps) == 1 and not tracker.database.entries


def test_initialized_inertial_prediction_survives_visual_loss_until_original_five_second_timeout(trained_vocabulary):
    tracker,_,_,keyframes,_ = existing_map(trained_vocabulary,sensor=Sensor.MONOCULAR_IMU,
                                           insert_keyframes_when_lost=False)
    old_map = tracker.atlas.current
    old_map.imu_initialized = old_map.inertial_ba1 = old_map.inertial_ba2 = True
    for keyframe in keyframes:
        keyframe.inertial = True
    tracker.last.timestamp = .5
    tracker.history[-1].timestamp = .5
    for index in range(12):
        timestamp = 1.+index*.5
        begin = timestamp-.5
        measurements = tuple(ImuMeasurement(i/200.,[0,0,9.81],[0,0,0])
                             for i in range(int(round(begin*200)),int(round(timestamp*200))+1))
        result = tracker.track(empty_frame(101+index,timestamp=timestamp),measurements)
        if index <= 10:  # original timeout is strict '>5', not '>=5'
            assert result.state == TrackingState.RECENTLY_LOST
            assert result.pose is not None and np.isfinite(result.pose).all()
            # With zero angular rate and only gravity in the accelerometer,
            # the known .6 m/s camera motion must continue without images.
            np.testing.assert_allclose(inverse_pose(result.pose)[:3,3],
                                       [.2+.6*(timestamp-.5),0.,0.],atol=1e-7)
            assert tracker.atlas.current is old_map
            assert tracker.current.frame_preintegrated.dt == pytest.approx(.5,abs=1e-12)
            assert not tracker.pending_keyframes
        else:
            assert result.state == TrackingState.NO_IMAGES_YET
            assert tracker.atlas.current is not old_map
            assert tracker.atlas.maps[old_map.id] is old_map and not old_map.bad
            assert tracker.last is None and tracker.accumulated_imu.dt == 0
