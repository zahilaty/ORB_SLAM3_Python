import numpy as np
import pytest
from orbslam3.camera import Pinhole
from orbslam3.features import FeatureSet
from orbslam3.input import Sensor
from orbslam3.map import Atlas, Frame
from orbslam3.mapping import LocalMapper, reprojection_observations, select_visual_local_window


class VocabularyForSyntheticTest:
    def transform(self, descriptors, levelsup=4):
        return {0:1.},{0:list(range(len(descriptors)))}


def add_keyframe(atlas, points, descriptors, translation=(0,0,0), timestamp=0.,
                 right_camera=None, n_left=-1, T_right_left=None):
    points = np.asarray(points,np.float32).reshape(-1,2)
    n = len(points)
    features = FeatureSet(points,np.asarray(descriptors,np.uint8).reshape(n,32),np.zeros(n,np.int32),
                          np.zeros(n,np.float32),np.ones(n,np.float32),np.full(n,31,np.float32),n)
    pose = np.eye(4)
    pose[:3,3] = translation
    frame = Frame(atlas.next_keyframe_id,timestamp,features,Pinhole(500,500,320,240),640,480,
                  np.array([1.,1.2,1.44]),1/np.array([1.,1.44,2.0736]),baseline_fx=50.,
                  depth_threshold=10.,pose=pose,right_camera=right_camera,
                  n_left=n_left,T_right_left=T_right_left)
    keyframe = atlas.add_keyframe(frame)
    keyframe.bow,keyframe.feature_nodes = VocabularyForSyntheticTest().transform(features.descriptors)
    return keyframe


def test_recent_point_culling_uses_sensor_observation_threshold_and_age():
    for sensor,threshold in [(Sensor.MONOCULAR,2),(Sensor.RGBD,3)]:
        atlas = Atlas()
        keyframe = add_keyframe(atlas,[],np.empty((0,32),np.uint8))
        mapper = LocalMapper(atlas,sensor,VocabularyForSyntheticTest())
        poor = atlas.add_point([0,0,5],keyframe)
        poor.visible,poor.found = 5,1
        young = atlas.add_point([1,0,5],keyframe)
        young.observation_count = threshold
        good = atlas.add_point([2,0,5],keyframe)
        good.observation_count = threshold+1
        mapper.recent_points = [poor,young,good]
        keyframe.id = 1
        assert mapper.cull_recent_points(keyframe) == 1
        assert poor.bad and not young.bad
        keyframe.id = 2
        assert mapper.cull_recent_points(keyframe) == 1
        assert young.bad and not good.bad
        keyframe.id = 3
        assert mapper.cull_recent_points(keyframe) == 0
        assert not mapper.recent_points and not good.bad


def test_local_mapping_triangulates_source_epipolar_matches():
    atlas = Atlas()
    camera = Pinhole(500,500,320,240)
    world = np.array([[-.5,-.2,5.],[.2,.3,4.],[.7,-.3,6.]])
    descriptors = np.random.default_rng(3).integers(0,256,(3,32),dtype=np.uint8)
    neighbor = add_keyframe(atlas,camera.project(world+[-.2,0,0]),descriptors,(-.2,0,0))
    current = add_keyframe(atlas,camera.project(world),descriptors)
    current.add_connection(neighbor,3)
    mapper = LocalMapper(atlas,Sensor.STEREO,VocabularyForSyntheticTest())
    assert mapper.create_new_points(current) == 3
    assert len(atlas.current.points) == 3
    for expected,point in zip(world,current.map_points):
        np.testing.assert_allclose(point.position,expected,atol=1e-5)
        assert set(point.observations) == {current.id,neighbor.id}
        assert point.descriptor is not None
        assert point.max_distance > point.min_distance > 0


def test_triangulation_rejects_short_baseline_and_far_points():
    atlas = Atlas()
    descriptors = np.zeros((1,32),np.uint8)
    neighbor = add_keyframe(atlas,[[310,240]],descriptors,(-.05,0,0))
    current = add_keyframe(atlas,[[320,240]],descriptors)
    current.add_connection(neighbor,1)
    mapper = LocalMapper(atlas,Sensor.STEREO,VocabularyForSyntheticTest())
    assert mapper.create_new_points(current) == 0  # baseline .05 < calibrated .1
    neighbor.pose[0,3] = -.2
    neighbor.features.points[0] = [300,240]
    mapper.far_points,mapper.far_threshold = True,4.
    assert mapper.create_new_points(current) == 0


def test_observation_builder_keeps_stereo_and_rigid_right_factors():
    atlas = Atlas()
    descriptors = np.zeros((1,32),np.uint8)
    stereo = add_keyframe(atlas,[[320,240]],descriptors)
    point = atlas.add_point([0,0,5],stereo)
    point.add_observation(stereo,0)
    stereo.map_points[0] = point
    stereo.right_u[0] = 310.
    observations = reprojection_observations([stereo])
    assert len(observations) == 1
    np.testing.assert_array_equal(observations[0].measurement,[320,240,310])
    right_pose = np.eye(4)
    right_pose[0,3] = -.1
    rig = add_keyframe(atlas,[[320,240],[310,240]],np.zeros((2,32),np.uint8),
                       right_camera=stereo.camera,n_left=1,T_right_left=right_pose)
    for index in (0,1):
        point.add_observation(rig,index)
        rig.map_points[index] = point
    observations = reprojection_observations([rig])
    assert len(observations) == 2
    assert observations[0].camera_from_reference is None
    np.testing.assert_array_equal(observations[1].camera_from_reference,right_pose)
    assert reprojection_observations([rig],point_filter={999}) == []


def test_visual_window_selects_observing_fixed_cameras_and_skips_unanchored():
    atlas = Atlas()
    keyframes = [add_keyframe(atlas,[[320,240]],np.zeros((1,32),np.uint8)) for _ in range(3)]
    point = atlas.add_point([0,0,5],keyframes[1])
    for keyframe in keyframes:
        point.add_observation(keyframe,0)
        keyframe.map_points[0] = point
    current = keyframes[2]
    current.add_connection(keyframes[1],1)
    local,fixed,point_ids,included = select_visual_local_window(current)
    assert local == {1,2} and fixed == {0} and point_ids == {point.id}
    assert included == {0,1,2}
    point.erase_observation(keyframes[0])  # this also culls a weak point
    mapper = LocalMapper(atlas,Sensor.MONOCULAR,VocabularyForSyntheticTest())
    assert mapper.local_visual_ba(current)["skipped"] == "no fixed keyframe"


def test_keyframe_culling_needs_four_other_similar_scale_observers():
    def build(count):
        atlas = Atlas()
        keyframes = [add_keyframe(atlas,[[320,240]],np.zeros((1,32),np.uint8)) for _ in range(count)]
        point = atlas.add_point([0,0,5],keyframes[1])
        for keyframe in keyframes:
            keyframe.map_points[0] = point
            point.add_observation(keyframe,0)
        current = keyframes[-1]
        current.add_connection(keyframes[1],1)
        keyframes[1].change_parent(keyframes[0])
        return atlas,current,keyframes[1]
    atlas,current,redundant = build(4)
    mapper = LocalMapper(atlas,Sensor.MONOCULAR,VocabularyForSyntheticTest())
    assert mapper.cull_keyframes(current) == 0  # only three OTHER observers
    assert not redundant.bad
    atlas,current,redundant = build(5)
    mapper = LocalMapper(atlas,Sensor.MONOCULAR,VocabularyForSyntheticTest())
    assert mapper.cull_keyframes(current) == 1
    assert redundant.bad and redundant.id not in atlas.current.keyframes


def test_primary_local_ba_keeps_initial_pose_and_reduces_reprojection_cost():
    atlas = Atlas()
    camera = Pinhole(500,500,320,240)
    rng = np.random.default_rng(5)
    world = rng.uniform([-.8,-.6,4.],[.8,.6,7.],(20,3))
    descriptors = rng.integers(0,256,(20,32),dtype=np.uint8)
    keyframes = []
    for index,translation in enumerate(([0,0,0],[-.2,0,0],[-.4,0,0])):
        pixel = camera.project(world+translation)
        keyframes.append(add_keyframe(atlas,pixel,descriptors,translation))
    for index,position in enumerate(world):
        point = atlas.add_point(position+rng.normal(0,.02,3),keyframes[0])
        for keyframe in keyframes:
            point.add_observation(keyframe,index)
            keyframe.map_points[index] = point
        point.update_normal_and_depth()
    current = keyframes[-1]
    current.add_connection(keyframes[0],20)
    current.add_connection(keyframes[1],20)
    keyframes[1].pose[:3,3] += [.01,.005,-.01]
    fixed = keyframes[0].pose.copy()
    mapper = LocalMapper(atlas,Sensor.MONOCULAR,VocabularyForSyntheticTest())
    diagnostics = mapper.local_visual_ba(current)
    assert diagnostics["final_cost"] < diagnostics["initial_cost"]*.001
    np.testing.assert_array_equal(keyframes[0].pose,fixed)
    assert diagnostics["fixed_keyframes"] == 1
    assert atlas.current.change_index == 1


def test_visual_inertial_local_ba_commits_body_and_camera_states_consistently():
    from orbslam3.imu import ImuCalibration, Bias, Preintegrated
    from orbslam3.inertial_optimization import BodyState
    atlas = Atlas(inertial=True)
    camera = Pinhole(500,500,320,240)
    calibration = ImuCalibration(np.eye(4),.01,.05,.001,.005)
    rng = np.random.default_rng(14)
    world = rng.uniform([-.6,-.4,4.],[.6,.4,6.],(12,3))
    descriptors = rng.integers(0,256,(12,32),dtype=np.uint8)
    keyframes = []
    for identifier in range(4):
        x = .1*identifier
        keyframe = add_keyframe(atlas,camera.project(world+[-x,0,0]),descriptors,
                               translation=(-x,0,0),timestamp=.1*identifier)
        keyframe.body_state = BodyState(np.eye(3),np.array([x,0,0]),np.array([1.,0,0]))
        keyframe.imu_calibration = calibration
        keyframe.right_u = keyframe.features.points[:,0]-50/world[:,2]
        keyframe.depth = world[:,2].copy()
        if identifier:
            keyframe.previous_id = identifier-1
            keyframes[-1].next_id = identifier
            keyframe.preintegrated = Preintegrated(Bias(),calibration)
            for _ in range(5):
                keyframe.preintegrated.integrate(np.array([0.,0,9.81]),np.zeros(3),.02)
        keyframes.append(keyframe)
    for index,position in enumerate(world):
        point = atlas.add_point(position+rng.normal(0,.005,3),keyframes[0])
        for keyframe in keyframes:
            point.add_observation(keyframe,index)
            keyframe.map_points[index] = point
        point.update_normal_and_depth()
    atlas.current.imu_initialized = True
    current = keyframes[-1]
    current.body_state.position += [.005,-.003,.002]
    current.pose[:3,3] = -current.body_state.position
    mapper = LocalMapper(atlas,Sensor.STEREO_IMU,VocabularyForSyntheticTest())
    diagnostics = mapper.local_inertial_ba(current)
    assert diagnostics["accepted"]
    assert diagnostics["final_cost"] < diagnostics["initial_cost"]*.01
    for keyframe in keyframes:
        np.testing.assert_allclose(keyframe.pose[:3,:3],keyframe.body_state.rotation.T,atol=1e-12)
        np.testing.assert_allclose(keyframe.pose[:3,3],-keyframe.body_state.rotation.T @ keyframe.body_state.position,atol=1e-12)
    assert atlas.current.change_index == 1


def test_inertial_keyframe_culling_merges_temporal_preintegrations():
    from orbslam3.imu import ImuCalibration, Bias, Preintegrated
    from orbslam3.inertial_optimization import BodyState
    atlas = Atlas(inertial=True)
    calibration = ImuCalibration(np.eye(4),.01,.05,.001,.005)
    keyframes = []
    for identifier in range(22):
        keyframe = add_keyframe(atlas,[[320,240]],np.zeros((1,32),np.uint8),timestamp=.1*identifier)
        keyframe.depth[:] = 5.
        keyframe.body_state = BodyState(np.eye(3),np.zeros(3),np.zeros(3))
        keyframe.imu_calibration = calibration
        if identifier:
            keyframe.previous_id = identifier-1
            keyframes[-1].next_id = identifier
            keyframe.preintegrated = Preintegrated(Bias(),calibration)
            keyframe.preintegrated.integrate([0,0,9.81],[0,0,0],.1)
        keyframes.append(keyframe)
    p = atlas.add_point([0,0,5],keyframes[0])
    for keyframe in keyframes:
        p.add_observation(keyframe,0)
        keyframe.map_points[0] = p
    current = keyframes[-1]
    current.add_connection(keyframes[1],1)
    atlas.current.imu_initialized = True
    mapper = LocalMapper(atlas,Sensor.STEREO_IMU,VocabularyForSyntheticTest())
    assert mapper.cull_keyframes(current) == 1
    assert keyframes[1].bad
    assert keyframes[2].previous_id == 0 and keyframes[0].next_id == 2
    assert keyframes[2].preintegrated.dt == pytest.approx(.2)


def test_sequential_process_defers_fusion_and_ba_if_keyframes_are_pending():
    atlas = Atlas()
    keyframe = add_keyframe(atlas,[],np.empty((0,32),np.uint8))
    mapper = LocalMapper(atlas,Sensor.MONOCULAR,VocabularyForSyntheticTest())
    mapper.pending_keyframes = 1
    diagnostics = mapper.process(keyframe)
    assert diagnostics["local_ba"] is None
    assert diagnostics["new_points"] == diagnostics["culled_points"] == 0
    assert mapper.accepts_keyframes


def test_preinitialization_visual_ba_keeps_cached_imu_body_pose_coherent():
    from orbslam3.imu import ImuCalibration, Bias
    from orbslam3.inertial_mapping import body_state_from_camera,camera_pose_from_body
    atlas = Atlas(inertial=True)
    camera = Pinhole(500,500,320,240)
    extrinsic = np.eye(4)
    extrinsic[:3,3] = [.03,-.01,.02]
    calibration = ImuCalibration(extrinsic,.01,.05,.001,.005)
    rng = np.random.default_rng(85)
    world = rng.uniform([-.8,-.5,4.],[.8,.5,6.],(20,3))
    descriptors = rng.integers(0,256,(20,32),dtype=np.uint8)
    keyframes = []
    for translation in ([0,0,0],[-.2,0,0],[-.4,0,0]):
        keyframes.append(add_keyframe(atlas,camera.project(world+translation),descriptors,translation))
    keyframes[1].pose[:3,3] += [.02,.01,-.01]
    for keyframe in keyframes:
        keyframe.imu_calibration = calibration
        keyframe.body_state = body_state_from_camera(keyframe.pose,calibration,
                          velocity=np.array([.5,.1,-.2]),bias=Bias([.01,.02,.03],[.001,.002,.003]))
    for index,position in enumerate(world):
        p = atlas.add_point(position,keyframes[0])
        for keyframe in keyframes:
            p.add_observation(keyframe,index)
            keyframe.map_points[index] = p
        p.update_normal_and_depth()
    current = keyframes[-1]
    current.add_connection(keyframes[0],20)
    current.add_connection(keyframes[1],20)
    before = keyframes[1].body_state.position.copy()
    mapper = LocalMapper(atlas,Sensor.STEREO_IMU,VocabularyForSyntheticTest())
    result = mapper.local_visual_ba(current)
    assert result["final_cost"] < result["initial_cost"]*.001
    assert np.linalg.norm(keyframes[1].body_state.position-before) > .001
    for keyframe in keyframes:
        np.testing.assert_allclose(camera_pose_from_body(keyframe.body_state,calibration),keyframe.pose,atol=1e-12)
        np.testing.assert_array_equal(keyframe.body_state.velocity,[.5,.1,-.2])
        np.testing.assert_array_equal(keyframe.body_state.bias.accelerometer,[.01,.02,.03])
        np.testing.assert_array_equal(keyframe.body_state.bias.gyroscope,[.001,.002,.003])
