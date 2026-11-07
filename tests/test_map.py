import numpy as np
import pytest
from orbslam3.camera import Pinhole
from orbslam3.features import FeatureSet
from orbslam3.geometry import inverse_pose, make_pose, transform_points
from orbslam3.imu import ImuCalibration
from orbslam3.inertial_optimization import BodyState
from orbslam3.map import Frame, Atlas


def make_frame(points,octaves=None,**kwargs):
    points = np.asarray(points,np.float32).reshape(-1,2)
    n = len(points)
    features = FeatureSet(points,np.zeros((n,32),np.uint8),
                          np.zeros(n,np.int32) if octaves is None else np.asarray(octaves,np.int32),
                          np.zeros(n,np.float32),np.ones(n,np.float32),np.full(n,31,np.float32),n)
    arguments = dict(id=0,timestamp=0.,features=features,camera=Pinhole(500,500,320,240),
                     width=640,height=480,scales=np.array([1.,1.2,1.44]),
                     inverse_variances=np.array([1.,1/1.44,1/2.0736]),pose=np.eye(4))
    arguments.update(kwargs)
    return Frame(**arguments)


def test_feature_query_uses_strict_square_bounds_and_octave_limits():
    frame = make_frame([[320,240],[324,244],[325,240],[320,235],[320,240]],octaves=[0,1,0,0,2])
    assert set(frame.get_features_in_area(320,240,5)) == {0,1,4}
    assert list(frame.get_features_in_area(320,240,5,1,1)) == [1]
    assert list(frame.get_features_in_area(320,240,5,0,0)) == [0]


def test_grid_rounds_half_cells_away_from_zero_before_grid_ordering():
    frame = make_frame([[25,20],[24,40],[25,10],[-5,1]])
    assert list(frame.get_features_in_area(25,25,25)) == [1,2,0]
    assert list(frame.get_features_in_area(0,0,20)) == []  # -0.5 grid x rounds to -1


def test_dual_query_returns_local_right_indices_and_camera_geometry():
    relative = make_pose(translation=[-.1,0,0])
    frame = make_frame([[320,240],[320,240],[310,240]],n_left=1,
                       right_camera=Pinhole(500,500,320,240),T_right_left=relative)
    assert list(frame.get_features_in_area(320,240,2)) == [0]
    assert list(frame.get_features_in_area(320,240,2,right=True)) == [0]
    assert frame.camera_for(1) is frame.right_camera
    np.testing.assert_array_equal(frame.pose_for(1),relative)
    np.testing.assert_array_equal(frame.center_for(1),[.1,0,0])


def test_depth_unprojection_respects_world_to_camera_convention():
    pose = make_pose(translation=[-.2,0,0])
    frame = make_frame([[320,240]],pose=pose,depth=np.array([5.]))
    np.testing.assert_allclose(frame.unproject_depth(0),[.2,0,5])
    frame.depth[0] = -1
    assert frame.unproject_depth(0) is None


def test_atlas_copies_mutable_frame_state_and_retains_culled_keyframes():
    atlas = Atlas(inertial=True)
    calibration = ImuCalibration(np.eye(4),.01,.05,.001,.005)
    frame = make_frame([[320,240]],body_state=BodyState(np.eye(3),np.zeros(3),np.ones(3)),
                       imu_calibration=calibration)
    initial = atlas.add_keyframe(frame)
    second = atlas.add_keyframe(frame)
    frame.pose[0,3] = 1
    frame.outliers[0] = True
    frame.body_state.position[0] = 2
    assert initial.pose[0,3] == 0 and not initial.outliers[0]
    assert initial.body_state.position[0] == 0
    second.change_parent(initial)
    second.set_bad()
    assert second.id not in atlas.current.keyframes
    assert atlas.all_keyframes[second.id] is second
    np.testing.assert_array_equal(second.pose_to_parent,np.eye(4))


def test_observation_count_distinguishes_rectified_depth_and_rigid_rig():
    atlas = Atlas()
    depth_keyframe = atlas.add_keyframe(make_frame([[320,240]],right_u=np.array([310.])))
    p = atlas.add_point([0,0,5],depth_keyframe)
    p.add_observation(depth_keyframe,0)
    assert p.observation_count == 2
    rig = atlas.add_keyframe(make_frame([[320,240],[310,240]],n_left=1,
                              right_camera=depth_keyframe.camera,T_right_left=make_pose(translation=[-.1,0,0])))
    p.add_observation(rig,0)
    p.add_observation(rig,1)
    assert p.observations[rig.id] == (0,1)
    assert p.observation_count == 4
    depth_keyframe.map_points[0] = p
    rig.map_points[:] = [p,p]
    p.erase_observation(depth_keyframe)
    assert p.bad and p.id not in atlas.current.points
    assert rig.map_points == [None,None]


def test_replacement_transfers_both_rig_observations_and_counts():
    atlas = Atlas()
    keyframe = atlas.add_keyframe(make_frame([[320,240],[310,240]],n_left=1,
                        right_camera=Pinhole(500,500,320,240),T_right_left=make_pose(translation=[-.1,0,0])))
    old,new = atlas.add_point([0,0,5],keyframe),atlas.add_point([0,0,5],keyframe)
    for index in (0,1):
        old.add_observation(keyframe,index)
        keyframe.map_points[index] = old
    old.found,old.visible = 2,4
    old.replace(new)
    assert old.bad and old.replaced is new
    assert keyframe.map_points == [new,new]
    assert new.observations[keyframe.id] == (0,1) and new.observation_count == 2
    assert new.found == 3 and new.visible == 5


def test_normal_and_distance_bounds_include_both_rig_centers():
    atlas = Atlas()
    keyframe = atlas.add_keyframe(make_frame([[320,240],[310,240]],octaves=[1,0],n_left=1,
                         right_camera=Pinhole(500,500,320,240),T_right_left=make_pose(translation=[-.1,0,0])))
    p = atlas.add_point([0,0,5],keyframe)
    for index in (0,1):
        p.add_observation(keyframe,index)
        keyframe.map_points[index] = p
    p.update_normal_and_depth()
    right_direction = np.array([-.1,0,5.])/np.linalg.norm([-.1,0,5.])
    np.testing.assert_allclose(p.normal,(np.array([0,0,1.])+right_direction)/2)
    assert p.max_distance == pytest.approx(6)
    assert p.min_distance == pytest.approx(6/1.44)
    assert p.predict_scale(5,keyframe.scales) == 1


def test_covisibility_parent_is_strongest_neighbor_and_depth_point_counts_once():
    atlas = Atlas()
    keyframes = [atlas.add_keyframe(make_frame([[320,240]])) for _ in range(3)]
    point = atlas.add_point([0,0,5],keyframes[0])
    for keyframe in keyframes:
        point.add_observation(keyframe,0)
        keyframe.map_points[0] = point
    keyframes[2].update_connections()
    assert keyframes[2].covisibility == {0:1,1:1}
    assert keyframes[2].ordered_covisibility == [0]  # strongest fallback, first tie
    assert keyframes[2].parent_id == 0
    assert keyframes[2].id in keyframes[0].children


def test_scaled_world_transform_preserves_projected_pixels():
    atlas = Atlas()
    keyframe = atlas.add_keyframe(make_frame([[300,220]],pose=make_pose(translation=[-.2,0,0])))
    p = atlas.add_point([-.2,-.2,5],keyframe)
    p.add_observation(keyframe,0)
    keyframe.map_points[0] = p
    before = keyframe.camera.project(transform_points(keyframe.pose,p.position))
    angle = .4
    rotation = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    transform = make_pose(rotation,[2,3,1])
    atlas.current.apply_scaled_rotation(transform,1.7)
    after = keyframe.camera.project(transform_points(keyframe.pose,p.position))
    np.testing.assert_allclose(after,before,atol=1e-12)
    assert atlas.current.change_index == 1
