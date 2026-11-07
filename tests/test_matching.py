"""Decision-focused tests for the source ORBmatcher overloads."""
import numpy as np
import pytest

from orbslam3.camera import Pinhole
from orbslam3.features import FeatureSet
from orbslam3.geometry import Sim3
from orbslam3.map import Frame, KeyFrame, Map, MapPoint
from orbslam3.matching import (
    compute_three_maxima, search_for_initialization, search_by_bow_frame,
    search_by_bow_keyframes, search_by_projection_local,
    search_by_projection_last_frame, search_by_projection_relocalization,
    search_for_triangulation, search_by_projection_sim3, search_by_sim3,
    fuse, fuse_sim3, _record_orientation,
)


def descriptor(distance=0):
    bits = np.zeros(256,np.uint8)
    bits[:distance] = 1
    return np.packbits(bits,bitorder="little")


def frame(points, descriptors=None, octaves=None, angles=None, keyframe=False,
          frame_id=0, map_=None, **extra):
    points = np.asarray(points,dtype=np.float32).reshape(-1,2)
    n = len(points)
    features = FeatureSet(points,
                          np.asarray(descriptors,dtype=np.uint8) if descriptors is not None else np.zeros((n,32),np.uint8),
                          np.asarray(octaves,dtype=np.int32) if octaves is not None else np.zeros(n,np.int32),
                          np.asarray(angles,dtype=np.float32) if angles is not None else np.zeros(n,np.float32),
                          np.ones(n,np.float32),np.full(n,31,np.float32),n)
    cls = KeyFrame if keyframe else Frame
    kwargs = dict(id=frame_id,timestamp=0.,features=features,camera=Pinhole(500,500,320,240),
                  width=640,height=480,scales=np.array([1.,1.2,1.44]),
                  inverse_variances=1/np.array([1.,1.44,2.0736]),pose=np.eye(4),
                  feature_nodes={1:list(range(n))})
    if keyframe:
        kwargs["map"] = map_ if map_ is not None else Map(0,frame_id)
    kwargs.update(extra)
    result = cls(**kwargs)
    if keyframe:
        result.map.keyframes[frame_id] = result
    return result


def point(map_,point_id=0,position=(0,0,5),observations=0):
    result = MapPoint(point_id,np.array(position,dtype=float),-1,map_,
                      descriptor=descriptor(),normal=np.array([0.,0.,1.]),
                      min_distance=1.,max_distance=5.,observation_count=observations)
    map_.points[point_id] = result
    return result


def test_orientation_preserves_source_factor_and_three_maxima_rules():
    histogram = [[] for _ in range(30)]
    _record_orientation(histogram,90.,0.,7)
    assert histogram[3] == [7]  # actual factor is 1/30; conventional bin 7.5 is wrong here
    histogram[0] = list(range(100))
    histogram[1] = list(range(9))
    assert compute_three_maxima(histogram) == (0,-1,-1)
    histogram[1] = list(range(10))
    assert compute_three_maxima(histogram) == (0,1,-1)


def test_initialization_reassigns_only_to_strictly_better_source_match():
    first = frame([[320,240],[321,240],[322,240]],
                  [descriptor(5),descriptor(0),descriptor(0)],octaves=[0,0,1])
    second = frame([[325,240]])
    previous = first.features.points.copy()
    matches = search_for_initialization(first,second,previous,check_orientation=False)
    np.testing.assert_array_equal(matches,[-1,0,-1])
    np.testing.assert_array_equal(previous[1],[325,240])
    np.testing.assert_array_equal(previous[[0,2]],first.features.points[[0,2]])


def test_initialization_ratio_rejects_equally_good_matches():
    first,second = frame([[320,240]]),frame([[320,240],[321,240]])
    assert search_for_initialization(first,second,first.features.points.copy())[0] == -1


def test_bow_node_restriction_and_threshold_boundary():
    map_ = Map(0,0)
    source = frame([[320,240]],keyframe=True,map_=map_)
    p = point(map_)
    source.map_points[0] = p
    target = frame([[320,240]],[descriptor(50)])
    assert search_by_bow_frame(source,target,check_orientation=False) == [p]
    target.feature_nodes = {2:[0]}
    assert search_by_bow_frame(source,target) == [None]
    target = frame([[320,240]],[descriptor(50)],keyframe=True,frame_id=1,map_=map_)
    target.map_points[0] = point(map_,1)
    assert search_by_bow_keyframes(source,target,check_orientation=False) == [None]


def test_bow_right_camera_ratio_is_disabled_but_left_threshold_gates_right():
    map_ = Map(0,0)
    source = frame([[320,240]],keyframe=True,map_=map_)
    p = point(map_)
    source.map_points[0] = p
    target = frame([[320,240],[320,240],[321,240]],
                   [descriptor(80),descriptor(),descriptor()],n_left=1,
                   right_camera=source.camera,T_right_left=np.eye(4))
    assert search_by_bow_frame(source,target,check_orientation=False) == [None,None,None]
    target.features.descriptors[0] = descriptor(40)
    assert search_by_bow_frame(source,target,check_orientation=False) == [p,p,None]


def test_local_projection_same_octave_ratio_and_right_reprojection_gate():
    map_ = Map(0,0)
    p = point(map_)
    current = frame([[320,240],[321,240]],[descriptor(20),descriptor(21)])
    assert search_by_projection_local(current,[p]) == 0
    current.features.octaves[1] = 1
    # At predicted scale0 the octave1 candidate is outside the source's
    # [-1,0] range, so only the first candidate remains and ratio is inactive.
    assert search_by_projection_local(current,[p]) == 1
    current.map_points = [None,None]
    current.right_u[0] = 100  # expected 320, no baseline
    assert search_by_projection_local(current,[p]) == 0
    current.right_u[0] = -1
    assert search_by_projection_local(current,[p],far_points=True,far_threshold=4) == 0
    p.normal[:] = [0,0,-1]
    assert search_by_projection_local(current,[p]) == 0


def test_local_dual_camera_match_propagates_global_counterpart():
    map_ = Map(0,0)
    p = point(map_)
    current = frame([[320,240],[320,240]],n_left=1,right_camera=Pinhole(500,500,320,240),
                    T_right_left=np.eye(4),stereo_matches=np.array([1,0]))
    assert search_by_projection_local(current,[p]) >= 2
    assert current.map_points == [p,p]


def test_motion_model_projection_uses_outlier_and_stereo_error_gates():
    map_ = Map(0,0)
    p = point(map_)
    last,current = frame([[320,240]]),frame([[320,240]],baseline_fx=50.)
    last.map_points[0] = p
    current.right_u[0] = 310.
    assert search_by_projection_last_frame(current,last,check_orientation=False) == 1
    current.map_points[0] = None
    last.outliers[0] = True
    assert search_by_projection_last_frame(current,last) == 0
    last.outliers[0] = False
    current.right_u[0] = 290.
    assert search_by_projection_last_frame(current,last) == 0


def test_relocalization_projection_respects_already_found_and_distance_bounds():
    map_ = Map(0,0)
    p = point(map_)
    source = frame([[320,240]],keyframe=True,map_=map_)
    source.map_points[0] = p
    target = frame([[320,240]])
    assert search_by_projection_relocalization(target,source,{p}) == 0
    assert search_by_projection_relocalization(target,source,set()) == 1
    target.map_points[0] = None
    p.min_distance = 10
    assert search_by_projection_relocalization(target,source,set()) == 0


def test_triangulation_preserves_source_many_to_one_tie_choice():
    first = frame([[320,240],[320,240]],keyframe=True)
    second = frame([[300,240],[301,240]],keyframe=True,frame_id=1,map_=first.map)
    second.pose[0,3] = -0.2
    pairs = search_for_triangulation(first,second)
    assert pairs == [(0,1),(1,1)]  # source equal distances accept the last candidate
    assert search_for_triangulation(first,second,only_stereo=True) == []
    first.right_u[:] = 310
    second.right_u[:] = 290
    assert search_for_triangulation(first,second,only_stereo=True) == pairs
    first.map_points[0] = point(first.map)
    assert search_for_triangulation(first,second) == [(1,1)]


def test_sim3_projection_and_mutual_agreement():
    map_ = Map(0,0)
    first = frame([[320,240]],keyframe=True,map_=map_)
    second = frame([[320,240]],keyframe=True,frame_id=1,map_=map_)
    p1,p2 = point(map_,0),point(map_,1)
    first.map_points[0],second.map_points[0] = p1,p2
    identity = Sim3(np.eye(3),np.zeros(3))
    matches = [None]
    assert search_by_sim3(first,second,matches,identity) == 1
    assert matches == [p2]
    matches = [None]
    assert search_by_projection_sim3(first,identity,[p2],matches) == 1
    assert matches == [p2]
    assert search_by_projection_sim3(first,identity,[p2],matches) == 0


def test_fusion_adds_observation_and_replaces_lower_support_landmark():
    map_ = Map(0,0)
    keyframe = frame([[320,240]],keyframe=True,map_=map_)
    existing = point(map_,0)
    assert fuse(keyframe,[existing]) == 1
    assert keyframe.map_points[0] is existing
    assert existing.observations[keyframe.id] == (0,-1)
    assert fuse(keyframe,[existing]) == 0
    incoming = point(map_,1,observations=3)
    assert fuse(keyframe,[incoming]) == 1
    assert existing.bad and existing.replaced is incoming
    assert keyframe.map_points[0] is incoming


def test_sim3_fusion_reports_pending_replacement():
    map_ = Map(0,0)
    keyframe = frame([[320,240]],keyframe=True,map_=map_)
    existing,incoming = point(map_,0),point(map_,1)
    keyframe.map_points[0] = existing
    fused,replacements = fuse_sim3(keyframe,Sim3(np.eye(3),np.zeros(3)),[incoming])
    assert fused == 1 and replacements == [existing]
    assert not incoming.bad and not existing.bad
