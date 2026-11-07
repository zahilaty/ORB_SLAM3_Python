import cv2
import numpy as np
import pytest
from orbslam3.camera import Pinhole, KannalaBrandt8
from orbslam3.features import FeatureSet
from orbslam3.imu import ImuCalibration
from orbslam3.input import (Sensor, Calibration, FrameInput, FrameBuilder,
                            grayscale, undistort_features, image_bounds, rectified_stereo,video_frames)


def image_pair():
    rng = np.random.default_rng(33)
    left = rng.integers(0,256,(240,320),dtype=np.uint8)
    right = np.clip(np.roll(left,-10,axis=1).astype(np.int16)+rng.integers(-2,3,left.shape),0,255).astype(np.uint8)
    return left,right


def test_color_order_and_input_shape_checks():
    red_rgb = np.full((3,4,3),[255,0,0],np.uint8)
    assert grayscale(red_rgb,"RGB")[0,0] == 76
    assert grayscale(red_rgb,"BGR")[0,0] == 29
    gray = np.zeros((4,3),np.uint8)
    assert grayscale(gray) is gray
    with pytest.raises(ValueError,match="uint8"):
        grayscale(gray.astype(float))
    with pytest.raises(ValueError,match="grayscale"):
        grayscale(np.zeros((4,3,2),np.uint8))


@pytest.mark.parametrize("sensor",list(Sensor))
def test_all_six_sensor_inputs_have_consistent_feature_and_depth_arrays(sensor):
    left,right = image_pair()
    imu = ImuCalibration(np.eye(4),.01,.05,.001,.005) if sensor.inertial else None
    calibration = Calibration(Pinhole(250,250,160,120),baseline_fx=25,depth_threshold=4.,
                              depth_units_per_meter=1000.,imu=imu)
    builder = FrameBuilder(sensor,calibration,nfeatures=200,nlevels=3)
    data = FrameInput(1.25,left,right_image=right if sensor.stereo else None,
                      depth=np.full(left.shape,2000,np.uint16) if not sensor.monocular and not sensor.stereo else None)
    frame = builder.build(data)
    assert len(frame.features) >= 190
    assert len(frame.map_points) == len(frame.right_u) == len(frame.depth) == len(frame.features)
    assert frame.timestamp == 1.25
    if sensor.monocular:
        assert (frame.depth == -1).all()
    elif not sensor.stereo:
        np.testing.assert_array_equal(frame.depth,2.)
        np.testing.assert_allclose(frame.right_u,frame.features.points[:,0]-12.5)
    else:
        valid = frame.depth > 0
        assert valid.sum() > 40
        assert np.median(frame.depth[valid]) == pytest.approx(2.5,abs=.02)


def test_rgbd_depth_indexing_uses_raw_pixels_but_residuals_use_undistorted_pixels():
    left,_ = image_pair()
    depth = np.broadcast_to(np.arange(1,321),(240,320)).astype(np.float32)
    camera = Pinhole(250,250,160,120)
    builder = FrameBuilder(Sensor.RGBD,Calibration(camera,baseline_fx=25,distortion=(.1,-.05,0,0)),
                           nfeatures=100,nlevels=3)
    frame = builder.build(FrameInput(0,left,depth=depth))
    raw_indices = frame.raw_features.points.astype(int)
    np.testing.assert_array_equal(frame.depth,depth[raw_indices[:,1],raw_indices[:,0]])
    assert not np.array_equal(frame.raw_features.points,frame.features.points)
    np.testing.assert_allclose(frame.right_u,frame.features.points[:,0]-25/frame.depth)
    assert not np.array_equal(frame.bounds,(0.,320.,0.,240.))


def test_invalid_depth_samples_do_not_create_synthetic_stereo_measurements():
    left,_ = image_pair()
    depth = np.full(left.shape,np.nan)
    builder = FrameBuilder(Sensor.RGBD,Calibration(Pinhole(250,250,160,120),baseline_fx=25),
                           nfeatures=100,nlevels=3)
    frame = builder.build(FrameInput(0,left,depth=depth))
    assert np.all(frame.depth == -1) and np.all(frame.right_u == -1)


def test_explicit_nonoverlapping_fisheye_rig_remains_unmatched():
    left,right = image_pair()
    camera = KannalaBrandt8(250,250,160,120,.01,-.001,0,0)
    relative = np.eye(4)
    relative[0,3] = -.1
    builder = FrameBuilder(Sensor.STEREO,Calibration(camera,right_camera=camera,T_right_left=relative,
                                                overlap_left=(-1,-1),overlap_right=(-1,-1)),
                           nfeatures=100,nlevels=3)
    frame = builder.build(FrameInput(0,left,right_image=right))
    assert 0 < frame.n_left < len(frame.features)
    assert frame.stereo_matches.shape == (len(frame.features),)
    assert np.all(frame.stereo_matches == -1)
    assert np.all(frame.depth == -1)
    assert frame.stereo_points.shape == (frame.n_left,3)


def test_required_mode_inputs_and_calibration_are_checked():
    left,_ = image_pair()
    camera = Pinhole(250,250,160,120)
    with pytest.raises(ValueError,match="IMU calibration"):
        FrameBuilder(Sensor.MONOCULAR_IMU,Calibration(camera))
    with pytest.raises(ValueError,match="baseline_fx"):
        FrameBuilder(Sensor.RGBD,Calibration(camera))
    builder = FrameBuilder(Sensor.STEREO,Calibration(camera,baseline_fx=25),nfeatures=100,nlevels=3)
    with pytest.raises(ValueError,match="right_image"):
        builder.build(FrameInput(0,left))
    builder = FrameBuilder(Sensor.RGBD,Calibration(camera,baseline_fx=25),nfeatures=100,nlevels=3)
    with pytest.raises(ValueError,match="image-aligned"):
        builder.build(FrameInput(0,left,depth=np.zeros((2,2))))
    with pytest.raises(ValueError,match="finite"):
        builder.build(FrameInput(float("nan"),left))


def test_rectified_empty_features_are_valid_without_empty_median_access():
    empty = FeatureSet(np.empty((0,2),np.float32),np.empty((0,32),np.uint8),np.empty(0,np.int32),
                       np.empty(0,np.float32),np.empty(0,np.float32),np.empty(0,np.float32),0)
    gray = np.zeros((240,320),np.uint8)
    right_u,depth = rectified_stereo(empty,empty,[gray],[gray],np.array([1.]),25,250)
    assert right_u.shape == depth.shape == (0,)


def test_user_video_adapter_emits_seconds_and_releases_file(tmp_path):
    path = tmp_path/"own_video.avi"
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),20.,(64,48))
    assert writer.isOpened()
    for value in (40,80,120):
        writer.write(np.full((48,64,3),value,np.uint8))
    writer.release()
    frames = list(video_frames(path,start_time=10.))
    assert len(frames) == 3
    assert frames[0].timestamp == pytest.approx(10.)
    np.testing.assert_allclose(np.diff([frame.timestamp for frame in frames]),.05,atol=1e-6)
    assert frames[0].image.shape == (48,64,3)
    with pytest.raises(ValueError,match="Cannot open"):
        list(video_frames(tmp_path/"missing.avi"))
