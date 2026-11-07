import numpy as np
import pytest
from orbslam3.geometry import (so3_exp, so3_log, se3_exp, se3_log, inverse_pose,
                              transform_points, triangulate, fundamental, Sim3)
from orbslam3.camera import Pinhole, KannalaBrandt8, triangulate_match


@pytest.mark.parametrize("angle", [1e-10, .4, np.pi-1e-8])
def test_lie_roundtrips_and_inverse(angle):
    tangent = np.r_[np.array([1., 2., -1.]) / np.sqrt(6)*angle, [.2, -.3, .7]]
    pose = se3_exp(tangent)
    np.testing.assert_allclose(se3_exp(se3_log(pose)), pose, atol=2e-8)
    np.testing.assert_allclose(pose @ inverse_pose(pose), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(so3_exp(so3_log(pose[:3, :3])), pose[:3, :3], atol=2e-8)


@pytest.mark.parametrize("camera", [Pinhole(500, 510, 320, 240), KannalaBrandt8(500, 510, 320, 240, .01, -.001, .0001, -.00001)])
def test_projection_jacobian_and_roundtrip(camera):
    for point in (np.array([.4, -.6, 2.]), np.array([0., 0., 2.])):
        pixel = camera.project(point)
        np.testing.assert_allclose(camera.unproject(pixel), point/point[2], atol=1e-10)
        numeric = np.column_stack([(camera.project(point+np.eye(3)[i]*1e-6)-camera.project(point-np.eye(3)[i]*1e-6))/2e-6 for i in range(3)])
        np.testing.assert_allclose(camera.project_jacobian(point), numeric, atol=5e-7)


def test_pose_triangulation_and_epipolar_conventions():
    camera = Pinhole(500, 510, 320, 240)
    pose1 = se3_exp(np.array([.02, -.04, .03, .1, .2, .1]))
    pose2 = se3_exp(np.array([.04, -.01, .06, -.3, .2, .1]))
    point = np.array([.3, -.1, 3.])
    p1, p2 = camera.project(transform_points(pose1, point)), camera.project(transform_points(pose2, point))
    np.testing.assert_allclose(triangulate(camera.unproject(p1), camera.unproject(p2), pose1[:3], pose2[:3]), point, atol=1e-12)
    assert abs(np.r_[p1, 1.] @ fundamental(camera, pose1, camera, pose2) @ np.r_[p2, 1.]) < 1e-12
    relative = pose1 @ inverse_pose(pose2)
    assert camera.epipolar_constrain(camera, p1, p2, relative[:3, :3], relative[:3, 3], 1., 1.)
    np.testing.assert_allclose(triangulate_match(camera, camera, p1, p2, pose1, pose2), point, atol=1e-12)
    assert triangulate_match(camera, camera, p1, p1, pose1, pose1) is None


def test_sim3_exp_log_composition_and_camera_pose():
    tangent = np.array([.1, -.3, .2, .3, -.4, .5, .7])
    similarity = Sim3.exp(tangent)
    np.testing.assert_allclose(similarity.log(), tangent, atol=1e-12)
    point = np.array([.3, .4, 3.])
    np.testing.assert_allclose(similarity.inverse().map(similarity.map(point)), point, atol=1e-12)
    np.testing.assert_allclose((similarity @ similarity.inverse()).map(point), point, atol=1e-12)
    camera = Pinhole(500, 510, 320, 240)
    np.testing.assert_allclose(camera.project(similarity.map(point)), camera.project(transform_points(similarity.camera_pose(), point)))
