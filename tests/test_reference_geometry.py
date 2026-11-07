"""Standalone source golden checks; original C++ compilation is opt-in."""

import json
import os
from pathlib import Path
import subprocess
import hashlib

import numpy as np
import pytest

from orbslam3.camera import Pinhole, KannalaBrandt8
from orbslam3.geometry import Sim3
from orbslam3.initialization import TwoViewReconstruction, score_homography, score_fundamental
from orbslam3.optimization import ReprojectionObservation, local_bundle_adjustment


FIXTURE_PATH = Path(__file__).parent / "fixtures/geometry_initialization_reference.json"


def reference_fixture():
    return json.loads(FIXTURE_PATH.read_text())


def test_original_camera_projection_unprojection_and_jacobian_goldens():
    for case in reference_fixture()["camera_cases"]:
        parameters = case["parameters"]
        camera = (Pinhole(*parameters[:4]) if case["model"] == "pinhole"
                  else KannalaBrandt8(*parameters))
        actual = np.array([np.r_[camera.project(point), camera.unproject(pixel),
                                  camera.project_jacobian(point).ravel()]
                           for point, pixel in zip(case["points"], case["pixels"])])
        expected = np.array(case["expected"])
        # Original KB8's Eigen double projection still uses atan2f/sqrtf, and
        # calibration/unprojection are float32. Differences are numerical.
        np.testing.assert_allclose(actual[:, :2], expected[:, :2], atol=5e-5, rtol=1e-7)
        np.testing.assert_allclose(actual[:, 2:5], expected[:, 2:5], atol=3e-7, rtol=1e-6)
        np.testing.assert_allclose(actual[:, 5:], expected[:, 5:], atol=6e-6, rtol=1e-7)


def test_original_two_view_symmetric_scores_and_inlier_masks():
    for case in reference_fixture()["two_view_cases"]:
        first, second = np.array(case["keys1"]), np.array(case["keys2"])
        homography_score, homography_inliers = score_homography(
            np.array(case["homography"]), first, second, 1.)
        fundamental_score, fundamental_inliers = score_fundamental(
            np.array(case["fundamental"]), first, second, 1.)
        # Both scores sum 200 float32 terms in C++; Python accumulates float64.
        assert homography_score == pytest.approx(case["homography_score"], abs=.002)
        assert fundamental_score == pytest.approx(case["fundamental_score"], abs=.002)
        np.testing.assert_array_equal(homography_inliers, case["homography_inliers"])
        np.testing.assert_array_equal(fundamental_inliers, case["fundamental_inliers"])


def test_original_initialization_decisions_and_clean_reconstruction():
    camera = Pinhole(500, 500, 320, 240)
    for case in reference_fixture()["two_view_cases"]:
        first, second = np.array(case["keys1"]), np.array(case["keys2"])
        result = TwoViewReconstruction(camera.K).reconstruct(first, second, np.arange(len(first)))
        assert result.success == case["success"], case["name"]
        assert result.model == case["model"], case["name"]
        if result.success:
            # DUtils and NumPy sample different minimal subsets. The source's
            # selected noisy minimal model differs; clean geometry is tight.
            tolerance = .02 if case["name"] == "general_noisy" else 4e-5
            np.testing.assert_allclose(result.pose, case["pose"], atol=tolerance, rtol=0)
            np.testing.assert_array_equal(result.triangulated, case["triangulated"])
            if case["name"] == "general_clean":
                np.testing.assert_allclose(result.points, case["points"], atol=.002, rtol=0)


def test_sim3_logarithm_is_defined_at_a_half_turn():
    # A principal complex matrix log followed by .real is not a Lie logarithm.
    # Loop transforms can contain 180-degree relative camera rotations.
    for axis in (np.array([1., 0, 0]), np.array([1., 2., -1.]) / np.sqrt(6)):
        tangent = np.r_[axis * np.pi, [.2, -.1, .3], .05]
        similarity = Sim3.exp(tangent)
        recovered = Sim3.exp(similarity.log())
        np.testing.assert_allclose(recovered.rotation, similarity.rotation, atol=2e-10)
        np.testing.assert_allclose(recovered.translation, similarity.translation, atol=2e-10)
        assert recovered.scale == pytest.approx(similarity.scale, abs=1e-12)


def test_original_visual_lba_on_a_real_image_pipeline_graph():
    graph = np.load(Path(__file__).parent / "fixtures/visual_lba_reference.npz")
    observations = []
    for pose_id, point_id, measurement, invvar, bf, intrinsics in zip(
            graph["obs_pose_ids"], graph["obs_point_ids"], graph["obs_measurements"],
            graph["obs_inverse_variance"], graph["obs_baseline_fx"], graph["obs_intrinsics"]):
        measurement = measurement[:2] if np.isnan(measurement[2]) else measurement
        observations.append(ReprojectionObservation(int(pose_id), int(point_id), measurement,
                                                      Pinhole(*intrinsics), invvar, bf))
    result = local_bundle_adjustment(dict(zip(graph["pose_ids"], graph["poses"])),
                                      dict(zip(graph["point_ids"], graph["points"])),
                                      observations, graph["fixed_ids"])
    actual_poses = np.array([result.poses[i] for i in graph["pose_ids"]])
    actual_points = np.array([result.points[i] for i in graph["point_ids"]])
    np.testing.assert_allclose(actual_poses, graph["expected_poses"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual_points, graph["expected_points"], atol=2e-5, rtol=0)
    np.testing.assert_allclose(result.chi2, graph["expected_chi2"], atol=1e-4, rtol=1e-5)
    np.testing.assert_array_equal(result.outliers, graph["expected_outliers"])
    np.testing.assert_allclose([result.initial_cost, result.final_cost],
                               graph["expected_summary"][:2], atol=.002, rtol=0)


def test_original_rgbd_system_tracking_on_identical_generated_images():
    from orbslam3 import SLAM, Sensor, Calibration
    from orbslam3.geometry import inverse_pose
    from synthetic import TexturedPlanes
    fixture = json.loads((Path(__file__).parent / "fixtures/rgbd_system_reference.json").read_text())
    scene = TexturedPlanes()
    calibration = Calibration(scene.camera, baseline_fx=26., depth_threshold=4., frames_per_second=20.)
    slam = SLAM(Sensor.RGBD, calibration, nfeatures=750, nlevels=4)
    poses = []
    for i, timestamp in enumerate(fixture["times"]):
        frame = scene.frame(Sensor.RGBD, timestamp)
        assert hashlib.sha256(frame.image.tobytes()).hexdigest() == fixture["provenance"]["image_sha256"][i]
        assert hashlib.sha256(frame.depth.astype(np.float32).tobytes()).hexdigest() == fixture["provenance"]["depth_meters_sha256"][i]
        result = slam.process(frame)
        assert int(result.state) == fixture["states"][i]
        assert len(slam.frame.features) == fixture["features"][i]
        poses.append(result.pose)
    # Source has live workers; Python commits mapping at deterministic keyframe
    # boundaries. Compare the returned poses, not an unrelated ground-truth fit.
    np.testing.assert_allclose(poses, fixture["poses"], atol=.001, rtol=0)
    positions = np.array([inverse_pose(pose)[:3, 3] for pose in poses])
    original_positions = np.array([inverse_pose(np.array(pose))[:3, 3] for pose in fixture["poses"]])
    assert np.linalg.norm(positions - original_positions, axis=1).max() < .001


CPP_REFERENCE = r'''
// Test-only original source oracle. SPDX-License-Identifier: GPL-3.0-or-later
#include <bits/stdc++.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#define private public
#include "TwoViewReconstruction.h"
#undef private
#include "CameraModels/Pinhole.h"
#include "CameraModels/KannalaBrandt8.h"
int main(){
 std::cout<<std::setprecision(17);
 int mode;std::cin>>mode;
 if(mode==0){
  int model,count;std::cin>>model>>count;std::vector<float> p(8);for(float&v:p)std::cin>>v;
  ORB_SLAM3::Pinhole pinhole(std::vector<float>(p.begin(),p.begin()+4));ORB_SLAM3::KannalaBrandt8 fisheye(p);
  ORB_SLAM3::GeometricCamera*camera=model?static_cast<ORB_SLAM3::GeometricCamera*>(&fisheye):static_cast<ORB_SLAM3::GeometricCamera*>(&pinhole);
  for(int i=0;i<count;i++){
   Eigen::Vector3d point;float u,v;std::cin>>point[0]>>point[1]>>point[2]>>u>>v;
   auto pixel=camera->project(point);auto jac=camera->projectJac(point);auto ray=camera->unproject(cv::Point2f(u,v));
   std::cout<<pixel[0]<<" "<<pixel[1]<<" "<<ray.x<<" "<<ray.y<<" "<<ray.z<<" ";
   for(int r=0;r<2;r++)for(int c=0;c<3;c++)std::cout<<jac(r,c)<<" ";std::cout<<"\n";
  }
 }else{
  int count;float fx,fy,cx,cy;std::cin>>count>>fx>>fy>>cx>>cy;
  Eigen::Matrix3f calibration;calibration<<fx,0,cx,0,fy,cy,0,0,1;
  std::vector<cv::KeyPoint>first,second;std::vector<int>matches;
  for(int i=0;i<count;i++){float x,y,u,v;int m;std::cin>>x>>y>>u>>v>>m;first.emplace_back(x,y,31);second.emplace_back(u,v,31);matches.push_back(m);}
  ORB_SLAM3::TwoViewReconstruction reconstruction(calibration);
  Sophus::SE3f pose;std::vector<cv::Point3f>points;std::vector<bool>triangulated;
  bool success=reconstruction.Reconstruct(first,second,matches,pose,points,triangulated);
  std::vector<bool>hi,fi;float hs,fs;Eigen::Matrix3f h,f;
  reconstruction.FindHomography(hi,hs,h);reconstruction.FindFundamental(fi,fs,f);
  std::cout<<success<<" "<<hs<<" "<<fs<<" "<<(hs/(hs+fs)>.50?1:0)<<" "<<points.size()<<" "<<triangulated.size()<<"\n";
  Eigen::Matrix4f matrix=pose.matrix();for(int i=0;i<4;i++)for(int j=0;j<4;j++)std::cout<<matrix(i,j)<<" ";std::cout<<"\n";
  for(int i=0;i<3;i++)for(int j=0;j<3;j++)std::cout<<h(i,j)<<" ";std::cout<<"\n";
  for(int i=0;i<3;i++)for(int j=0;j<3;j++)std::cout<<f(i,j)<<" ";std::cout<<"\n";
  for(bool v:hi)std::cout<<v<<" ";std::cout<<"\n";for(bool v:fi)std::cout<<v<<" ";std::cout<<"\n";
  for(bool v:triangulated)std::cout<<v<<" ";std::cout<<"\n";
  for(auto v:points)std::cout<<v.x<<" "<<v.y<<" "<<v.z<<" ";std::cout<<"\n";
 }
}
'''


@pytest.mark.reference
def test_live_original_two_view_and_camera_fixture(tmp_path):
    reference_path = os.environ.get("ORBSLAM3_REFERENCE")
    if not reference_path:
        pytest.skip("Set ORBSLAM3_REFERENCE to opt into the original C++ oracle")
    reference = Path(reference_path)
    source = tmp_path / "geometry_reference.cpp"
    source.write_text(CPP_REFERENCE)
    executable = tmp_path / "geometry_reference"
    native = reference / ".venv"
    includes = [reference, reference / "include", reference / "Thirdparty/Sophus",
                native / "include", native / "include/eigen3", native / "include/opencv4"]
    libraries = [reference / "lib", reference / "Thirdparty/g2o/lib",
                 reference / "Thirdparty/DBoW2/lib", reference / ".local/lib", native / "lib"]
    command = ["g++", "-std=c++17", "-O1", "-w", "-march=native", "-DNDEBUG",
               str(source), "-o", str(executable)]
    command += [f"-I{path}" for path in includes]
    command += [f"-L{path}" for path in libraries]
    command += ["-lORB_SLAM3", "-lg2o", "-lopencv_core", "-lopencv_features2d"]
    command += [f"-Wl,-rpath,{path}" for path in libraries]
    subprocess.run(command, check=True, capture_output=True, text=True)
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(map(str, libraries))
    for case in reference_fixture()["two_view_cases"]:
        rows = ["1", f"{len(case['keys1'])} 500 500 320 240"]
        rows += [" ".join(map(str, [*first, *second, i]))
                 for i, (first, second) in enumerate(zip(case["keys1"], case["keys2"]))]
        output = subprocess.run([str(executable)], input="\n".join(rows), text=True,
                                capture_output=True, check=True, env=environment).stdout.splitlines()
        header = np.fromstring(output[0], sep=" ")
        assert bool(header[0]) == case["success"]
        assert ("homography" if header[3] else "fundamental") == case["model"]
        np.testing.assert_allclose(header[1:3], [case["homography_score"], case["fundamental_score"]], atol=1e-5)
