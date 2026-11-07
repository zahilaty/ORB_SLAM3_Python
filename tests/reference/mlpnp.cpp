// Test-only oracle for regenerating fixtures/mlpnp_reference.json.
// Calls the exact original shared library; never part of Python runtime.
// Derived from ORB-SLAM3, University of Zaragoza, 2014-2021.
// SPDX-License-Identifier: GPL-3.0-or-later
#include <iomanip>
#include <iostream>
#include <vector>
#include "Frame.h"
#include "MapPoint.h"
#include "CameraModels/Pinhole.h"
#define private public
#include "MLPnPsolver.h"
#undef private
int main() {
 using namespace ORB_SLAM3;
 using Solver=MLPnPsolver;
 Solver::bearingVectors_t bearings; Solver::points_t points;
 int count; std::cin>>count;
 for(int i=0;i<count;i++){Eigen::Vector3d point,ray;std::cin>>point[0]>>point[1]>>point[2]>>ray[0]>>ray[1]>>ray[2];points.push_back(point);bearings.push_back(ray);}
 Pinhole camera(std::vector<float>{500,510,320,240});
 Frame frame;frame.mpCamera=&camera;frame.mvLevelSigma2={1};
 std::vector<MapPoint*> matches;
 for(int i=0;i<6;i++){auto point=new MapPoint();point->SetWorldPos(points[i].cast<float>());matches.push_back(point);frame.mvKeysUn.push_back(cv::KeyPoint(0,0,1));}
 frame.mvpMapPoints=matches;
 auto solver=new Solver(frame,matches);
 std::vector<int> indices;for(int i=0;i<count;i++)indices.push_back(i);
 Solver::cov3_mats_t covariance(1);Solver::transformation_t pose;
 solver->computePose(bearings,points,covariance,indices,pose);
 std::cout<<std::setprecision(17);
 for(int i=0;i<3;i++)for(int j=0;j<4;j++)std::cout<<pose(i,j)<<" ";
 std::cout<<"\n";
}
