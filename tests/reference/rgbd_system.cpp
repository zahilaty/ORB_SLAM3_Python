// Test-only generated-image source oracle; never part of Python runtime.
// Derived from ORB-SLAM3, University of Zaragoza contributors, 2014-2021.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "System.h"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
int main(int argc,char**argv){
 if(argc!=6)return 2;
 ORB_SLAM3::System slam(argv[1],argv[2],ORB_SLAM3::System::RGBD,false);
 std::ofstream output(argv[4]);output<<std::setprecision(17);
 int count=std::stoi(argv[5]);
 for(int i=0;i<count;i++){
  cv::Mat image=cv::imread(std::string(argv[3])+"/image_"+std::to_string(i)+".png",cv::IMREAD_GRAYSCALE);
  cv::Mat depth=cv::imread(std::string(argv[3])+"/depth_"+std::to_string(i)+".png",cv::IMREAD_UNCHANGED);
  Sophus::SE3f result=slam.TrackRGBD(image,depth,i*.12);
  auto pose=result.matrix();
  auto points=slam.GetTrackedMapPoints();int valid=0;for(auto point:points)if(point&&!point->isBad())valid++;
  output<<i*.12<<" "<<slam.GetTrackingState()<<" "<<slam.GetTrackedKeyPointsUn().size()<<" "<<valid<<" ";
  for(int r=0;r<4;r++)for(int c=0;c<4;c++)output<<pose(r,c)<<" ";output<<"\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
 }
 slam.Shutdown();
}
