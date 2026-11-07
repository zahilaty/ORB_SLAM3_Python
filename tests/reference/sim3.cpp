// Test-only oracle for regenerating fixtures/sim3_optimization_reference.json.
// Calls the exact original factors and g2o; never part of Python runtime.
// Derived from ORB-SLAM3, University of Zaragoza, 2014-2021.
// SPDX-License-Identifier: GPL-3.0-or-later
#include <iomanip>
#include <iostream>
#include <vector>
#include "OptimizableTypes.h"
#include "CameraModels/Pinhole.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
int main(){
 using namespace g2o;
 ORB_SLAM3::Pinhole camera(std::vector<float>{500,510,320,240});
 SparseOptimizer optimizer;
 auto linear=new LinearSolverDense<BlockSolverX::PoseMatrixType>();
 optimizer.setAlgorithm(new OptimizationAlgorithmLevenberg(new BlockSolverX(linear)));
 Eigen::Matrix3d rotation;Eigen::Vector3d translation;double scale;int fixed,n;
 for(int i=0;i<3;i++)for(int j=0;j<3;j++)std::cin>>rotation(i,j);
 std::cin>>translation[0]>>translation[1]>>translation[2]>>scale>>fixed>>n;
 Sim3 initial(rotation,translation,scale);
 auto vertex=new ORB_SLAM3::VertexSim3Expmap();
 vertex->_fix_scale=fixed;vertex->setEstimate(initial);vertex->setId(0);
 vertex->pCamera1=&camera;vertex->pCamera2=&camera;optimizer.addVertex(vertex);
 std::vector<ORB_SLAM3::EdgeSim3ProjectXYZ*> forward;
 std::vector<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*> inverse;
 for(int i=0;i<n;i++){
  Eigen::Vector3d p1,p2;Eigen::Vector2d obs1,obs2;double inv1,inv2;
  std::cin>>p1[0]>>p1[1]>>p1[2]>>p2[0]>>p2[1]>>p2[2]>>obs1[0]>>obs1[1]>>obs2[0]>>obs2[1]>>inv1>>inv2;
  auto point1=new VertexSBAPointXYZ();point1->setId(2*i+1);point1->setEstimate(p1);point1->setFixed(true);optimizer.addVertex(point1);
  auto point2=new VertexSBAPointXYZ();point2->setId(2*i+2);point2->setEstimate(p2);point2->setFixed(true);optimizer.addVertex(point2);
  auto e12=new ORB_SLAM3::EdgeSim3ProjectXYZ();e12->setVertex(0,point2);e12->setVertex(1,vertex);e12->setMeasurement(obs1);e12->setInformation(Eigen::Matrix2d::Identity()*inv1);
  auto r12=new RobustKernelHuber();r12->setDelta(std::sqrt(10.0));e12->setRobustKernel(r12);optimizer.addEdge(e12);forward.push_back(e12);
  auto e21=new ORB_SLAM3::EdgeInverseSim3ProjectXYZ();e21->setVertex(0,point1);e21->setVertex(1,vertex);e21->setMeasurement(obs2);e21->setInformation(Eigen::Matrix2d::Identity()*inv2);
  auto r21=new RobustKernelHuber();r21->setDelta(std::sqrt(10.0));e21->setRobustKernel(r21);optimizer.addEdge(e21);inverse.push_back(e21);
 }
 optimizer.initializeOptimization();optimizer.optimize(5);
 std::vector<int> inlier(n,1);int bad=0;
 for(int i=0;i<n;i++){
  if(forward[i]->chi2()>10||inverse[i]->chi2()>10){inlier[i]=0;bad++;optimizer.removeEdge(forward[i]);optimizer.removeEdge(inverse[i]);}
  else{forward[i]->setRobustKernel(nullptr);inverse[i]->setRobustKernel(nullptr);}
 }
 if(n-bad>=10){optimizer.initializeOptimization();optimizer.optimize(bad>0?10:5);}
 for(int i=0;i<n;i++){if(inlier[i]){forward[i]->computeError();inverse[i]->computeError();inlier[i]=forward[i]->chi2()<=10&&inverse[i]->chi2()<=10;}}
 Sim3 result=n-bad>=10?vertex->estimate():initial;
 std::cout<<std::setprecision(17);
 for(int i=0;i<3;i++)for(int j=0;j<3;j++)std::cout<<result.rotation().toRotationMatrix()(i,j)<<" ";
 for(int i=0;i<3;i++)std::cout<<result.translation()[i]<<" ";std::cout<<result.scale()<<"\n";
 for(int i=0;i<n;i++)std::cout<<inlier[i]<<" ";std::cout<<"\n";
}
