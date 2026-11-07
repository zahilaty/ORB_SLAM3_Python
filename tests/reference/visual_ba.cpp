// Test-only original visual-LBA g2o graph oracle; GPL-3.0-or-later.
#include <iomanip>
#include <iostream>
#include <vector>
#include <map>
#include "OptimizableTypes.h"
#include "CameraModels/Pinhole.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
int main(){
 using namespace g2o;
 int np,nl,ne,nfixed;std::cin>>np>>nl>>ne>>nfixed;
 std::set<int>fixed;for(int i=0;i<nfixed;i++){int id;std::cin>>id;fixed.insert(id);}
 SparseOptimizer optimizer;
 auto linear=new LinearSolverEigen<BlockSolver_6_3::PoseMatrixType>();
 optimizer.setAlgorithm(new OptimizationAlgorithmLevenberg(new BlockSolver_6_3(linear)));
 std::vector<int>poseids,pointids;int maximum=-1;
 for(int i=0;i<np;i++){
  int id;std::cin>>id;poseids.push_back(id);maximum=std::max(maximum,id);Eigen::Matrix4d matrix;
  for(int r=0;r<4;r++)for(int c=0;c<4;c++)std::cin>>matrix(r,c);
  auto vertex=new VertexSE3Expmap();vertex->setId(id);vertex->setEstimate(SE3Quat(matrix.topLeftCorner<3,3>(),matrix.topRightCorner<3,1>()));vertex->setFixed(fixed.count(id));optimizer.addVertex(vertex);
 }
 for(int i=0;i<nl;i++){
  int id;Eigen::Vector3d point;std::cin>>id>>point[0]>>point[1]>>point[2];pointids.push_back(id);
  auto vertex=new VertexSBAPointXYZ();vertex->setId(id+maximum+1);vertex->setEstimate(point);vertex->setMarginalized(true);optimizer.addVertex(vertex);
 }
 std::vector<OptimizableGraph::Edge*>edges;
 for(int i=0;i<ne;i++){
  int poseid,pointid,dim;double x,y,right,invvar,bf,fx,fy,cx,cy;
  std::cin>>poseid>>pointid>>dim>>x>>y>>right>>invvar>>bf>>fx>>fy>>cx>>cy;
  if(dim==2){
   auto edge=new ORB_SLAM3::EdgeSE3ProjectXYZ();edge->setVertex(0,optimizer.vertex(pointid+maximum+1));edge->setVertex(1,optimizer.vertex(poseid));edge->setMeasurement(Eigen::Vector2d(x,y));edge->setInformation(Eigen::Matrix2d::Identity()*invvar);edge->pCamera=new ORB_SLAM3::Pinhole(std::vector<float>{(float)fx,(float)fy,(float)cx,(float)cy});auto kernel=new RobustKernelHuber();kernel->setDelta(std::sqrt(5.991));edge->setRobustKernel(kernel);optimizer.addEdge(edge);edges.push_back(edge);
  }else{
   auto edge=new EdgeStereoSE3ProjectXYZ();edge->setVertex(0,optimizer.vertex(pointid+maximum+1));edge->setVertex(1,optimizer.vertex(poseid));edge->setMeasurement(Eigen::Vector3d(x,y,right));edge->setInformation(Eigen::Matrix3d::Identity()*invvar);edge->fx=fx;edge->fy=fy;edge->cx=cx;edge->cy=cy;edge->bf=bf;auto kernel=new RobustKernelHuber();kernel->setDelta(std::sqrt(7.815));edge->setRobustKernel(kernel);optimizer.addEdge(edge);edges.push_back(edge);
  }
 }
 optimizer.initializeOptimization();optimizer.computeActiveErrors();double before=optimizer.activeRobustChi2();int iterations=optimizer.optimize(10);optimizer.computeActiveErrors();
 std::cout<<std::setprecision(17)<<before<<" "<<optimizer.activeRobustChi2()<<" "<<iterations<<"\n";
 for(int id:poseids){auto vertex=static_cast<VertexSE3Expmap*>(optimizer.vertex(id));auto matrix=vertex->estimate().to_homogeneous_matrix();for(int r=0;r<4;r++)for(int c=0;c<4;c++)std::cout<<matrix(r,c)<<" ";std::cout<<"\n";}
 for(int id:pointids){auto vertex=static_cast<VertexSBAPointXYZ*>(optimizer.vertex(id+maximum+1));auto point=vertex->estimate();for(int i=0;i<3;i++)std::cout<<point[i]<<" ";std::cout<<"\n";}
 for(auto edge:edges)std::cout<<edge->chi2()<<" ";std::cout<<"\n";
}
