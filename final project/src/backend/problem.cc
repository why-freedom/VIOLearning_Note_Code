#include <iostream>
#include <fstream>

#include <thread>

#include <eigen3/Eigen/Dense>
#include <iomanip>
#include "backend/problem.h"
#include "utility/tic_toc.h"

// 编译选项
// 控制使用LM 或者Dogleg
const int alogrithm_option = 1;   // 0: LM,   1: Dogleg
// 控制是否使用加速, 以及加速方式
const int acc_option = 0;  // 0: Normal,non-acc,   1: OpenMP acc,   2: Multi Threads acc
// 控制LM算法中的IsGoodStep的策略(不影响Dogleg)
const int lm_strategy_option = 2;  // 0: 原策略;  1: 原策略;  2: 新策略(主选);   3: 新策略(不建议)  4: 论文中3rd Lambda策略
// 控制 LM 或者 Dogleg 算法的chi和lambda初始化选项
const int chi_lambda_init_option = 4;   // 0:Nielsen; 1:Levenberg;  2:Marquardt;  3:Quadratic;  4:Dogleg
   
#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace std;

// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) {
    std::ofstream f(name.c_str());
    f << matrix.format(CSVFormat);
}

namespace myslam {
namespace backend {
void Problem::LogoutVectorSize() {
    // LOG(INFO) <<
    //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
    //           " edges:" << edges_.size();
}

Problem::Problem(ProblemType problemType) :
    problemType_(problemType) {
    LogoutVectorSize();
    verticies_marg_.clear();
}

Problem::~Problem() {
//    std::cout << "Problem IS Deleted"<<std::endl;
    global_vertex_id = 0;
}

bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
        // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
        return false;
    } else {
        verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        if (IsPoseVertex(vertex)) {
            ResizePoseHessiansWhenAddingPose(vertex);
        }
    }
    return true;
}

void Problem::AddOrderingSLAM(std::shared_ptr<myslam::backend::Vertex> v) {
    if (IsPoseVertex(v)) {
        v->SetOrderingId(ordering_poses_);
        idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
        ordering_poses_ += v->LocalDimension();
    } else if (IsLandmarkVertex(v)) {
        v->SetOrderingId(ordering_landmarks_);
        ordering_landmarks_ += v->LocalDimension();
        idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
    }
}

void Problem::ResizePoseHessiansWhenAddingPose(shared_ptr<Vertex> v) {

    int size = H_prior_.rows() + v->LocalDimension();
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(v->LocalDimension()).setZero();
    H_prior_.rightCols(v->LocalDimension()).setZero();
    H_prior_.bottomRows(v->LocalDimension()).setZero();

}
void Problem::ExtendHessiansPriorSize(int dim)
{
    int size = H_prior_.rows() + dim;
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(dim).setZero();
    H_prior_.rightCols(dim).setZero();
    H_prior_.bottomRows(dim).setZero();
}

bool Problem::IsPoseVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPose") ||
            type == string("VertexSpeedBias");
}

bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPointXYZ") ||
           type == string("VertexInverseDepth");
}

bool Problem::AddEdge(shared_ptr<Edge> edge) {
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
    } else {
        // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
        return false;
    }

    for (auto &vertex: edge->Verticies()) {
        vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
    }
    return true;
}

vector<shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) {
    vector<shared_ptr<Edge>> edges;
    auto range = vertexToEdge_.equal_range(vertex->Id());
    for (auto iter = range.first; iter != range.second; ++iter) {

        // 并且这个edge还需要存在，而不是已经被remove了
        if (edges_.find(iter->second->Id()) == edges_.end())
            continue;

        edges.emplace_back(iter->second);
    }
    return edges;
}

bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) {
    //check if the vertex is in map_verticies_
    if (verticies_.find(vertex->Id()) == verticies_.end()) {
        // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
        return false;
    }

    // 这里要 remove 该顶点对应的 edge.
    vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
    for (size_t i = 0; i < remove_edges.size(); i++) {
        RemoveEdge(remove_edges[i]);
    }

    if (IsPoseVertex(vertex))
        idx_pose_vertices_.erase(vertex->Id());
    else
        idx_landmark_vertices_.erase(vertex->Id());

    vertex->SetOrderingId(-1);      // used to debug
    verticies_.erase(vertex->Id());
    vertexToEdge_.erase(vertex->Id());

    return true;
}

bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) {
    //check if the edge is in map_edges_
    if (edges_.find(edge->Id()) == edges_.end()) {
        // LOG(WARNING) << "The edge " << edge->Id() << " is not in the problem!" << endl;
        return false;
    }

    edges_.erase(edge->Id());
    return true;
}

// 负责负责进行LM或者Dogleg算法的选择和分发
bool Problem::Solve(int iterations){
    bool result = false;

    // To Do: 此处使用一个变量保存所使用的策略, 未来可以变为外部配置文件中的配置项
    int option = alogrithm_option;   
    switch(option) {  
        case 0: result = SolveLM(iterations);
                break;
        case 1: result = SolveDogleg(iterations);
                break;
        default:
                std::cerr << "Unkown solve option : "<< option << std::endl;
                result = false;
                break;
    }
    return result;
}

void Problem::SaveCostTime(std::string filename, double SolveTime, double HessianTime)
{
    std::ofstream save_pose;
    save_pose.setf(std::ios::fixed, std::ios::floatfield);
    save_pose.open(filename.c_str(), std::ios::app);
    // long int timeStamp = floor(time*1e9);
    save_pose << SolveTime << " "
              << HessianTime << std::endl;
    save_pose.close();
}

// Dogleg 方法=====================================================
// 1、设置参数： 初始值，信赖域上界，信赖域半径，\mu
// 2、寻找最优解：首先确定方向，再确定步长 
bool Problem::SolveDogleg(int iterations) {

    if (edges_.size() == 0 || verticies_.size() == 0) {
        std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
        return false;
    }

    TicToc t_solve;
    // 统计优化变量的维数，为构建 H 矩阵做准备
    SetOrdering();
    // 遍历edge, 构建 H 矩阵。里面有delta_x_初值
    MakeHessian(); 

    // 使用新的 Chi 和 Lambda 的初始化
    //ComputeLambdaInitLM();
    ComputeChiInitAndLambdaInit();
    
    // 尝试把 r 从1 增大到 1e4 来避免MH-05数据集上漂移的问题
    bool found = false;
    radius_ = 1e4;

    //bool stop = false;
    int iter = 0;
    const int numIterationsMax = 10;
    double last_chi_ = 1e20;

    while ( !found && (iter < numIterationsMax)) {
        std::cout << "iter: " << iter << " , currentChi= " << currentChi_ << " , radius= " << radius_ << std::endl;
        iter++;

        bool oneStepSuccess = false;
        int false_cnt = 0;
        while (!oneStepSuccess && false_cnt < 10)  // 不断尝试 Lambda, 直到成功迭代一步
        {
            // 计算alpha 和 h_gn 
            VecX auxVector1 = Hessian_ * b_;
            double alpha_ = b_.squaredNorm() / auxVector1.dot(b_);
            //alpha_ = b_.squaredNorm() / (b_.transpose()*Hessian_*b_);
            h_sd_ = alpha_ * b_;
            // To Do: 此处Hessian_比较大, 直接求逆很耗时, 可采用 Gauss-Newton法求解
            //h_gn_ = Hessian_.inverse() * b_;
            h_gn_ = Hessian_.ldlt().solve(b_);

            double h_sd_norm = h_sd_.norm();
            double h_gn_norm = h_gn_.norm();

            // 计算h_dl 步长
            if ( h_gn_norm <= radius_){
                h_dl_ = h_gn_;
            // 此处条件判断直接用了 h_sd_norm, 和论文的 alpha_*h_sd_norm不同
            }else if ( h_sd_norm >= radius_ ) { 
                // 此处条件判断 直接用了 h_sd_norm, 和论文的 alpha_*h_sd_norm不同
                h_dl_ = ( radius_ / h_sd_.norm() ) * h_sd_;
            } else {
                // 计算beta用于更新步长(此处a直接等于 h_sd_, 和论文的 alpha_* h_sd_ 有所不同)
                //VecX a = alpha_ * h_sd_;            
                VecX a  = h_sd_;
                VecX b = h_gn_;
                //double c = a.transpose() * (b - a);
                  double c = h_sd_.dot( b - a );
                if (c <= 0){
                    beta_ = ( -c + sqrt(c*c + (b-a).squaredNorm() * (radius_*radius_ - a.squaredNorm())) )
                             / (b - a).squaredNorm();
                }else{ 
                    beta_ = (radius_*radius_ - a.squaredNorm()) / (c + sqrt(c*c + (b-a).squaredNorm() 
                            * (radius_*radius_ - a.squaredNorm())));
                }
                assert(beta_ > 0.0 && beta_ < 1.0 && "Error while computing beta");
                //h_dl_ = alpha_ * h_sd_  + beta_ * (h_gn_ - alpha_ * h_sd_);
                // 此处 a 直接等于 h_sd_, 和论文的有所不同
                h_dl_= a + beta_ * ( b - a );
            } 
            delta_x_ = h_dl_;

            UpdateStates();
            oneStepSuccess = IsGoodStepInDogleg();
            // 后续处理，
            if(oneStepSuccess)
            {
                MakeHessian();
                false_cnt = 0;
            }
            else
            {
                false_cnt++;
                RollbackStates();
            }

        }
        iter++;

        if(last_chi_ - currentChi_ < 1e-5  || b_.norm() < 1e-5 )
        {
            std::cout << "Dogleg: find the right result. " << std::endl;
            found = true;
        }
        last_chi_ = currentChi_;
    }
    std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
    // 记录本次Hessian处理时长
    hessian_time_per_frame = t_hessian_cost_; 
    // 记录本次frame时长(包括hessian时长)
    time_per_frame = t_solve.toc();
    // 记录本frame的求解次数
    solve_count_per_frame = iter;
    
    SaveCostTime("costTime.txt", t_solve.toc(), t_hessian_cost_);
    t_hessian_cost_ = 0.;
    return true;
}

// Dogleg策略因子, 用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
bool Problem::IsGoodStepInDogleg(){
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->RobustChi2();
    }
    if (err_prior_.size() > 0)
        tempChi += err_prior_.norm();
    tempChi *= 0.5;          // 1/2 * err^2

    // 计算rho
    double scale=0.0;
    if(h_dl_ == h_gn_){
        scale = currentChi_;
    } else if(h_dl_ == radius_ * b_ / b_.norm()) {
        scale = radius_ * (2 * (alpha_ * b_).norm() - radius_) / (2 * alpha_);
    } else { 
         scale = 0.5 * alpha_ * pow( (1 - beta_), 2) * b_.squaredNorm() 
                    + beta_ * (2 - beta_) * currentChi_;
    }

    double nonLinearGain = currentChi_ - tempChi; 
    double linearGain = - double(delta_x_.transpose() * Hessian_ * delta_x_) + 2 * b_.dot(delta_x_); // 和g2o一致
    // if(scale < 1e-12)
    //     scale = 1e-12;
    // double rho_ = nonLinearGain / linearGain;
    double rho_ = nonLinearGain / scale;

    if (rho_ > 0.75 && isfinite(tempChi)) {
        radius_ = std::max(radius_, 3 * delta_x_.norm());
    }
    else if (rho_ < 0.25) {
        radius_ = std::max(radius_ / 4, 1e-7);
        // radius_ = 0.5 * radius_; // 论文中
    } else {
        // do nothing
    }

   if (rho_ > 0 && isfinite(tempChi)) {
        currentChi_ = tempChi;
        return true;
    } else {
        return false;
    }

    //以下为根据论文的策略
    // double rho = (currentChi_ - tempChi) / scale;
    // if(rho > 0 && isfinite(tempChi))
    // {
    //     delta_x_ = delta_x_ + h_dl_;
    //     currentChi_ = tempChi;
    //     return true;
    // }
    // if(rho > 3.0/4.0 && isfinite(tempChi)) 
    // {
    //     radius_ = max(radius_, 3*h_dl_.norm());
    //     return false;
    // }  
    // else if(rho < 1.0/4.0 && isfinite(tempChi)) 
    // {
    //     radius_ = radius_ / 2.0; 
    //     return false;
    // } else {
    //     return false;
    // }
}

// Chi 和 Lambda 初始化
void Problem::ComputeChiInitAndLambdaInit()
{
    currentChi_ = 0.0;
    for (auto edge: edges_) {
        // 在MakeHessian()中已经计算了edge.second->ComputeResidual()
        currentChi_ += edge.second->RobustChi2();
    }
    if (err_prior_.rows() > 0)
        currentChi_ += err_prior_.squaredNorm();
    currentChi_ *= 0.5;

    maxDiagonal_ = 0;
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        maxDiagonal_ = std::max(fabs(Hessian_(i, i)), maxDiagonal_);
    }
    maxDiagonal_ = std::min(5e10, maxDiagonal_);

    // 更新currentLambda选项
    int option = chi_lambda_init_option; // 0:Nielsen; 1:Levenberg;  2:Marquardt;  3:Quadratic;  4:Doglet
    switch (option) {
        case 0: // NIELSEN:
            ComputeLambdaInitLM_Nielsen();
            break;
        case 1: // LEVENBERG
            ComputeLambdaInitLM_Levenberg();
            break;
        case 2:  // MARQUARDT
            ComputeLambdaInitLM_Marquardt();
            break;
        case 3:  // QUADRATIC
            ComputeLambdaInitLM_Quadratic();
            break;
        case 4:  // DOGLEG
            ComputeLambdaInitDogleg();
            break;
        default:
            cout << "Please choose correct LM strategy in .ymal file: 0 Nielsen; 1 LevenbergMarquardt; 2 Quadratic" << endl;
            exit(-1);
            break;
    }
}

/// LM
void Problem::ComputeLambdaInitLM_Nielsen() {
    ni_ = 2.;
    
    stopThresholdLM_ = 1e-10 * currentChi_;          // 迭代条件为 误差下降 1e-6 倍

    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal_;
    std::cout << "currentLamba_: "<<currentLambda_<<", maxDiagonal: "<<maxDiagonal_<<std::endl;
}

/// LM
void Problem::ComputeLambdaInitLM_Levenberg() {
    currentLambda_ = 1e-2;

    lastLambda_ = currentLambda_;
//  std::cout << "currentLamba_: "<<maxDiagonal<<" "<<currentLambda_<<std::endl;
}

void Problem::ComputeLambdaInitLM_Marquardt() {
    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal_;

    //currentLambda_ = 1e-2;
//  std::cout << "currentLamba_: "<<maxDiagonal<<" "<<currentLambda_<<std::endl;
}

void Problem::ComputeLambdaInitLM_Quadratic() {
    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal_;
}

void Problem::ComputeLambdaInitDogleg() {
    currentLambda_ = 1e-7;
}


bool Problem::SolveLM(int iterations) {
    if (edges_.size() == 0 || verticies_.size() == 0) {
        std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
        return false;
    }

    TicToc t_solve;
    // 统计优化变量的维数，为构建 H 矩阵做准备
    SetOrdering();
    // 遍历edge, 构建 H 矩阵
    MakeHessian();
    // LM 初始化
    ComputeLambdaInitLM();
    // LM 算法迭代求解
    bool stop = false;
    int iter = 0;
    double last_chi_ = 1e20;

    while (!stop && (iter < iterations)) {
        std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
        bool oneStepSuccess = false;
        int false_cnt = 0;
        while (!oneStepSuccess && false_cnt < 10)  // 不断尝试 Lambda, 直到成功迭代一步
        {
            // setLambda
//            AddLambdatoHessianLM();
            // 第四步，解线性方程
            SolveLinearSystem();
            //
//            RemoveLambdaHessianLM();

            // 优化退出条件1： delta_x_ 很小则退出
//            if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10)
            // TODO:: 退出条件还是有问题, 好多次误差都没变化了，还在迭代计算，应该搞一个误差不变了就中止
//            if ( false_cnt > 10)
//            {
//                stop = true;
//                break;
//            }

            // 更新状态量
            UpdateStates();
            // 判断当前步是否可行以及 LM 的 lambda 怎么更新, chi2 也计算一下
            oneStepSuccess = IsGoodStepInLM();
            // 后续处理，
            if (oneStepSuccess) {
//                std::cout << " get one step success\n";

                // 在新线性化点 构建 hessian
                MakeHessian();
                // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
//                double b_max = 0.0;
//                for (int i = 0; i < b_.size(); ++i) {
//                    b_max = max(fabs(b_(i)), b_max);
//                }
//                // 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
//                stop = (b_max <= 1e-12);
                false_cnt = 0;
            } else {
                false_cnt ++;
                RollbackStates();   // 误差没下降，回滚
            }
        }
        iter++;

        // 优化退出条件3： currentChi_ 跟第一次的 chi2 相比，下降了 1e6 倍则退出
        // TODO:: 应该改成前后两次的误差已经不再变化
//        if (sqrt(currentChi_) <= stopThresholdLM_)
//        if (sqrt(currentChi_) < 1e-15)
        if(last_chi_ - currentChi_ < 1e-5)
        {
            std::cout << "sqrt(currentChi_) <= stopThresholdLM_" << std::endl;
            stop = true;
        }
        last_chi_ = currentChi_;
    }
    std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;

    // ----- new code start -----
    // 记录本次Hessian处理时长
    hessian_time_per_frame = t_hessian_cost_; 
    // 记录本次frame时长(包括hessian时长)
    time_per_frame = t_solve.toc();  
    // 记录本frame的求解次数
    solve_count_per_frame = iter;
    // ----- new code end -----    
    SaveCostTime("costTime.txt", t_solve.toc(), t_hessian_cost_);

    t_hessian_cost_ = 0.;
    return true;
}

bool Problem::SolveGenericProblem(int iterations) {
    return true;
}

void Problem::SetOrdering() {

    // 每次重新计数
    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;

    // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
    for (auto vertex: verticies_) {
        ordering_generic_ += vertex.second->LocalDimension();  // 所有的优化变量总维数

        if (problemType_ == ProblemType::SLAM_PROBLEM)    // 如果是 slam 问题，还要分别统计 pose 和 landmark 的维数，后面会对他们进行排序
        {
            AddOrderingSLAM(vertex.second);
        }

    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        // 这里要把 landmark 的 ordering 加上 pose 的数量，就保持了 landmark 在后,而 pose 在前
        ulong all_pose_dimension = ordering_poses_;
        for (auto landmarkVertex : idx_landmark_vertices_) {
            landmarkVertex.second->SetOrderingId(
                landmarkVertex.second->OrderingId() + all_pose_dimension
            );
        }
    }

//    CHECK_EQ(CheckOrdering(), true);
}

bool Problem::CheckOrdering() {
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        int current_ordering = 0;
        for (auto v: idx_pose_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }

        for (auto v: idx_landmark_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }
    }
    return true;
}

// 构造大H矩阵
void Problem::MakeHessian() {
    int option = acc_option; // 0: Normal,non-acc,   1: OpenMP acc,   2: Multi Threads acc
    switch (option) {
        // 非加速
        case 0: MakeHessianNormal();
                break;
        // openMP加速
        case 1: MakeHessianWithOpenMP();
                break;
        // 手工多线程加速
        case 2: MakeHessianWithMultiThreads();
                break;
    }
}


#ifdef USE_OPENMP
#pragma omp declare reduction (+: VecX: omp_out=omp_out+omp_in)\
     initializer(omp_priv=VecX::Zero(omp_orig.size()))
#pragma omp declare reduction (+: MatXX: omp_out=omp_out+omp_in)\
     initializer(omp_priv=MatXX::Zero(omp_orig.rows(), omp_orig.cols()))
#endif 
void Problem::MakeHessianNormal(){
    TicToc t_h;
    // 直接构造大的 H 矩阵
    ulong size = ordering_generic_;
    MatXX H(MatXX::Zero(size, size));
    VecX b(VecX::Zero(size));
 
    vector< shared_ptr<myslam::backend::Edge> > vec_edge_;
    int edgesSize = edges_.size();
    vec_edge_.reserve(edges_.size());
    for (auto edge: edges_) {
        vec_edge_.push_back(edge.second);
    }
        
    #ifdef USE_OPENMP
    omp_set_num_threads(12);
    Eigen::setNbThreads(1);
    #pragma omp parallel for reduction(+: H) reduction(+: b) 
    #endif

    // for (auto &edge: edges_) {
    for (int i = 0; i < edgesSize; i++) {
        
        auto edge = vec_edge_[i];
        // edge.second->ComputeResidual();
        // edge.second->ComputeJacobians();
        edge->ComputeResidual();
        edge->ComputeJacobians();

        // TODO:: robust cost
        // auto jacobians = edge.second->Jacobians();
        // auto verticies = edge.second->Verticies();
        auto jacobians = edge->Jacobians();
        auto verticies = edge->Verticies();
        // assert(jacobians.size() == verticies.size());

        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
            double drho;
            MatXX robustInfo(edge->Information().rows(),edge->Information().cols());
            edge->RobustInfo(drho,robustInfo);

            MatXX JtW = jacobian_i.transpose() * robustInfo;
            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];

                if (v_j->IsFixed()) continue;

                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                assert(v_j->OrderingId() != -1);
                MatXX hessian = JtW * jacobian_j;

                // 所有的信息矩阵叠加起来
                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    // 对称的下三角
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();

                }
            }
            b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edge->Information() * edge->Residual();
        }

    }
    Hessian_ = H;
    b_ = b;
    t_hessian_cost_ += t_h.toc();

    if(H_prior_.rows() > 0)
    {
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp = b_prior_;

        /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
        /// landmark 没有先验
        for (auto vertex: verticies_) {
            if (IsPoseVertex(vertex.second) && vertex.second->IsFixed() ) {
                int idx = vertex.second->OrderingId();
                int dim = vertex.second->LocalDimension();
                H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
            }
        }
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }
    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}

//  使用OpenMP加速
void Problem::MakeHessianWithOpenMP() {
    TicToc t_h;
    // 直接构造大的 H 矩阵
    ulong size = ordering_generic_;
    MatXX H(MatXX::Zero(size, size));
    VecX b(VecX::Zero(size));

    // TODO:: accelate, accelate, accelate
    //  ----- new code start -----
    // 由于edges_是map, 因此需要把id存起来,等下在for循环时可以直接用
    std::vector<unsigned long> edge_ids;
    for (auto& edge: edges_ ){
        // first 为key, second 为value
        edge_ids.push_back( edge.first );
    }

    //for (auto &edge: edges_) {
    // 由于openmp严格要求for循环下标必须是整数, 因此需要改写为如下形式
    #pragma omp parallel for num_threads(4) // ---------------------------------------------------
    for(unsigned int idx=0; idx < edges_.size(); idx++ ) {
        // 使用如下方法得到当前第 idx 个元素
        // 1. 使用第idx个位置上预先保存的id取到对应的edge
        auto edge = edges_[edge_ids[idx]];
        // 2. 遍历到第idx个元素(不建议)
        // auto it = edges_.begin();
        // for(int i=0; i<idx; i++ ){
        //     ++it;
        // }
        // auto edge = *it; 

        //edge->second->ComputeResidual();
        edge->ComputeResidual();
        //edge->second->ComputeJacobians();
        edge->ComputeJacobians();

        // TODO:: robust cost
        auto jacobians = edge->Jacobians();
        auto verticies = edge->Verticies();
        assert(jacobians.size() == verticies.size());

        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
            double drho;
            MatXX robustInfo(edge->Information().rows(),edge->Information().cols());
            edge->RobustInfo(drho,robustInfo);

            MatXX JtW = jacobian_i.transpose() * robustInfo;
            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];

                if (v_j->IsFixed()) continue;

                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                assert(v_j->OrderingId() != -1);
                MatXX hessian = JtW * jacobian_j;

                // 所有的信息矩阵叠加起来
                // 由于多线程对 H 的访问时不同块, 不会冲突, 可以不加 访问控制
                //#pragma omp critical 
                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    // 对称的下三角
                    //#pragma omp critical 
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                }
            }
            #pragma omp critical  // --------------------------------------------------------------
            b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edge->Information() * edge->Residual();
        }
    }
    Hessian_ = H;
    b_ = b;
    t_hessian_cost_ += t_h.toc();
    // ----- new code end -----

    if(H_prior_.rows() > 0)
    {
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp = b_prior_;

        /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
        /// landmark 没有先验
        for (auto vertex: verticies_) {
           if (IsPoseVertex(vertex.second) && vertex.second->IsFixed() ) {
                int idx = vertex.second->OrderingId();
                int dim = vertex.second->LocalDimension();
                H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
            }
        }
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }

    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}

// 线程函数, 负责处理一部分的edges, 然后拼装到大矩阵H中和b中
//  被MakeHessianWithMultiThreads()调用
void Problem::thdDoEdges(int start, int end) {
    // 从 start 下标开始, 一直到 end 下标, 闭区间, 进行处理
    auto it = edges_.begin();
    for(int i=0; i<=end; i++ ){
        if( i < start ) {
            ++it;
            continue;
        };

        auto edge = *it; 
        edge.second->ComputeResidual();
        edge.second->ComputeJacobians();    
        // TODO:: robust cost
        auto jacobians = edge.second->Jacobians();
        auto verticies = edge.second->Verticies();
        assert(jacobians.size() == verticies.size());

        for (size_t i = 0; i < verticies.size(); ++i) {
            //std::cout << "debug:  in for verticies of i: " << i << std::endl;

            auto v_i = verticies[i];
            if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
            double drho;
            MatXX robustInfo(edge.second->Information().rows(),edge.second->Information().cols());
            edge.second->RobustInfo(drho,robustInfo);

            MatXX JtW = jacobian_i.transpose() * robustInfo;
            for (size_t j = i; j < verticies.size(); ++j) {
                //std::cout << "  debug:  in for verticies of j: " << j << std::endl;
                auto v_j = verticies[j];

                if (v_j->IsFixed()) continue;

                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                assert(v_j->OrderingId() != -1);
                MatXX hessian = JtW * jacobian_j;

                // 所有的信息矩阵叠加起来
                //H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                // 由于不同线程访问 H 矩阵的不同块, 因此不加访问锁
                //m_mu.lock();
                m_H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    // 对称的下三角
                    //H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    m_H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                }
                //m_mu.unlock();
            }
            //b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edge.second->Information() * edge.second->Residual();
            m_mu.lock();
            m_b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edge.second->Information() * edge.second->Residual();
            m_mu.unlock();   
        }                
    }
}
//  使用多线程加速
void Problem::MakeHessianWithMultiThreads(){
   TicToc t_h;
    // 直接构造大的 H 矩阵
    ulong size = ordering_generic_;
    //MatXX H(MatXX::Zero(size, size));
    //VecX b(VecX::Zero(size));
    m_H.setZero(size,  size );
    m_b.setZero(size );
 
    // 建立 thd_num 个线程 
    int thd_num = 4;
    // edges_ 均匀等分为  thd_num 份
    int start=0, end=0;
    cout << " Total edges: " << edges_.size() << std::endl;
    for(int i=1; i<=thd_num; i++ ) {
        end = edges_.size() * i / thd_num; 
        std::thread t = std::thread(std::mem_fn(&Problem::thdDoEdges), this, start, end-1);
        t.join();
        start = end ;
    }

    Hessian_ = m_H;
    b_ = m_b;
    t_hessian_cost_ += t_h.toc();

    if(H_prior_.rows() > 0)
    {
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp = b_prior_;

        /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
        /// landmark 没有先验
        for (auto vertex: verticies_) {
            if (IsPoseVertex(vertex.second) && vertex.second->IsFixed() ) {
                int idx = vertex.second->OrderingId();
                int dim = vertex.second->LocalDimension();
                H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
            }
        }
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }

    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}

/*
 * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
 */
void Problem::SolveLinearSystem() {
    if (problemType_ == ProblemType::GENERIC_PROBLEM) {
        // PCG solver
        MatXX H = Hessian_;
        for (unsigned int i = 0; i < Hessian_.cols(); ++i) {
            H(i, i) += currentLambda_;
        }
        // delta_x_ = PCGSolver(H, b_, H.rows() * 2);
        delta_x_ = H.ldlt().solve(b_);

    } else {

//        TicToc t_Hmminv;
        // step1: schur marginalization --> Hpp, bpp
        int reserve_size = ordering_poses_;
        int marg_size = ordering_landmarks_;
        MatXX Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
        VecX bpp = b_.segment(0, reserve_size);
        VecX bmm = b_.segment(reserve_size, marg_size);

        // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块为1维的，则直接为对角线的倒数，这里可以加速
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
        // TODO:: use openMP
        for (auto landmarkVertex : idx_landmark_vertices_) {
            int idx = landmarkVertex.second->OrderingId() - reserve_size;
            int size = landmarkVertex.second->LocalDimension();
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
        }

        MatXX tempH = Hpm * Hmm_inv;
        H_pp_schur_ = Hessian_.block(0, 0, ordering_poses_, ordering_poses_) - tempH * Hmp;
        b_pp_schur_ = bpp - tempH * bmm;

        // step2: solve Hpp * delta_x = bpp
        VecX delta_x_pp(VecX::Zero(reserve_size));

        for (ulong i = 0; i < ordering_poses_; ++i) {
            H_pp_schur_(i, i) += currentLambda_;              // LM Method
        }

        // TicToc t_linearsolver;
        delta_x_pp =  H_pp_schur_.ldlt().solve(b_pp_schur_);//  SVec.asDiagonal() * svd.matrixV() * Ub;    
        delta_x_.head(reserve_size) = delta_x_pp;
        // std::cout << " Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;

        // step3: solve Hmm * delta_x = bmm - Hmp * delta_x_pp;
        VecX delta_x_ll(marg_size);
        delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
        delta_x_.tail(marg_size) = delta_x_ll;

//        std::cout << "schur time cost: "<< t_Hmminv.toc()<<std::endl;
    }

}

void Problem::UpdateStates() {

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->BackUpParameters();    // 保存上次的估计值

        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        VecX delta = delta_x_.segment(idx, dim);
        vertex.second->Plus(delta);
    }

    // update prior
    if (err_prior_.rows() > 0) {
        // BACK UP b_prior_
        b_prior_backup_ = b_prior_;
        err_prior_backup_ = err_prior_;

        /// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
        /// \delta x = Computes the linearized deviation from the references (linearization points)
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);       // update the error_prior
        err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);

//        std::cout << "                : "<< b_prior_.norm()<<" " <<err_prior_.norm()<< std::endl;
//        std::cout << "     delta_x_ ex: "<< delta_x_.head(6).norm() << std::endl;
    }

}

void Problem::RollbackStates() {

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->RollBackParameters();
    }

    // Roll back prior_
    if (err_prior_.rows() > 0) {
        b_prior_ = b_prior_backup_;
        err_prior_ = err_prior_backup_;
    }
}

/// LM
void Problem::ComputeLambdaInitLM() {
    ni_ = 2.;
    currentLambda_ = -1.;
    currentChi_ = 0.0;

    for (auto edge: edges_) {
        currentChi_ += edge.second->RobustChi2();
    }
    if (err_prior_.rows() > 0)
        currentChi_ += err_prior_.norm();
    currentChi_ *= 0.5;

    stopThresholdLM_ = 1e-10 * currentChi_;          // 迭代条件为 误差下降 1e-6 倍

    double maxDiagonal = 0;
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
    }

    maxDiagonal = std::min(5e10, maxDiagonal);
    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal;
//        std::cout << "currentLamba_: "<<maxDiagonal<<" "<<currentLambda_<<std::endl;
}

void Problem::AddLambdatoHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) += currentLambda_;
    }
}

void Problem::RemoveLambdaHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) -= currentLambda_;
    }
}



bool Problem::IsGoodStepInLM() {
    double scale = 0;
//    scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
//    scale += 1e-3;    // make sure it's non-zero :)
    scale = 0.5* delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
    scale += 1e-6;    // make sure it's non-zero :)

    // recompute residuals after update state
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->RobustChi2();
    }
    if (err_prior_.size() > 0)
        tempChi += err_prior_.norm();
    tempChi *= 0.5;          // 1/2 * err^2

    // To Do: 此处使用一个变量保存所使用的策略, 未来可以变为外部配置文件中的配置项
    int option = lm_strategy_option;  // 0: 原策略;  1: 原策略;  2: 新策略(主选);   3: 新策略(不建议)
    switch(option) {
        case 0: {
                double rho = (currentChi_ - tempChi) / scale;
                if (rho > 0 && isfinite(tempChi))   // last step was good, 误差在下降
                {
                    double alpha = 1. - pow((2 * rho - 1), 3);
                    alpha = std::min(alpha, 2. / 3.);
                    double scaleFactor = (std::max)(1. / 3., alpha);
                    currentLambda_ *= scaleFactor;
                    ni_ = 2;
                    currentChi_ = tempChi;
                    return true;
                } else {
                    currentLambda_ *= ni_;
                    ni_ *= 2;
                    return false;
                }
                break;
            }

        case 1: {
                double frac1 = delta_x_.transpose() * b_;
                double alpha = frac1 / (((tempChi - currentChi_)/2.) + 2 * frac1);
                RollbackStates();
                delta_x_ *= alpha;
                UpdateStates();
                double scale = 0;
                scale = delta_x_.transpose() * (currentLambda_ * delta_x_ + b_ );
                scale += 1e-3;

                double rho = (currentChi_ - tempChi ) / scale;
                if( rho > 0 && isfinite(tempChi)) {
                    currentLambda_ = std::max(currentLambda_ /(1+alpha), 1e-7 );
                    currentChi_= tempChi;
                    return true;
                }else{
                    currentLambda_ = currentLambda_ + abs(currentChi_ - tempChi ) / (2.*alpha);
                    return false;
                }
                break;
            }
        case 2: {
                //参见论文"The Levenberg-Marquardt algorithm for nonlinear 
                //         least squares curve-fitting problems, Henri P. Gavin"
                // h =  delta_x_*b 
                // diff = currentChi_ - tempChi;
                // alpha = h / (0.5*diff + h )
                // rho > 0, lambda = max( lambda/(1+alpha), 1.e-7)
                // rho <=0, lambda = lambda + abs(diff*0.5/alpha)
                double rho = (currentChi_ - tempChi) / scale;
                double  h = delta_x_.transpose() * b_;
                double  diff = currentChi_ - tempChi;
                double  alpha_ = h / (0.5*diff + h);
                if ( rho > 0 && isfinite(tempChi) ){
                    currentLambda_ = std::max(currentLambda_/(1+alpha_), 1.e-7 );
                    currentChi_ = tempChi;
                    return true;
                }else if( rho <=0 && isfinite(tempChi) ){
                    currentLambda_ = currentLambda_ + std::abs(diff*0.5/alpha_);
                    currentChi_ = tempChi;
                    return true;
                } else {
                    // do nothing
                    return false;
                }
                break;
            }
        case 3: {
                // rho < 0.25 , lambda = lambda*2.0
                // rho > 0.75 , lambda = lambda/3.0
                double rho = (currentChi_ - tempChi) / scale;
                if ( rho < 0.25 && isfinite(tempChi)) {
                    currentLambda_ *= 2.0;
                    currentChi_ = tempChi;
                    return true;
                }else if ( rho > 0.75 && isfinite(tempChi) ) {
                    currentLambda_ /= 3.0;
                    currentChi_ = tempChi;
                    return true;    
                } else {
                    // do nothing
                    return false;
                }
                break;
             }
        //why添加：  
        case 4:{
                double maxDiagonal = 0;
                ulong size = Hessian_.cols();
                assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
                for (ulong i = 0; i < size; ++i) {
                    maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal); //Hessian矩阵对角元素最大值
                }

                scale = delta_x_.transpose() * (currentLambda_ * maxDiagonal * delta_x_ + b_); // L(0)-L(\delta{x}_lm)=1/2...
                scale += 1e-3;
                double rho = (currentChi_ - tempChi) / scale;
                if(rho > 0 && isfinite(tempChi))
                {
                    double scaleFactor = (std::max)(currentLambda_/9, 10e-7);
                    currentLambda_ = scaleFactor;
                    currentChi_ = tempChi;
                    return true;
                } else{
                    double scaleFactor = (std::min)(currentLambda_/11, 10e+7);
                    currentLambda_ = scaleFactor;
                    return false;
                }
        }
    }
}

/** @brief conjugate gradient with perconditioning
 *
 *  the jacobi PCG method
 *
 */
VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
    assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();
    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w = A * p;
    double r0z0 = r0.dot(z0);
    double alpha = r0z0 / p.dot(w);
    VecX r1 = r0 - alpha * w;
    int i = 0;
    double threshold = 1e-6 * r0.norm();
    while (r1.norm() > threshold && i < n) {
        i++;
        VecX z1 = M_inv * r1;
        double r1z1 = r1.dot(z1);
        double belta = r1z1 / r0z0;
        z0 = z1;
        r0z0 = r1z1;
        r0 = r1;
        p = belta * p + z1;
        w = A * p;
        alpha = r1z1 / p.dot(w);
        x += alpha * p;
        r1 -= alpha * w;
    }
    return x;
}

/*
 *  marg 所有和 frame 相连的 edge: imu factor, projection factor
 *  如果某个landmark和该frame相连，但是又不想加入marg, 那就把改edge先去掉
 *
 */
bool Problem::Marginalize(const std::vector<std::shared_ptr<Vertex> > margVertexs, int pose_dim) {

    SetOrdering();
    /// 找到需要 marg 的 edge, margVertexs[0] is frame, its edge contained pre-intergration
    std::vector<shared_ptr<Edge>> marg_edges = GetConnectedEdges(margVertexs[0]);

    std::unordered_map<int, shared_ptr<Vertex>> margLandmark;
    // 构建 Hessian 的时候 pose 的顺序不变，landmark的顺序要重新设定
    int marg_landmark_size = 0;
//    std::cout << "\n marg edge 1st id: "<< marg_edges.front()->Id() << " end id: "<<marg_edges.back()->Id()<<std::endl;
    for (size_t i = 0; i < marg_edges.size(); ++i) {
//        std::cout << "marg edge id: "<< marg_edges[i]->Id() <<std::endl;
        auto verticies = marg_edges[i]->Verticies();
        for (auto iter : verticies) {
            if (IsLandmarkVertex(iter) && margLandmark.find(iter->Id()) == margLandmark.end()) {
                iter->SetOrderingId(pose_dim + marg_landmark_size);
                margLandmark.insert(make_pair(iter->Id(), iter));
                marg_landmark_size += iter->LocalDimension();
            }
        }
    }
//    std::cout << "pose dim: " << pose_dim <<std::endl;
    int cols = pose_dim + marg_landmark_size;
    /// 构建误差 H 矩阵 H = H_marg + H_pp_prior
    MatXX H_marg(MatXX::Zero(cols, cols));
    VecX b_marg(VecX::Zero(cols));
    int ii = 0;
    for (auto edge: marg_edges) {
        edge->ComputeResidual();
        edge->ComputeJacobians();
        auto jacobians = edge->Jacobians();
        auto verticies = edge->Verticies();
        ii++;

        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            double drho;
            MatXX robustInfo(edge->Information().rows(),edge->Information().cols());
            edge->RobustInfo(drho,robustInfo);

            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];
                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                MatXX hessian = jacobian_i.transpose() * robustInfo * jacobian_j;

                assert(hessian.rows() == v_i->LocalDimension() && hessian.cols() == v_j->LocalDimension());
                // 所有的信息矩阵叠加起来
                H_marg.block(index_i, index_j, dim_i, dim_j) += hessian;
                if (j != i) {
                    // 对称的下三角
                    H_marg.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
                }
            }
            b_marg.segment(index_i, dim_i) -= drho * jacobian_i.transpose() * edge->Information() * edge->Residual();
        }

    }
        std::cout << "edge factor cnt: " << ii <<std::endl;

    /// marg landmark
    int reserve_size = pose_dim;
    if (marg_landmark_size > 0) {
        int marg_size = marg_landmark_size;
        MatXX Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hpm = H_marg.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
        VecX bpp = b_marg.segment(0, reserve_size);
        VecX bmm = b_marg.segment(reserve_size, marg_size);

        // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块为1维的，则直接为对角线的倒数，这里可以加速
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
        // TODO:: use openMP
        for (auto iter: margLandmark) {
            int idx = iter.second->OrderingId() - reserve_size;
            int size = iter.second->LocalDimension();
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
        }

        MatXX tempH = Hpm * Hmm_inv;
        MatXX Hpp = H_marg.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
        bpp = bpp - tempH * bmm;
        H_marg = Hpp;
        b_marg = bpp;
    }

    VecX b_prior_before = b_prior_;
    if(H_prior_.rows() > 0)
    {
        H_marg += H_prior_;
        b_marg += b_prior_;
    }

    /// marg frame and speedbias
    int marg_dim = 0;

    // index 大的先移动
    for (int k = margVertexs.size() -1 ; k >= 0; --k)
    {

        int idx = margVertexs[k]->OrderingId();
        int dim = margVertexs[k]->LocalDimension();
//        std::cout << k << " "<<idx << std::endl;
        marg_dim += dim;
        // move the marg pose to the Hmm bottown right
        // 将 row i 移动矩阵最下面
        Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
        Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
        H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
        H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

        // 将 col i 移动矩阵最右边
        Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
        Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
        H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
        H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

        Eigen::VectorXd temp_b = b_marg.segment(idx, dim);
        Eigen::VectorXd temp_btail = b_marg.segment(idx + dim, reserve_size - idx - dim);
        b_marg.segment(idx, reserve_size - idx - dim) = temp_btail;
        b_marg.segment(reserve_size - dim, dim) = temp_b;
    }

    double eps = 1e-8;
    int m2 = marg_dim;
    int n2 = reserve_size - marg_dim;   // marg pose
    Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
            (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
                              saes.eigenvectors().transpose();

    Eigen::VectorXd bmm2 = b_marg.segment(n2, m2);
    Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
    Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
    Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);
    Eigen::VectorXd brr = b_marg.segment(0, n2);
    Eigen::MatrixXd tempB = Arm * Amm_inv;
    H_prior_ = Arr - tempB * Amr;
    b_prior_ = brr - tempB * bmm2;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_prior_);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd(
            (saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    Jt_prior_inv_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    err_prior_ = -Jt_prior_inv_ * b_prior_;

    MatXX J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    H_prior_ = J.transpose() * J;
    MatXX tmp_h = MatXX( (H_prior_.array().abs() > 1e-9).select(H_prior_.array(),0) );
    H_prior_ = tmp_h;

    // std::cout << "my marg b prior: " <<b_prior_.rows()<<" norm: "<< b_prior_.norm() << std::endl;
    // std::cout << "    error prior: " <<err_prior_.norm() << std::endl;

    // remove vertex and remove edge
    for (size_t k = 0; k < margVertexs.size(); ++k) {
        RemoveVertex(margVertexs[k]);
    }

    for (auto landmarkVertex: margLandmark) {
        RemoveVertex(landmarkVertex.second);
    }

    return true;

}

}
}






