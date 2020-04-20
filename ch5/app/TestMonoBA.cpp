#include <iostream>
#include <random>
#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/edge_reprojection.h"
#include "backend/problem.h"

using namespace myslam::backend;
using namespace std;

/* 定义结构体：结构体语句定义了一个新的数据类型，拥有以下成员：
 * Frame : 保存每帧的姿态和观测
 */
struct Frame {
    Frame(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), qwc(R), twc(t) {}; // 构造函数初始化
    Eigen::Matrix3d Rwc;      // 矩阵表示旋转
    Eigen::Quaterniond qwc;   // 四元数表旋转
    Eigen::Vector3d twc;      
    /*【unordered_map 哈希表对应的容器。哈希表是根据关键码值而直接进行访问的数据结构，也就是说，它通过把关键码值映射到表中一个位置来访问记录，
    以加快查找的速度，这个映射函数叫做散列函数】 */ 
    unordered_map<int, Eigen::Vector3d> featurePerId; // 该帧观测到的特征以及特征id，
};

/**
 * @brief  产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 * @note   
 * @param  &cameraPoses: 相机位姿
 * @param  &points:  路标点
 * @retval None
 */
void GetSimDataInWordFrame(vector<Frame> &cameraPoses, vector<Eigen::Vector3d> &points) {
    int featureNums = 20;  // 特征数目，假设每帧都能观测到所有的特征
    int poseNums = 3;     // 相机数目

    double radius = 8;
    for (int n = 0; n < poseNums; ++n) {
        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()); // 角度和轴
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta)); // r*cos(theta) - r, r*sin(theta), 1*sin(2*theta)
        cameraPoses.push_back(Frame(R, t)); // 返回R t
    }

    // 随机数生成三维特征点
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal 。噪声分布；越大
    for (int j = 0; j < featureNums; ++j) {
        std::uniform_real_distribution<double> xy_rand(-4, 4.0); // 随机生成数的范围
        std::uniform_real_distribution<double> z_rand(4., 8.);

        Eigen::Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator)); // 随机生成3维特征点
        points.push_back(Pw);

        // 在每一帧上的观测量
        for (int i = 0; i < poseNums; ++i) {
            Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            Pc = Pc / Pc.z();               // 归一化图像平面
            Pc[0] += noise_pdf(generator);  // 高斯噪声
            Pc[1] += noise_pdf(generator);
            /*【C++标准程序库中凡是“必须返回两个值”的函数，也都会利用pair对象，pair可以将两个值视为一个单元；make_pair 无需写出型别，就可以生成一个pair对象】*/
            cameraPoses[i].featurePerId.insert(make_pair(j, Pc)); 
        }
    }
}

int main() {
    // 准备数据
    vector<Frame> cameras;
    vector<Eigen::Vector3d> points;
    GetSimDataInWordFrame(cameras, points); // 得到仿真数据（相机位姿和三维点）
    Eigen::Quaterniond qic(1, 0, 0, 0);
    Eigen::Vector3d tic(0, 0, 0);

    // 构建 problem
    Problem problem(Problem::ProblemType::SLAM_PROBLEM);

    // 所有 Pose
    vector<shared_ptr<VertexPose> > vertexCams_vec;         /*【shared_ptr智能指针，用于管理可以由多个智能指针共同拥有的动态分配对象，特别是，类型shared_ptr<T>用于管理T类型对象的所有权】 */
    for (size_t i = 0; i < cameras.size(); ++i) {
        shared_ptr<VertexPose> vertexCam(new VertexPose());         /*【new：1、开辟单变量地址空间 2、开辟数组空间】 */
        Eigen::VectorXd pose(7); // 实际参与的是6自由度，此处为3+4，为了方便用四元数表示旋转
        // 平移 四元数
        pose << cameras[i].twc, cameras[i].qwc.x(), cameras[i].qwc.y(), cameras[i].qwc.z(), cameras[i].qwc.w();
        vertexCam->SetParameters(pose);

        if(i < 2)
            vertexCam->SetFixed(); // 固定相机初始值

        problem.AddVertex(vertexCam);
        vertexCams_vec.push_back(vertexCam);

        
    }

    // 所有 Point 及 edge
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0, 1.);
    double noise = 0;
    vector<double> noise_invd;

    vector<shared_ptr<VertexInverseDepth> > allPoints;
    for (size_t i = 0; i < points.size(); ++i) {
        //假设所有特征点的起始帧为第0帧， 逆深度容易得到
        Eigen::Vector3d Pw = points[i];
        Eigen::Vector3d Pc = cameras[0].Rwc.transpose() * (Pw - cameras[0].twc); // 转到相机坐标
        noise = noise_pdf(generator);
        double inverse_depth = 1. / (Pc.z() + noise); // 逆深度 + 噪声
//        double inverse_depth = 1. / Pc.z();
        noise_invd.push_back(inverse_depth);

        // 初始化特征 vertex
        shared_ptr<VertexInverseDepth> verterxPoint(new VertexInverseDepth());
        VecX inv_d(1);                          /* 【VecX:定义动态向量】 */
        inv_d << inverse_depth;
        verterxPoint->SetParameters(inv_d);
        problem.AddVertex(verterxPoint);
        allPoints.push_back(verterxPoint);      // 点加入到求解器

        // 每个特征对应的投影误差, 第 0 帧为起始帧
        for (size_t j = 1; j < cameras.size(); ++j) {
            Eigen::Vector3d pt_i = cameras[0].featurePerId.find(i)->second; // 起始帧 0帧
            Eigen::Vector3d pt_j = cameras[j].featurePerId.find(i)->second; // 第j帧

            shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
            edge->SetTranslationImuFromCamera(qic, tic);

            std::vector<std::shared_ptr<Vertex> > edge_vertex;
            edge_vertex.push_back(verterxPoint);
            edge_vertex.push_back(vertexCams_vec[0]);
            edge_vertex.push_back(vertexCams_vec[j]);
            edge->SetVertex(edge_vertex);

            problem.AddEdge(edge); 
        }
    }

    problem.Solve(5); // 迭代5步
 
    std::cout << "\nCompare MonoBA results after opt..." << std::endl;
    for (size_t k = 0; k < allPoints.size(); k+=1) {
        std::cout << "after opt, point " << k << " : gt " << 1. / points[k].z() << " ,noise "
                  << noise_invd[k] << " ,opt " << allPoints[k]->Parameters() << std::endl;
    }
    std::cout<<"------------ pose translation ----------------"<<std::endl;
    for (int i = 0; i < vertexCams_vec.size(); ++i) {
        std::cout<<"translation after opt: "<< i <<" :"<< vertexCams_vec[i]->Parameters().head(3).transpose() << " || gt: "<<cameras[i].twc.transpose()<<std::endl;
    }
    /// 优化完成后，第一帧相机的 pose 平移（x,y,z）不再是原点 0,0,0. 说明向零空间发生了漂移。
    /// 解决办法： fix 第一帧和第二帧，固定 7 自由度。 或者加上非常大的先验值。

    problem.TestMarginalize();

    return 0;
}

