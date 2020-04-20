//
// Created by hyj on 18-11-11.
//
#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

using namespace Eigen;
struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),qwc(R),twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;    // 这帧图像观测到的特征坐标
};
int main()
{

    int poseNums = 10;
    double radius = 8;
    double fx = 1.;
    double fy = 1.;
    std::vector<Pose> camera_pose;
    for(int n = 0; n < poseNums; ++n ) {
        double theta = n * 2 * M_PI / ( poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_pose.push_back(Pose(R,t));
    }

    // 随机数生成 1 个 三维特征点
    std::default_random_engine generator;
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz);
    // 这个特征从第三帧相机开始被观测，i=3
    int start_frame_id = 3;
    int end_frame_id = poseNums;
    for (int i = start_frame_id; i < end_frame_id; ++i) {
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose(); // to camera from world 旋转
        Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc); // 平移

        double x = Pc.x();
        double y = Pc.y();
        double z = Pc.z();

        camera_pose[i].uv = Eigen::Vector2d(x/z,y/z);// 每次观测，归一化平面坐标
    }
    
    /// TODO::homework; 请完成三角化估计深度的代码
    // 遍历所有的观测数据，并三角化
    Eigen::Vector3d P_est;           // 结果保存到这个变量
    P_est.setZero();

    /* your code begin */
    int n = end_frame_id - start_frame_id;// 观测数

    Eigen::MatrixXd D(2*n, 4); // 矩阵D初始化 2n×4
    D.setZero();
    int k=0; // D行数
    Eigen::Matrix<double, 3, 4> pose[poseNums]; // 投影矩阵
    for(int i = start_frame_id; i < end_frame_id; ++i)
    {
        double uk = camera_pose[i].uv(0);
        double vk = camera_pose[i].uv(1);
        
        pose[i].block<3, 3>(0, 0) = camera_pose[i].Rwc.transpose();
        pose[i].block<3, 1>(0, 3) = -camera_pose[i].Rwc.transpose() * camera_pose[i].twc;
        D.row(k++) = uk * pose[i].row(2) - pose[i].row(0);
        D.row(k++) = vk * pose[i].row(2) - pose[i].row(1);

        // Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        // Eigen::Vector3d Pc = -camera_pose[i].Rwc.transpose() * camera_pose[i].twc;
        
        // Eigen::Vector4d Pk_1, Pk_2, Pk_3; // 投影矩阵 3*4 ，World系到Camera系
        // Pk_1 << camera_pose[i].Rwc.row(0).transpose(), Pc(0);
        // Pk_2 << camera_pose[i].Rwc.row(1).transpose(), Pc(1);
        // Pk_3 << camera_pose[i].Rwc.row(2).transpose(), Pc(2);

        // D.row(k++) = uk * Pk_3.transpose() - Pk_1.transpose();
        // D.row(k++) = vk * Pk_3.transpose() - Pk_2.transpose();

    }
    Eigen::Matrix4d DTD = D.transpose() * D;
    // SVD分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(DTD, ComputeThinU | ComputeThinV);
    Eigen::MatrixXd V, S, U;
    V = svd.matrixV();
    U = svd.matrixU();
    S = svd.singularValues();  // 为 \Theta对角线元素
    std::cout<<"DTD :\n"<< DTD <<std::endl;
    std::cout<<"U :\n"<< U <<std::endl;
    std::cout<<"S :\n"<< S <<std::endl;
    std::cout<<"V :\n"<< V <<std::endl;

    P_est << U(0,3)/U(3,3), U(1,3)/U(3,3), U(2,3)/U(3,3);
    /* your code end */  
    std::cout <<"ground truth: \n"<< Pw.transpose() <<std::endl;
    std::cout <<"your result: \n"<< P_est.transpose() <<std::endl;
    return 0;
}
