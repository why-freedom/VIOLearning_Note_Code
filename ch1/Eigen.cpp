#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace std;
/* 1、定义R 及 q
 * 2、利用计算出来的w对R q更新.增量为w=[0.01,0.02,0.03]T
 * R <- R*exp(w^)
 * q <- q*[1,1/2*w]T
*/

int main()
{
    // 设置R q,假设初始旋转为绕z轴旋转90度
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d(0,0,1)).toRotationMatrix();
    Eigen::Quaterniond q = Eigen::Quaterniond(R);

    cout << "R is = " << endl << R << endl;
    cout << "q is = " << q.coeffs().transpose() << endl; // 实部在后，虚部在前

    // w是计算出来的增量.轴角形式=(v，theta),v为单位向量,theta是向量的模
    Eigen::Vector3d w(0.01, 0.02, 0.03);

    // w的模=m
    double m = sqrt( w(0)*w(0)+w(1)*w(1)+w(2)*w(2) );
    // 旋转向量转换成旋转矩阵
    // 此处等价于【w的指数映射】【罗德里格公式（旋转向量->旋转矩阵）】
    Eigen::Matrix3d w_ = Eigen::AngleAxisd( m, Eigen::Vector3d(0.01/m, 0.02/m, 0.03/m) ).toRotationMatrix();

    /**************** 此处用指数映射计算(不推介但可以用)： *************/
    // 【注】Eigen里边的exp()函数是对每个元素求exp，不可用在此处，此处通过matlab的expm函数算出w_hat的指数映射
    Eigen::Matrix3d w_hat;
    w_hat << 0, -w(2), w(1),
            w(2), 0, -w(0),
            -w(1), w(0), 0;
    Eigen::Matrix3d w_hat_exp;
    w_hat_exp <<    0.9994,   -0.0299,    0.0201,
                    0.0301,    0.9995,   -0.0097,
                    -0.0198,    0.0103,    0.9998;

    Eigen::Matrix3d R_exp = R * w_hat_exp;
    cout << "R_update with exp is =" << endl << R_exp << endl;
    /************************************************************/

    // 使用R方式存储，更新R
    R = R * w_;
    cout << endl <<  "R_update with Rodrigues's Formula is =" << endl << R << endl;

    // 使用q方式储存
    Eigen::Quaterniond q_ = Eigen::Quaterniond (1, 0.5*w(0), 0.5*w(1), 0.5*w(2));
    // 更新q
    q = q * q_;
    q.normalize(); // 归一化

    // 四元数转化为旋转矩阵
    Eigen::Matrix3d q2R = q.toRotationMatrix();
    cout << endl << "R_update form q_update is = " << endl << q2R << endl;

    // 作差比较精度
    Eigen::Matrix3d diff_exp = R_exp - q2R;
    cout << endl << "diff_ between R & q with exp is = " << endl << diff_exp << endl;

    Eigen::Matrix3d diff_rod = R - q2R;
    cout << endl << "diff_ between R & q with Rodrigues's Formula is = " << endl << diff_rod << endl;
    return 0;
}
