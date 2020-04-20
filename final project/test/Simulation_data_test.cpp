#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

string sData_path = "/home/why/Desktop/深蓝VIO课程内容/作业7/vins_sys_code/dataset/"; // 图像角点数据路径
string sConfig_path = "/home/why/Desktop/深蓝VIO课程内容/作业7/vins_sys_code/config/"; // 配置文件路径
std::shared_ptr<System> pSystem;

void PubImuData()
{
    string sim_Imu_data_file = sConfig_path + "imu_pose_noise_x3.txt";
    cout << "1 开始读取IMU数据: " << sim_Imu_data_file << endl;
    ifstream fsimImu;
    fsimImu.open(sim_Imu_data_file.c_str());
    if(!fsimImu.is_open())
    {
        cerr << "打开IMU失败! " << sim_Imu_data_file << endl;
        return;
    }

    std::string sim_Imu_line; //行
    double dStampNSec = 0.0;
    Vector3d vAcc;
    Vector3d vGyr;

    // Vector4d tmpQ;
    // Vector3d tmpP;
    while (std::getline(fsimImu, sim_Imu_line) && !sim_Imu_line.empty())
    {
        std::istringstream ssImuData(sim_Imu_line);
        // ssImuData >> dStampNSec >> tmpQ.x() >> tmpQ.y() >> tmpQ.z() >> tmpQ.w() >> tmpP.x() >> tmpP.y() >> tmpP.z() >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
        ssImuData >> dStampNSec >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
        pSystem->PubImuData(dStampNSec, vGyr, vAcc);
        usleep(15000);
    }
    fsimImu.close();
}

void readPoint(const string& filename, std::vector<cv::Point2f>& _points)
{
    ifstream f;
    f.open(filename.c_str());

    if(!f.is_open())
    {
        std::cerr << "打不开角点文件 " << std::endl;
        return;
    }
    float data[36][2] = { 0 };
    for(int i=0; i<36; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            f >> data[i][j];
        }
        cv::Point2f tmp=cv::Point2f(data[i][0],data[i][1]);
        _points.push_back(tmp);
    }
}


// 读取时间戳文件，和point对应的txt，all_point_n.txt
void PubPointData()
{
    string sim_point_time_file = sConfig_path + "time.txt"; // 时间戳路径
    cout << "1 开始打开角点时间戳文件: " << sim_point_time_file << endl;

    ifstream fsimPoint;
    fsimPoint.open(sim_point_time_file.c_str()); // 打开时间戳文件
    
    if(!fsimPoint.is_open())
    {
        cerr << "打开角点时间戳文件失败 " << sim_point_time_file << endl;
        return;
    }

    std::string sim_point_line; // 时间戳 行
    double dStampNsec;
    string sim_point_file_name; // Point文件的名字
    

    while(std::getline(fsimPoint, sim_point_line) && !sim_point_line.empty())
    {
        std::istringstream ssPointData(sim_point_line);
        ssPointData >> dStampNsec >> sim_point_file_name;
        string pointPath = sData_path + "keyframe/" + sim_point_file_name;

        std::vector<cv::Point2f> Points;
        //TODO： 读取point.txt文件里的角点。存到vector<cv::Point2f> prev_pts, cur_pts, forw_pts; 传入到Feature::readImage
        readPoint(pointPath, Points);
        // cout << "Points is " << Points[0].x << endl;
        // if(Points.empty())
        // {
        //     cerr << "points is empty! " << std::endl;
        //     return;
        // }
        pSystem->PubPointData(dStampNsec, Points);
        // cout << "points is " << Points << endl;
        usleep(100000); // 此处修改为相应的imu和camera对应的频率比
    }
    fsimPoint.close();
}



int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << "./sim_data_test 特征点文件路径 配置文件（包括参数、时间戳和imu数据）\n" << endl;
        return -1;
    }

    sData_path = argv[1];
	sConfig_path = argv[2];

    pSystem.reset(new System(sConfig_path)); 

    std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);
    std::thread thd_PubImuData(PubImuData);
    std::thread thd_PubPointData(PubPointData);
    std::thread thd_Draw(&System::Draw, pSystem);

    thd_PubImuData.join();
    thd_PubPointData.join();  
    thd_BackEnd.join();
	thd_Draw.join();
    std::cout << "end!!!!" << std::endl;
    return 0;
}