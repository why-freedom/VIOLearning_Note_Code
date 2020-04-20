#include <cstdlib>
#include "trustRegion.h"
#include "dogleg.h"
double rosen(vector<double> point);
vector<double> rosenDeriv(vector<double> point);
vector<vector<double> > rosenHei(vector<double> point);

int main(int argc, char * argv[])
{
	// 五个参数？？
	vector<double> inputVec; // 初始x0坐标[ , ]
	inputVec.push_back(atof(argv[1]));   
	inputVec.push_back(atof(argv[2]));
	double radMax = atof(argv[3]);  // Delta_hat  信赖域上界
	double radius = atof(argv[4]);  // Delta0     信赖域半径
	double eta = atof(argv[5]);     // eta        \mu 的值

	TrustRegion trustRegion(rosen, rosenDeriv, rosenHei); // 信赖域实例化(原式，一阶导，二阶导)

	trustRegion.setParameter(radMax, radius, eta); // 设置参数
	vector<double> finalPoint = trustRegion.searchMin(inputVec); // 寻找最优解
	cout << finalPoint[0] << "," <<  finalPoint[1] << endl;
	return 0;
}

// 被求解方程
double rosen(vector<double> point)
{
	return 100 * pow(point[1] - point[0] * point[0], 2) + pow(1 - point[0], 2);
}
// 求一阶导，雅可比
vector<double> rosenDeriv(vector<double> point) 
{
	vector<double> returnVec;
	returnVec.push_back(-400 * point[0] * point[1] + 400 * pow(point[0], 3) + 2 * point[0] -2);
	returnVec.push_back(200 * (point[1] - point[0] * point[0]));
	return returnVec;
}
// 海森矩阵，二阶导，J^T * \Sigma^-1 * J
vector<vector<double> > rosenHei(vector<double> point)
{
	vector<vector<double> > returnVec;
	vector<double> vec1, vec2;
	vec1.push_back(-400 * point[1] + 1200 * point[0] * point[0] + 2);
	vec1.push_back(-400 * point[0]);
	vec2.push_back(-400 * point[0]);
	vec2.push_back(200);
	returnVec.push_back(vec1);
	returnVec.push_back(vec2);
	return returnVec;
}
