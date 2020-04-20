#include  "dogleg.h"
#include <iostream>
#include <string>
Dog_leg::Dog_leg(vector<double>  deri, vector<vector<double> > hei, double rad): derivative(deri), heissan(hei), radius(rad){} 

//in this function, we can find the direction of the 
vector<double> Dog_leg::getDirection() // 下降方向
{
	getPu();
	getPb();
	if (getLength(pb) <= radius) //if the minimum is already inside the circle
	{
		t = 2;
		return pb;
	}

	int dimension = pb.size();

	double puLen = getLength(pu);
	if (puLen >= radius) //courner case 
	{
		t = radius / puLen;
		vector<double> pk; // 下降方向Pk
		for (int i = 0; i < dimension; i++)
			pk.push_back(pu[i] * radius / puLen);
		return pk;
	}

	double a = 0, b = 0, c = 0; //the coefficients for quadratic function
	//below we calculate the coefficients
	for (int i = 0; i < dimension; i++)
	{
		a += pow((pb[i] - pu[i]), 2);
		b += 2 * (2 * pu[i] - pb[i]) * (pb[i] - pu[i]);
		c += pow((2 * pu[i] - pb[i]), 2);
	}
	c -= radius * radius;
	//the degree 2 coefficient must exist
	// 三种情况统一的表达式。
	double t1 = (-b+sqrt(b * b - 4 * a * c)) / (2 * a);
	double t2 = (-b-sqrt(b * b - 4 * a * c)) / (2 * a);
	if (t1 > 1 && t1 < 2)
		t = t1;
	else if (t2 > 1 && t2 < 2)
		t = t2;
	vector<double> pk;
	for (int i = 0; i < dimension; i++)
		pk.push_back(pu[i] + (t - 1) *(pb[i] - pu[i]));
	return pk;
}

// Pu和Pb表示两个全局最优点，根据全局最优点是否落在信赖域中，来决定下降方向
// 求Pu
void Dog_leg::getPu()
{
	int dimension = derivative.size();
	double nominator = 0;
	for (int i = 0; i < dimension; i++)
		nominator += derivative[i] * derivative[i];
	double denominator = 0;
	for (int i = 0; i < dimension; i++)
		for (int j = 0; j < dimension; j++)
			denominator += derivative[i] * heissan[i][j] * derivative[j];
	double lambda = (0 - nominator) / denominator;
	for (int i = 0; i < dimension; i++)
	{
		pu.push_back(lambda * derivative[i]);
		// std::cout << "pu is " << pu[i] << std::endl;
	} 
}
// 求Pb
void Dog_leg::getPb()
{
	vector<vector<double> > inverseHei = getInverse(heissan);
	int dimension = derivative.size(); // 一阶导维度
	for (int i = 0; i < dimension; i++)
	{
		pb.push_back(0);
		for (int j = 0; j < dimension; j++)
			pb[i] -= inverseHei[i][j] * derivative[j]; // - H^-1 * b, 其中 b = -J^T * f 
			// std::cout << "pb is " << pb[i] << std::endl;
	}
	
}
// 求海森矩阵的逆
vector<vector<double> > getInverse(vector<vector<double> > & heissan)
{
	double a = heissan[0][0];
	double b = heissan[0][1];
	double c = heissan[1][0];
	double d = heissan[1][1];
	double denominator = a * d - b * c;
	vector<vector<double> > invHeissan;
	vector<double> row0;
	vector<double> row1;
	row0.push_back(d / denominator);
	row0.push_back(-b / denominator);
	row1.push_back(-c / denominator);
	row1.push_back( a / denominator);
	invHeissan.push_back(row0);
	invHeissan.push_back(row1);
	return invHeissan;
}
// 步长
double getLength(vector<double> inputVec) 
{
	double length = 0;
	int dimension = inputVec.size();
	for (int i = 0; i < dimension; i++)
		length += inputVec[i] * inputVec[i];
	return sqrt(length);
}
