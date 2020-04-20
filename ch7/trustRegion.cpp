#include "trustRegion.h"
TrustRegion::TrustRegion(double (*_func)(vector<double>), vector<double> (* _derivative)(vector<double>), vector<vector<double> >(* _heissan)(vector<double>)):func(_func), derivative(_derivative), heissan(_heissan)
{
	//below are some initial points
	radMax = 100;
	radius = 1;
	eta = 0.125;
}

// 设置参数
void TrustRegion::setParameter(double _radMax, double _radius, double _eta)
{
	radMax = _radMax;
	radius = _radius;
	eta = _eta;
}


// 首先确定方向，再确定步长
vector<double> TrustRegion::searchMin(vector<double> initPoint)
{
	int step = 0;
	vector<double> point = initPoint;
	while (true) //begin the loop
	{
		Dog_leg solver(derivative(point), heissan(point), radius);
		vector<double> pk = solver.getDirection();
		if (getLength(pk) < eps) //the search direction is so small, we just return the point directly
			return point;
		double rho = getRho(point, pk);
		if (rho < 1.0 / 4.0)
			radius = radius / 4;
		else
			if (rho > 3.0 / 4.0 && getLength(pk) == radius)
				radius = min (2 * radius, radMax);
		if (rho > eta)
		{
			//update the point
			int dimension = point.size();
			for (int i = 0; i < dimension; i++)
				point[i] += pk[i];
		}
		step++;
		cout << step << ":" << point[0] << "," << point[1] << endl;
	}
}

// 下降比
double TrustRegion::getRho(vector<double> point, vector<double> pk)
{
	vector<double> newPoint;
	int dimension = point.size();
	for (int i = 0; i < dimension; i++)
		newPoint.push_back(point[i] + pk[i]);
	double nominator = func(point) - func(newPoint);
	vector<double> deriVec = derivative(point);
	vector<vector<double> > heiVec = heissan(point);
	double denominator = 0;
	for (int i = 0; i < dimension; i++)
		denominator -= deriVec[i] * pk[i];
	for (int i = 0; i < dimension; i++)
		for (int j = 0; j < dimension; j++)
			denominator -= pk[i] * heiVec[i][j] * pk[j] / 2;

	return nominator / denominator;
}
