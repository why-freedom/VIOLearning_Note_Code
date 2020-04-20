#ifndef trust_region_h
#define trust_region_h
#include <iostream>
#include <vector>
#include <math.h>
#include "dogleg.h"
using namespace std;
class TrustRegion
{
	private:
	double (* func)(vector<double>);
	vector<double> (*derivative)(vector<double>);
	vector<vector<double> > (*heissan)(vector<double>);
	
	double radMax, radius, eta;
	//below we have some helper functions
	double getRho(vector<double> point, vector<double> pk);
	public:
	TrustRegion(double (*_func)(vector<double>), vector<double> (* _derivative)(vector<double>), vector<vector<double> >(* _heissan)(vector<double>));
	void setParameter(double _radMax, double _radius, double _eta);
	//this function searchs for the minPoint from initPoint
	vector<double> searchMin(vector<double> initPoint);
};
extern double getLength(vector<double> inputVec);
extern const double eps;  
#endif
