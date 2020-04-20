#ifndef dog_leg_h
#define dog_leg_h
#include <vector>
#include <math.h>
using namespace std;
const double eps = 1e-8;
class Dog_leg
{
	private:
	//radius, derivative, heissan are the only three things we need right now
	double radius;
	vector<double> derivative;
	vector<vector<double> > heissan;

	//blow are some helper variables
	vector<double> pu;
	vector<double> pb;
	double t;

	//below are some helper functions
	void getPu();
	void getPb();
	public:
	Dog_leg(vector<double>  deri, vector<vector<double> > hei, double rad);
	vector<double> getDirection();
};
//below are some helper functions
//probably there are some generous methods, but here I just hard code it for the case of dimension 2
vector<vector<double> > getInverse(vector<vector<double> > & heissan);
double getLength(vector<double> inputVec);
#endif


