#include <math.h>
#include <iostream>

using std::cerr;
using std::endl;

double log_zero = -1. / 0.;
double one = 0.;

double log_add(double x, double y)
{
  if (x == log_zero) return y;
  else 
    if (y == log_zero) return x;
    else
      {
	double max_val; 
	if (x > y) 
	  max_val = x;
	else 
	  max_val = y;
	double maxed_x = x - max_val;
	double maxed_y = y - max_val;
	return log (exp(maxed_x) + exp(maxed_y)) + max_val;
      }
}

double B_i[] = {1, -1. / 2., 1. / 6., 0., - 1. / 30., 0., 1. / 42., 0., - 1. / 30., 0., 5. / 66., 0., -691. / 2730., 0., 7. / 6., 0., -3617. / 510, 0., 43867. / 798., 0., - 174611. / 330., 0., 854513./138.};
int B_length = 16;

double half_log_two_pi = 0.5 * log (2. * 3.14159);

//Calculate n!  
//Approximation = Stirling's series
double approx_log_factorial(int n)
{
  if (n == 0)
    return 0.;
  else
    {
      double z = (double) (n + 1);
      double ret = half_log_two_pi + (z - 0.5) * log(z) - z;
      for (int i = 1; i <= B_length / 2; i++)
	ret += B_i[i*2] / ((double) 2*i * (2*i - 1)) / pow(z, (double) 2*i - 1);
      return ret;
    }
}

//Calculate n choose i
double approx_log_choose(int n, int i)
{
  return approx_log_factorial(n) - approx_log_factorial(i) - approx_log_factorial(n-i);
}

//Calculate: n choose i * p^i (1-p)^(n-i)
double log_binom_coeff(int n, int i, double p)
{
  if (p == 0. || p == 1.) 
    if (p == 0.) 
      if (i == 0) return 0.; 
      else return log_zero;
    else 
      if (i == n) return 0.;
      else return log_zero;
  else
    {
      double fi = (double) i;
      return fi * log (p) + ((double) n - fi) * log (1. - p) + approx_log_choose(n,i);
    } 
}

double too_tiny = 0.00000000001;

//Calculate CDF 
double CDF(int n, int i, double p, double sum)
{
  if (i < 0) return sum;
  else 
    {
      double ret = log_add(log_binom_coeff(n, i, p), sum);
      if ( ret - sum > too_tiny ) return CDF(n, i-1, p, ret);
      else return ret;
    }
}

//Search for upper or lower bounds
double upper_bound_rec(double target, int n, int i, double min, double max)
{
  if (max - min <= too_tiny) return max;
  else
    {
      double mid = (max + min) / 2;
      double value = CDF (n, i, mid, log_zero);
      if (value > target) return upper_bound_rec(target, n, i, mid, max);
      else return upper_bound_rec(target, n, i, min, mid);
    }
}

//The interfaces
//confidence = probability of bound failure.
//n = the number of examples
//i = the observed empirical error
double upper_bound(double confidence, int n, int i)
{
  if (i > n || i < 0 || n < 1) cerr << "invalid example or error count" << endl;
  else 
    if (confidence > 1. || confidence < 0.) cerr << "invalid confidence" << endl;
    else return upper_bound_rec(log(confidence), n, i, (double) i / (double) n, 1.);
  return 1.;
}

double lower_bound(double confidence, int n, int i)
{
  return 1. - upper_bound(confidence, n, n - i);
}
