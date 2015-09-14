#include "binomial.h"
#include <algorithm>
#include <iostream>

using std::cerr;
using std::cout;
using std::max;

int main(int argc, char *argv[])
{
  if (argc < 6)
    {
      cerr << argv[0] << " <allowed failure probability> <sample size A> <number of successes A> <sample size B> <number of successes B>\n";
      return 1;
    }
  else
    {
      double a_ub, b_ub, a_lb, b_lb;
      a_ub = upper_bound(atof(argv[1]) / 4., atoi(argv[2]), atoi(argv[3]));
      a_lb = lower_bound(atof(argv[1]) / 4., atoi(argv[2]), atoi(argv[3]));
      b_ub = upper_bound(atof(argv[1]) / 4., atoi(argv[4]), atoi(argv[5]));
      b_lb = lower_bound(atof(argv[1]) / 4., atoi(argv[4]), atoi(argv[5]));

      cout << "[" << max(a_lb - b_ub, b_lb - a_ub) << ", " << max(a_ub - b_lb, b_ub - a_lb) << "]\n";

      return 0;
    }
}
