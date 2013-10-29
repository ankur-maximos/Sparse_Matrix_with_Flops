#ifndef UTIL_H_
#define UTIL_H_
#include <stdlib.h>
#include <utility>
using namespace std;

#define MLMCL_PRUNE_A	(0.90) /* pruning parameter */
#define MLMCL_PRUNE_B	(2)	   /* pruning parameter */

double computeThreshold(double avg, double max);
double arrayMax(const double values[], const int count);
double arraySum(const double values[], const int count);
pair<double, double> arrayMaxSum(const double values[], const int count);
void arrayInflationR2(const double ivalues[], const int count, double ovalues[]);
double arrayThreshPruneNormalize(const double thresh, const int rindices[], const double rvalues[],
    int* count, int indices[], double values[]);
#endif
