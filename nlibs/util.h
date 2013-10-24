#ifndef UTIL_H_
#define UTIL_H_
#include <stdlib.h>

#define MLMCL_PRUNE_A	(0.90) /* pruning parameter */
#define MLMCL_PRUNE_B	(2)	   /* pruning parameter */

double computeThreshold(double avg, double max);
double arrayMax(const double values[], const int count);
double arraySum(const double values[], const int count);
void arrayInflationR2(const double ivalues[], const int count, double ovalues[]);
double arrayThreshPrune(const double thresh, int* count, int indices[], double values[]);
#endif
