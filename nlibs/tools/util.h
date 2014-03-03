#ifndef UTIL_H_
#define UTIL_H_
#include <stdlib.h>
#include <utility>
#include <stdio.h>
#include <vector>
#include <time.h>
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
void arrayOutput(const char* msg, FILE* fp, const int datas[], int len);
void arrayOutput(const char* msg, FILE* fp, const double datas[], int len);
void arrayOutput(const char *msg, FILE* fp, const vector<int> &datas);
void arrayOutput(const char *msg, FILE* fp, const vector<double> &datas);
void prefixSumToCounts(const int prefixSum[], const int len, int *counts);
void arrayEqualPartition(int prefixSum[], const int n, const int nthreads, int ends[]);
void randomPermutationVector(int* &pvector, const int len);
int* permutationTranspose(const int P[], const int len);

#endif
