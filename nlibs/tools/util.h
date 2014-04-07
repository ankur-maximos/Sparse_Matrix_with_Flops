#ifndef UTIL_H_
#define UTIL_H_
#include <stdlib.h>
#include <utility>
#include <stdio.h>
#include <vector>
#include <time.h>
#include "tools/macro.h"
using namespace std;

#define MLMCL_PRUNE_A	(0.90) /* pruning parameter */
#define MLMCL_PRUNE_B	(2)	   /* pruning parameter */

Value computeThreshold(Value avg, Value max);
Value arrayMax(const Value values[], const int count);
Value arraySum(const Value values[], const int count);
pair<Value, Value> arrayMaxSum(const Value values[], const int count);
void arrayInflationR2(const Value ivalues[], const int count, Value ovalues[]);
Value arrayThreshPruneNormalize(const Value thresh, const int rindices[], const Value rvalues[],
    int* count, int indices[], Value values[]);
void arrayOutput(const char* msg, FILE* fp, const int datas[], int len);
void arrayOutput(const char* msg, FILE* fp, const Value datas[], int len);
void arrayOutput(const char *msg, FILE* fp, const vector<int> &datas);
void arrayOutput(const char *msg, FILE* fp, const vector<Value> &datas);
void prefixSumToCounts(const int prefixSum[], const int len, int *counts);
void arrayEqualPartition(int prefixSum[], const int n, const int nthreads, int ends[]);
int* randomPermutationVector(const int len);
int* permutationTranspose(const int P[], const int len);
#endif
