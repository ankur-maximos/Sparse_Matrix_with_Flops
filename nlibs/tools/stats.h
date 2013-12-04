#ifndef STATS_H_
#define STATS_H_
#include <vector>
#include <cstdio>
using namespace std;

void pushToStats(const long stat, std::vector<int> &stats);
void outputStats(const std::vector<int> &stats);
std::vector<int> flopsStats(const int IA[], const int JA[], const double A[], const int nnzA,
    const int IB[], const int JB[], const double B[], const int nnzB,
    const int m, const int k, const int n);
#endif
