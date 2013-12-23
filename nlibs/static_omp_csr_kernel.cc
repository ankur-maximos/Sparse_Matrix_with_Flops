#include <omp.h>
#include <iostream>
#include "cpu_csr_kernel.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include "tools/prefixSum.h"
using namespace std;

long spmmFootPrints(const int IA[], const int JA[],
    const int IB[], const int IC[],
    const int m,
    long *footPrintSum) {
  long footPrints = 0;
  footPrintSum[0] = 0;
  for (int i = 0; i < m; ++i) {
    long row_flops = 0;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      long Brow_j_nnz = IB[j + 1] - IB[j];
      row_flops += Brow_j_nnz;
    }
    footPrints += row_flops + IC[i + 1] - IC[i] + 1;
    footPrintSum[i + 1] = footPrints;
  }
  return footPrints;
}

inline int footPrintsCrowiCount(const int i, const int IA[], const int JA[], const int IB[], const int JB[], int iJC[], bool xb[], int &footPrints) {
  if (IA[i] == IA[i + 1]) {
    return 0;
  }
  int count = -1;
  int vp = IA[i];
  int v = JA[vp];
  footPrints = 0;
  for (int kp = IB[v]; kp < IB[v+1]; ++kp) {
    int k = JB[kp];
    iJC[++count] = k;
    xb[k] = true;
  }
  footPrints += IB[v + 1] - IB[v];
  for (int vp = IA[i] + 1; vp < IA[i + 1]; ++vp) {
    int v = JA[vp];
    for (int kp = IB[v]; kp < IB[v+1]; ++kp) {
      int k = JB[kp];
      if(xb[k] == false) {
        iJC[++count] = k;
        xb[k] = true;
      }
    }
    footPrints += IB[v + 1] - IB[v];
  }
  ++count;
  for(int jp = 0; jp < count; ++jp) {
    int j = iJC[jp];
    xb[j] = false;
  }
  footPrints += count + 1;
  return count;
}

/*
 * omp_CSR_IC_nnzC reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void static_omp_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, int* footPrints, const int stride) {
  int *iJC = (int*)thread_data.index;
  bool *xb = thread_data.xb;
  memset(xb, 0, n);
#pragma omp for schedule(dynamic)
  for (int it = 0; it < m; it += stride) {
    int up = it + stride < m ? it + stride : m;
    for (int i = it; i < up; ++i) {
      IC[i] = footPrintsCrowiCount(i, IA, JA, IB, JB, iJC, xb, footPrints[i]);
    }
  }
#pragma omp barrier
  noTileOmpPrefixSum(IC, IC, m);
  noTileOmpPrefixSum(footPrints, footPrints, m);
#pragma omp single
  {
    nnzC = IC[m];
  }
}

void static_omp_CSR_SpMM(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
  IC = (int*)malloc((m + 1) * sizeof(int));
  int* footPrints = (int*)malloc((m + 1) * sizeof(int));
  static int ends[65];
#pragma omp parallel firstprivate(stride)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    static_omp_CSR_IC_nnzC(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, footPrints, stride);
#pragma omp barrier
#pragma omp single
    {
      double now = time_in_mill_now();
      //spmmFootPrints(IA, JA, IB, IC, m, footPrints);
      arrayEqualPartition(footPrints, m, nthreads, ends);
      //std::cout << "time passed for just partition " << time_in_mill_now() - now << std::endl;
      //std::cout << "time passed for footPrints and partition " << time_in_mill_now() - now << std::endl;
    }
#pragma omp master
    {
      JC = (int*)malloc(sizeof(int) * nnzC);
      C = (double*)malloc(sizeof(double) * nnzC);
    }
    double *x = thread_datas[tid].x;
    int *index = thread_datas[tid].index;
    memset(index, -1, n * sizeof(int));
#pragma omp barrier
    int low = ends[tid];
    int high = ends[tid + 1];
    for (int i = low; i < high; ++i) {
      indexProcessCRowI(index,
          IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
          IB, JB, B,
          JC + IC[i], C + IC[i]);
    }
  }
}

