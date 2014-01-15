#ifndef CPU_CSR_KERNEL_H_
#define CPU_CSR_KERNEL_H_
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#ifdef profiling
#include "tools/ntimer.h"
#endif
#include <iostream>
#include "tools/qmalloc.h"
#include "tools/stats.h"

const int LEVEL1_DCACHE_LINESIZE = 64;
struct thread_data_t {
  double* x;
  bool* xb;
  int* index;
  char pad_data[LEVEL1_DCACHE_LINESIZE];
  void init(const int n) {
    x = (double*)qmalloc(n * sizeof(double) + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
    //xb = (bool*)qcalloc(n + LEVEL1_DCACHE_LINESIZE, sizeof(bool), __FUNCTION__, __LINE__);
    xb = (bool*)qmalloc(n + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
    index = (int*)qmalloc(n * sizeof(int) + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
    //memset(index, -1, n * sizeof(int) + LEVEL1_DCACHE_LINESIZE);
  }

  void init(double *x, bool* xb, int *index) {
    this->x = x;
    this->xb = xb;
    this->index = index;
  }

  thread_data_t() {
    x = NULL;
    xb = NULL;
    index = NULL;
  }

  thread_data_t(const int n) {
    init(n);
  }

  ~thread_data_t() {
    free(xb);
    free(x);
    free(index);
    xb = NULL;
    x = NULL;
    index = NULL;
  }
};

long long getSpMMFlops(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        const int m, const int k, const int n);

long spmmFootPrints(const int IA[], const int JA[],
    const int IB[], const int IC[],
    const int m, long *footPrintSum);

void sequential_CSR_SpMM(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n);

void sequential_CSR_IC_nnzC(const int IA[], const int JA[],
        const int IB[], const int JB[],
        const int m, const int n, bool xb[],
        int* IC, int& nnzC);

void omp_CSR_SpMM(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
thread_data_t* allocateThreadDatas(int nthreads, int n);
void freeThreadDatas(thread_data_t* thread_datas, int nthreads);
void static_omp_CSR_SpMM(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
void omp_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride);
void omp_CSR_IC_nnzC_Wrapper(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t thread_datas[],
    int* IC, int& nnzC, const int stride);
void dynamic_omp_CSR_IC_nnzC_footprints(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, int* footPrints, const int stride);
void static_omp_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride, const int ends[], const int tid);
/*void processCRowI(
    //x and xb are used for temp use only and will have the same value when back.
    //xb must be all zeros before calling this functions.
    double x[], bool* xb,
    //IAi and IAi1 are starting and ending postions of A's row i in JA array.
    const int IAi, const int IAi1, const int JA[], const double A[],
        const int IB[], const int JB[], const double B[],
        const int ICi, int* JC, double* C);*/
int processCRowI(double x[], bool* xb,
    const int iAnnz, const int iJA[], const double iA[],
        const int IB[], const int JB[], const double B[],
        int* iJC, double* iC);

//inline int indexProcessCRowI(int *restrict index, // index array must be initilized with -1
#ifdef __CUDACC__
inline int indexProcessCRowI(int *index,
    const int iAnnz, const int iJA[], const double iA[],
    const int IB[], const int JB[], const double B[],
    int* iJC, double* iC) {
#else
inline int indexProcessCRowI(int *restrict index,
    const int iAnnz, const int iJA[], const double iA[],
    const int IB[], const int JB[], const double B[],
    int* restrict iJC, double* restrict iC) {
#endif
  if (iAnnz == 0) {
    return 0;
  }
  int ip = -1;
  int jp = 0;
  int j = iJA[jp];
  for(int tp = IB[j]; tp < IB[j + 1]; ++tp) {
    int t = JB[tp];
    iJC[++ip] = t;
    index[t] = ip;
    iC[ip] = iA[jp] * B[tp];
  }
  for(int jp = 1; jp < iAnnz; ++jp) {
    int j = iJA[jp];
#pragma unroll(2)
    for(int tp = IB[j]; tp < IB[j + 1]; ++tp) {
      int t = JB[tp];
      if(index[t] == -1) {
        iJC[++ip] = t;
        index[t] = ip;
        iC[ip] = iA[jp] * B[tp];
      } else {
        iC[index[t]] += iA[jp] * B[tp];
      }
      // This hack will remove if condition but it will make program slightly slow due to more operations.
      // This may worth a try on Xeon Phi machines.
      // int f = index[t] >> 31;
      // ip += f & 1;
      // index[t] += f & (ip + 1);
      // iJC[index[t]] = t;
      // iC[index[t]] += iA[jp] * B[tp];
    }
  }
  ++ip;
  for(int vp = 0; vp < ip; ++vp) {
    int v = iJC[vp];
    index[v] = -1;
  }
  return ip;
}

void omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
void cilk_CSR_RMCL_OneStep(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
void static_omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
void static_fair_CSR_RMCL_OneStep(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
void hybrid_omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
#endif
