#ifndef CPU_CSR_KERNEL_H_
#define CPU_CSR_KERNEL_H_
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#ifdef profiling
#include "ntimer.h"
#endif
#include <iostream>
#include "tools/qmalloc.h"

const int LEVEL1_DCACHE_LINESIZE = 64;
struct thread_data_t {
  double* x;
  bool* xb;
  int *iJC;
  char pad_data[LEVEL1_DCACHE_LINESIZE];
  void init(const int n) {
    x = (double*)qmalloc(n * sizeof(double) + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
    iJC = (int*)x;
    xb = (bool*)qcalloc(n + LEVEL1_DCACHE_LINESIZE, sizeof(bool), __FUNCTION__, __LINE__);
  }

  thread_data_t() {
    x = NULL;
    iJC = NULL;
    xb = NULL;
  }

  thread_data_t(const int n) {
    init(n);
  }

  ~thread_data_t() {
    free(xb);
    free(x);
    xb = NULL;
    x = NULL;
    iJC = NULL;
  }
};

long long getSpMMFlops(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        const int m, const int k, const int n);

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
        const int m, const int k, const int n);
thread_data_t* allocateThreadDatas(int nthreads, int n);
void freeThreadDatas(thread_data_t* thread_datas, int nthreads);
void omp_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t thread_datas[],
    int* IC, int& nnzC);
void omp_CSR_IC_nnzC_Wrapper(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t thread_datas[],
    int* IC, int& nnzC);
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
void omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas);
#endif
