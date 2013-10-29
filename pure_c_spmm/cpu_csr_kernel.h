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

const int LEVEL1_DCACHE_LINESIZE = 64;
struct thread_data_t {
  double* x;
  bool* xb;
  int *iJC;
  char pad_data[LEVEL1_DCACHE_LINESIZE];
  void init(const int n) {
    x = (double*)malloc(n * sizeof(double) + LEVEL1_DCACHE_LINESIZE);
    iJC = (int*)x;
    xb = (bool*)calloc(n + LEVEL1_DCACHE_LINESIZE, sizeof(bool));
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
#endif
