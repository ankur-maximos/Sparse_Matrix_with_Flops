#include <omp.h>
#include <iostream>
#include "tools/ntimer.h"
#include "cpu_csr_kernel.h"
#include "tools/util.h"
#include "tools/prefixSum.h"
using namespace std;

thread_data_t* allocateThreadDatas(int nthreads, int n) {
  thread_data_t* thread_datas = (thread_data_t*)calloc(nthreads, sizeof(thread_data_t));
#ifdef AGGR
  double *xs = (double*)qmalloc((n * sizeof(double) + LEVEL1_DCACHE_LINESIZE) * nthreads, __FUNCTION__, __LINE__);
  bool *xbs = (bool*)qcalloc((n + LEVEL1_DCACHE_LINESIZE) * nthreads, sizeof(bool), __FUNCTION__, __LINE__);
  int *indices = (int*)qmalloc((n * sizeof(int) + LEVEL1_DCACHE_LINESIZE) * nthreads, __FUNCTION__, __LINE__);
  memset(indices, -1, (n * sizeof(int) + LEVEL1_DCACHE_LINESIZE) * nthreads);
#endif
  for(int i = 0; i < nthreads; i++) {
#ifndef AGGR
    thread_datas[i].init(n);
#else
    double *x = (double*)((char*)xs + i * (n * sizeof(double) + LEVEL1_DCACHE_LINESIZE));
    bool *xb = (bool*)((char*)xbs + i * (n + LEVEL1_DCACHE_LINESIZE));
    int *index = (int*)((char*)indices + i * (n * sizeof(int) + LEVEL1_DCACHE_LINESIZE));
    thread_datas[i].init(x, xb, index);
#endif
  }
  return thread_datas;
}

void freeThreadDatas(thread_data_t* thread_datas, int nthreads) {
#ifndef AGGR
  for(int i = 0; i < nthreads; i++) {
    thread_datas[i].~thread_data_t();
  }
#else
  free(thread_datas[0].x);
  free(thread_datas[0].xb);
  free(thread_datas[0].index);
  free(thread_datas);
#endif
}

inline int cRowiCount(const int i, const int IA[], const int JA[], const int IB[], const int JB[], int iJC[], bool xb[]) {
  if (IA[i] == IA[i + 1]) {
    return 0;
  }
  int count = -1;
  int vp = IA[i];
  int v = JA[vp];
  for (int kp = IB[v]; kp < IB[v+1]; ++kp) {
    int k = JB[kp];
    iJC[++count] = k;
    xb[k] = true;
  }
  for (int vp = IA[i] + 1; vp < IA[i + 1]; ++vp) {
    int v = JA[vp];
    for (int kp = IB[v]; kp < IB[v+1]; ++kp) {
      int k = JB[kp];
      if(xb[k] == false) {
        iJC[++count] = k;
        xb[k] = true;
      }
    }
  }
  ++count;
  for(int jp = 0; jp < count; ++jp) {
    int j = iJC[jp];
    xb[j] = false;
  }
  return count;
}

const int nthreads = 8;
void omp_CSR_IC_nnzC_Wrapper(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t thread_datas[],
    int* IC, int& nnzC, const int stride) {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    omp_CSR_IC_nnzC(IA, JA, IB, JB,
    m, n, thread_datas[tid],
    IC, nnzC, stride);
  }
}

/*
 * omp_CSR_IC_nnzC reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void omp_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride) {
  int *iJC = (int*)thread_data.index;
  bool *xb = thread_data.xb;
  memset(xb, 0, n);
#pragma omp for schedule(dynamic)
  for (int it = 0; it < m; it += stride) {
    int up = it + stride < m ? it + stride : m;
    for (int i = it; i < up; ++i) {
      IC[i] = cRowiCount(i, IA, JA, IB, JB, iJC, xb);
    }
  }
#pragma omp barrier
  noTileOmpPrefixSum(IC, IC, m);
  //ompPrefixSum(IC, IC, m);
#pragma omp single
  {
    nnzC = IC[m];
  }
}

inline int processCRowI(double x[], bool* xb,
    const int iAnnz, const int iJA[], const double iA[],
        const int IB[], const int JB[], const double B[],
        int* iJC, double* iC) {
  int ip = 0;
  for(int jp = 0; jp < iAnnz; ++jp) {
    int j = iJA[jp];
    for(int tp = IB[j]; tp < IB[j + 1]; ++tp) {
      int t = JB[tp];
      if(xb[t] == false) {
        iJC[ip++] = t;
        xb[t] = true;
        x[t] = iA[jp] * B[tp];
      } else
        x[t] += iA[jp] * B[tp];
    }
    //x[JB[IBj : IBj1 - IBj]] += iA[jp] * B[IBj : IBj1 - IBj];
    //xb[JB[IBj : IBj1 - IBj]] = true;
  }
  for(int vp = 0; vp < ip; ++vp) {
    int v = iJC[vp];
    iC[vp] = x[v];
    x[v] = 0;
    xb[v] = false;
  }
  return ip;
}

int indexProcessCRowI(int *restrict index, // index array must be initilized with -1
    const int iAnnz, const int iJA[], const double iA[],
        const int IB[], const int JB[], const double B[],
        int* restrict iJC, double* restrict iC) {
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
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
    IC = (int*)calloc(m + 1, sizeof(int));
    int *rowsNnz = (int*)calloc(m + 1, sizeof(int));
#pragma omp parallel firstprivate(stride) //num_threads(1)
    {
      int tid = omp_get_thread_num();
      omp_CSR_IC_nnzC(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, stride);
#pragma omp master
      {
        JC = (int*)malloc(sizeof(int) * nnzC);
        C = (double*)malloc(sizeof(double) * nnzC);
      }
      double *x = thread_datas[tid].x;
      bool *xb = thread_datas[tid].xb;
      int *index = thread_datas[tid].index;
      memset(index, -1, n * sizeof(int));
#pragma omp barrier
#pragma omp for schedule(dynamic)
      for (int it = 0; it < m; it += stride) {
        int up = it + stride < m ? it + stride : m;
        for (int i = it; i < up; ++i) {
          double *cValues = C + IC[i];
          int *cColInd = JC + IC[i];
          //processCRowI(x, xb,
          indexProcessCRowI(index,
              IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
              IB, JB, B,
              cColInd, cValues);
          int count = IC[i + 1] - IC[i];
          arrayInflationR2(cValues, count, cValues);
          pair<double, double> maxSum = arrayMaxSum(cValues, count);
          double rmax = maxSum.first, rsum = maxSum.second;
          double thresh = computeThreshold(rsum / count, rmax);
          arrayThreshPruneNormalize(thresh, cColInd, cValues,
              &count, cColInd, cValues);
          rowsNnz[i] = count;
        }
      }
    }
    int top = rowsNnz[0];
    for (int i = 1; i < m; ++i) {
      int up = IC[i] + rowsNnz[i];
      const int preTop = top;
#pragma simd
      for (int j = IC[i]; j < up; ++j) {
        JC[top] = JC[j];
        C[top++] = C[j];
      }
      IC[i] = preTop;
    }
    IC[m] = top;
    free(rowsNnz);
    nnzC = top;
    JC = (int*)realloc(JC, sizeof(int) * nnzC);
    C = (double*)realloc(C, sizeof(double) * nnzC);
}

void omp_CSR_SpMM(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
    IC = (int*)calloc(m + 1, sizeof(int));
    double now = time_in_mill_now();
#pragma omp parallel firstprivate(stride) private(now)
    {
      int tid = omp_get_thread_num();
#ifdef profiling
#pragma omp master
      {
        now = time_in_mill_now();
      }
#endif
      omp_CSR_IC_nnzC(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, stride);
#pragma omp master
      {
#ifdef profiling
        std::cout << "time passed omp nnzC " << time_in_mill_now() - now << std::endl;
#endif
        JC = (int*)malloc(sizeof(int) * nnzC);
        C = (double*)malloc(sizeof(double) * nnzC);
      }
      double *x = thread_datas[tid].x;
      //bool *xb = thread_datas[tid].xb;
      int *index = thread_datas[tid].index;
      memset(index, -1, n * sizeof(int));
#pragma omp barrier
#pragma omp for schedule(dynamic) nowait
      for (int it = 0; it < m; it += stride) {
        int up = it + stride < m ? it + stride : m;
        for (int i = it; i < up; ++i) {
          //processCRowI(x, xb,
          indexProcessCRowI(index,
              IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
              IB, JB, B,
              JC + IC[i], C + IC[i]);
        }
      }
    }
#ifdef profiling
    std::cout << "time passed without memory allocate" << time_in_mill_now() - now << std::endl;
#endif
}

void omp_CSR_SpMM(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {
    thread_data_t* thread_datas = allocateThreadDatas(nthreads, n);
    omp_CSR_SpMM(IA, JA, A, nnzA,
        IB, JB, B, nnzB,
        IC, JC, C, nnzC,
        m, k, n, thread_datas, stride);
    freeThreadDatas(thread_datas, nthreads);
}
