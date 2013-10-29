#include <omp.h>
#include <iostream>
#include "ntimer.h"
#include "cpu_csr_kernel.h"
#include "util.h"
using namespace std;


thread_data_t* allocateThreadDatas(int nthreads, int n) {
  thread_data_t* thread_datas = (thread_data_t*)calloc(nthreads, sizeof(thread_data_t));
  for(int i = 0; i < nthreads; i++) {
    thread_datas[i].init(n);
  }
  return thread_datas;
}

void freeThreadDatas(thread_data_t* thread_datas, int nthreads) {
  for(int i = 0; i < nthreads; i++) {
    thread_datas[i].~thread_data_t();
  }
  free(thread_datas);
}

int cRowiCount(const int i, const int IA[], const int JA[], const int IB[], const int JB[], int iJC[], bool xb[]) {
    int count = 0;
    for (int vp = IA[i]; vp < IA[i + 1]; ++vp) {
        int v = JA[vp];
        for (int kp = IB[v]; kp < IB[v+1]; ++kp) {
            int k = JB[kp];
            if(xb[k] == false) {
                iJC[count++]=k;
                xb[k]=true;
            }
        }
    }
    for(int jp = 0; jp < count; ++jp) {
        int j = iJC[jp];
        xb[j] = false;
    }
    return count;
}

const int nthreads = 8;
/*
 * omp_CSR_IC_nnzC reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void omp_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t thread_datas[],
    int* IC, int& nnzC) {
  const int stride = 128;
  double now;
  int tid = omp_get_thread_num();
  int *iJC = thread_datas[tid].iJC;
  bool *xb = thread_datas[tid].xb;
#pragma omp for schedule(dynamic)
  for (int it = 0; it < m; it += stride) {
    int up = it + stride < m ? it + stride : m;
    for (int i = it; i < up; ++i) {
      IC[i] = cRowiCount(i, IA, JA, IB, JB, iJC, xb);
    }
  }
#pragma omp master
  {
    //double now = time_in_mill_now();
    int t0 = IC[0];
    int t1;
    IC[0] = 0;
    for (int i = 0; i < m; ++i) {
      t1 = IC[i + 1];
      IC[i + 1] = IC[i] + t0;
      t0 = t1;
    }
    nnzC = IC[m];
  }
  //std::cout << "time passed prefix sum " << time_in_mill_now() - now << std::endl;
}

int processCRowI(double x[], bool* xb,
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
  }
  for(int vp = 0; vp < ip; ++vp) {
    int v = iJC[vp];
    iC[vp] = x[v];
    x[v] = 0;
    xb[v] = false;
  }
  return ip;
}

void omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas) {
    IC = (int*)calloc(m + 1, sizeof(int));
    int *rowsNnz = (int*)calloc(m + 1, sizeof(int));
    const int stride = 128;
#pragma omp parallel firstprivate(stride) //num_threads(1)
    {
      omp_CSR_IC_nnzC(IA, JA, IB, JB, m, n, thread_datas, IC, nnzC);
#pragma omp master
      {
        JC = (int*)malloc(sizeof(int) * nnzC);
        C = (double*)malloc(sizeof(double) * nnzC);
      }
      int thread_id = omp_get_thread_num();
      double *x = thread_datas[thread_id].x;
      bool *xb = thread_datas[thread_id].xb;
#pragma omp barrier
#pragma omp for schedule(dynamic)
      for (int it = 0; it < m; it += stride) {
        int up = it + stride < m ? it + stride : m;
        for (int i = it; i < up; ++i) {
          double *cValues = C + IC[i];
          int *cColInd = JC + IC[i];
          processCRowI(x, xb,
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
      int preTop = top;
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
        const int m, const int k, const int n, const thread_data_t* thread_datas) {
    IC = (int*)calloc(m + 1, sizeof(int));
    const int stride = 128;
    double now = time_in_mill_now();
#pragma omp parallel firstprivate(stride) private(now)
    {
#ifdef profiling
#pragma omp master
      {
        now = time_in_mill_now();
      }
#endif
      omp_CSR_IC_nnzC(IA, JA, IB, JB, m, n, thread_datas, IC, nnzC);
#pragma omp master
      {
#ifdef profiling
        std::cout << "time passed omp nnzC " << time_in_mill_now() - now << std::endl;
#endif
        JC = (int*)malloc(sizeof(int) * nnzC);
        C = (double*)malloc(sizeof(double) * nnzC);
      }
      int thread_id = omp_get_thread_num();
      double *x = thread_datas[thread_id].x;
      bool *xb = thread_datas[thread_id].xb;
#pragma omp barrier
#pragma omp for schedule(dynamic)
      for (int it = 0; it < m; it += stride) {
        int up = it + stride < m ? it + stride : m;
        for (int i = it; i < up; ++i) {
          processCRowI(x, xb,
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
        const int m, const int k, const int n) {
    thread_data_t* thread_datas = allocateThreadDatas(nthreads, n);
    omp_CSR_SpMM(IA, JA, A, nnzA,
        IB, JB, B, nnzB,
        IC, JC, C, nnzC,
        m, k, n, thread_datas);
    freeThreadDatas(thread_datas, nthreads);
}
