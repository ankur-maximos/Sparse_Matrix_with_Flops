#include <omp.h>
#include <iostream>
#include "ntimer.h"
#include "cpu_csr_kernel.h"
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
void omp_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, thread_data_t thread_datas[],
    int* IC, int& nnzC) {
  const int stride = 128;
  double now = time_in_mill_now();
//#pragma omp parallel for schedule(static, (m + 8) / nthreads)
#pragma omp parallel for schedule(dynamic)
  for (int it = 0; it < m; it += stride) {
    int tid = omp_get_thread_num();
    int up = it + stride < m ? it + stride : m;
    for (int i = it; i < up; ++i) {
      int *iJC = thread_datas[tid].iJC;
      bool *xb = thread_datas[tid].xb;
      IC[i] = cRowiCount(i, IA, JA, IB, JB, iJC, xb);
    }
  }
  std::cout << "time passed IC count " << time_in_mill_now() - now << std::endl;
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
  //std::cout << "time passed prefix sum " << time_in_mill_now() - now << std::endl;
}

void omp_CSR_SpMM(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n) {
    IC = (int*)calloc(m + 1, sizeof(int));
    thread_data_t* thread_datas = allocateThreadDatas(nthreads, n);
    double now = time_in_mill_now();
    omp_CSR_IC_nnzC(IA, JA, IB, JB, m, n, thread_datas, IC, nnzC);
    //sequential_CSR_IC_nnzC(IA, JA, IB, JB, m, n, xb0, IC, nnzC);
    std::cout << "time passed omp nnzC " << time_in_mill_now() - now << std::endl;
    JC = (int*)malloc(sizeof(int) * nnzC);
    C = (double*)malloc(sizeof(double) * nnzC);
    const int stride = 128;
#pragma omp parallel for schedule(dynamic)
    for (int it = 0; it < m; it += stride) {
      int thread_id = omp_get_thread_num();
      int up = it + stride < m ? it + stride : m;
      for (int i = it; i < up; ++i) {
        //assert(ip == IC[i]);
        double *x = thread_datas[thread_id].x;
        bool *xb = thread_datas[thread_id].xb;
        int ip = IC[i];
        for(int jp = IA[i]; jp < IA[i + 1]; jp++) {
          int j = JA[jp];
          for(int tp = IB[j]; tp < IB[j + 1]; tp++) {
            int t = JB[tp];
            if(xb[t] == false) {
              JC[ip++] = t;
              xb[t] = true;
              x[t] = A[jp]*B[tp];
            } else
              x[t] += A[jp]*B[tp];
          }
        }
        for(int vp = IC[i]; vp < ip; vp++) {
          int v = JC[vp];
          C[vp] = x[v];
          x[v] = 0;
          xb[v] = false;
        }
      }
    }
    std::cout << "time passed without memory allocate" << time_in_mill_now() - now << std::endl;
    freeThreadDatas(thread_datas, nthreads);
}
