#include "cpu_csr_kernel.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include <cilk/cilk.h>
#include <iostream>
#include <omp.h>
using namespace std;

void cilk_CSR_RMCL_OneStep(const int IA[], const int JA[], const double A[], const int nnzA,
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
    }
    cilk_for (int it = 0; it < m; it += stride) {
      int tid = __cilkrts_get_worker_number();
      double *x = thread_datas[tid].x;
      int *index = thread_datas[tid].index;
      bool *xb = thread_datas[tid].xb;
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
