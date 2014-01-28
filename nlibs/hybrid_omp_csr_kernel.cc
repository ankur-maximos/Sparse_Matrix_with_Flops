#include <omp.h>
#include <math.h>
#include "tools/prefixSum.h"
#include "tools/util.h"
#include "cpu_csr_kernel.h"

void hybrid_omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
  IC = (int*)malloc((m + 1) * sizeof(int));
  int* rowsNnz = (int*)malloc((m + 1) * sizeof(int));
  int* footPrints = (int*)malloc((m + 1) * sizeof(int));
  static int ends[65];
  static double diffs = 1.1;
  const double alpha = 0.008;
  //static int pnnzB = -1;
  double now;
#pragma omp parallel firstprivate(stride)
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      if (diffs > alpha) {
        //pnnzB = nnzB;
        dynamic_omp_CSR_IC_nnzC_footprints(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, footPrints, stride);
#pragma omp barrier
#pragma omp single
        {
          arrayEqualPartition(footPrints, m, nthreads, ends);
        }
      } else {
        omp_CSR_IC_nnzC(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, stride);
        //static_omp_CSR_IC_nnzC(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, stride, ends, tid);
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
        double *cValues = C + IC[i];
        int *cColInd = JC + IC[i];
        indexProcessCRowI(index,
            IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
            IB, JB, B,
            JC + IC[i], C + IC[i]);
        int count = IC[i + 1] - IC[i];
        arrayInflationR2(cValues, count, cValues);
        pair<double, double> maxSum = arrayMaxSum(cValues, count);
        double rmax = maxSum.first, rsum = maxSum.second;
        double thresh = computeThreshold(rsum / count, rmax);
        arrayThreshPruneNormalize(thresh, cColInd, cValues,
            &count, cColInd, cValues);
        rowsNnz[i] = count;
      }
      //omp_matrix_relocation(rowsNnz, m, tid, stride, IC, JC, C, nnzC);
    }
    free(footPrints);
    matrix_relocation(rowsNnz, m, IC, JC, C, nnzC);
    free(rowsNnz);
    if (diffs > alpha) {
      diffs = fabs((double)nnzB - nnzC) / nnzB;
    } else {
      diffs += fabs((double)nnzB - nnzC) / nnzB;
    }
    printf("nnzB=%d top=%d diffs=%lf", nnzB, nnzC, diffs * 100);
    //printf("Compression ratio = %lf\n", (double)IC[m] / top);
    //printf("nnzB=%d pnnzB=%d top=%d diffs=%lf", nnzB, pnnzB, top, diffs * 100);
}
