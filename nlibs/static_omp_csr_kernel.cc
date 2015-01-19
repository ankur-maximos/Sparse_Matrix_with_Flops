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
  footPrints += count + 32 + (IA[i + 1] - IA[i]);
  //footPrints += count + 1;
  footPrints >>= 1; // The way to remove integer overflow in later prefix sum.
  return count;
}

/*
 * dynamic_omp_CSR_IC_nnzC_footprints reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void dynamic_omp_CSR_IC_nnzC_footprints(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, int* footPrints, const int stride) {
  int *iJC = (int*)thread_data.index;
  bool *xb = thread_data.xb;
#ifdef profiling
    QValue xnow = time_in_mill_now();
#endif
  memset(xb, 0, n);
#ifdef profiling
  printf("Time passed for thread %d memset xb with %lf milliseconds\n", omp_get_thread_num(), time_in_mill_now() - xnow);
#endif
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

//int indexRowId = -1;
void static_omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
#ifdef profiling
  QValue now = time_in_mill_now();
#endif
  IC = (int*)malloc((m + 1) * sizeof(int));
  int* footPrints = (int*)malloc((m + 1) * sizeof(int));
#ifdef profiling
  printf("Time passed for malloc IC and footprints with %lf milliseconds\n", time_in_mill_now() - now);
  now = time_in_mill_now();
#endif
  static int ends[MAX_THREADS_NUM];
#pragma omp parallel firstprivate(stride)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
#ifdef profiling
    QValue now = time_in_mill_now();
#endif
    dynamic_omp_CSR_IC_nnzC_footprints(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, footPrints, stride);
#ifdef profiling
    printf("Time passed for thread %d footprints nnzC with %lf milliseconds\n", tid, time_in_mill_now() - now);
#endif
#pragma omp barrier
#pragma omp single
    {
#ifdef profiling
      QValue now = time_in_mill_now();
#endif
      //spmmFootPrints(IA, JA, IB, IC, m, footPrints);
      arrayEqualPartition(footPrints, m, nthreads, ends);
#ifdef profiling
      std::cout << "time passed for just partition " << time_in_mill_now() - now << std::endl;
      arrayOutput("ends partitions ", stdout, ends, nthreads + 1);
      printf("Footprints partitions\n");
      for (int i = 0; i < nthreads; ++i) {
        printf("%d ", footPrints[ends[i + 1]] - footPrints[ends[i]]);
      }
      printf("\n");
      //std::cout << "time passed for footPrints and partition " << time_in_mill_now() - now << std::endl;
#endif
    }
#pragma omp master
    {
#ifdef profiling
      QValue mnow = time_in_mill_now();
#endif
      JC = (int*)malloc(sizeof(int) * nnzC);
      C = (QValue*)malloc(sizeof(QValue) * nnzC);
#ifdef profiling
      printf("time passed for malloc JC and C in main thread with %lf milliseconds\n", time_in_mill_now() - mnow);
      now = time_in_mill_now();
#endif
    }
    QValue *x = thread_datas[tid].x;
    int *index = thread_datas[tid].index;
#ifdef profiling
      QValue inow = time_in_mill_now();
#endif
    memset(index, -1, n * sizeof(int));
#ifdef profiling
    printf("Time passed for thread %d memset index with %lf milliseconds\n", tid, time_in_mill_now() - inow);
#endif
#pragma omp barrier
#ifdef profiling
      QValue tnow = time_in_mill_now();
#endif
    int low = ends[tid];
    int high = ends[tid + 1];
    for (int i = low; i < high; ++i) {
      //indexRowId = i;
      indexProcessCRowI(index,
          IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
          IB, JB, B,
          JC + IC[i], C + IC[i]);
    }
#ifdef profiling
     printf("Time passed for thread %d indexProcessCRowI with %lf milliseconds\n", tid, time_in_mill_now() - tnow);
#endif
  }
  free(footPrints);
#ifdef profiling
    std::cout << "time passed without memory allocate" << time_in_mill_now() - now << std::endl;
#endif
}

void static_omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {
#ifdef profiling
    QValue now = time_in_mill_now();
#endif
    int nthreads = 8;
#pragma omp parallel
#pragma omp master
    nthreads = omp_get_num_threads();
    thread_data_t* thread_datas = allocateThreadDatas(nthreads, n);
    static_omp_CSR_SpMM(IA, JA, A, nnzA,
        IB, JB, B, nnzB,
        IC, JC, C, nnzC,
        m, k, n, thread_datas, stride);
    freeThreadDatas(thread_datas, nthreads);
#ifdef profiling
    std::cout << "time passed for static_omp_CSR_SpMM total " <<  time_in_mill_now() - now << std::endl;
#endif
}

void static_omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
  IC = (int*)malloc((m + 1) * sizeof(int));
  int* rowsNnz = (int*)malloc((m + 1) * sizeof(int));
  int* footPrints = (int*)malloc((m + 1) * sizeof(int));
  static int ends[MAX_THREADS_NUM];
  QValue now;
#pragma omp parallel firstprivate(stride)
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      dynamic_omp_CSR_IC_nnzC_footprints(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, footPrints, stride);
#pragma omp barrier
#pragma omp single
      {
#ifdef profiling
        QValue now = time_in_mill_now();
#endif
        arrayEqualPartition(footPrints, m, nthreads, ends);
#ifdef profiling
        std::cout << "time passed for just partition " << time_in_mill_now() - now << std::endl;
        arrayOutput("ends partitions ", stdout, ends, nthreads + 1);
        printf("Footprints partitions\n");
        for (int i = 0; i < nthreads; ++i) {
          printf("%d ", footPrints[ends[i + 1]] - footPrints[ends[i]]);
        }
        printf("\n");
#endif
      }
#pragma omp master
      {
        JC = (int*)malloc(sizeof(int) * nnzC);
        C = (QValue*)malloc(sizeof(QValue) * nnzC);
#ifdef profiling
        now = time_in_mill_now();
#endif
      }
      QValue *x = thread_datas[tid].x;
      int *index = thread_datas[tid].index;
      memset(index, -1, n * sizeof(int));
#pragma omp barrier
#ifdef profiling
      QValue tnow = time_in_mill_now();
#endif
      int low = ends[tid];
      int high = ends[tid + 1];
      for (int i = low; i < high; ++i) {
        QValue *cQValues = C + IC[i];
        int *cColInd = JC + IC[i];
        indexProcessCRowI(index,
            IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
            IB, JB, B,
            JC + IC[i], C + IC[i]);
        int count = IC[i + 1] - IC[i];
        arrayInflationR2(cQValues, count, cQValues);
        pair<QValue, QValue> maxSum = arrayMaxSum(cQValues, count);
        QValue rmax = maxSum.first, rsum = maxSum.second;
        QValue thresh = computeThreshold(rsum / count, rmax);
        arrayThreshPruneNormalize(thresh, cColInd, cQValues,
            &count, cColInd, cQValues);
        rowsNnz[i] = count;
      }
#ifdef profiling
      printf("SOMP time passed for thread %d indexProcessCRowI with %lf milliseconds\n", tid, time_in_mill_now() - tnow);
#endif
#pragma omp barrier
      omp_matrix_relocation(rowsNnz, m, tid, stride, IC, JC, C, nnzC);
    }
    free(footPrints);
    //matrix_relocation(rowsNnz, m, IC, JC, C, nnzC);
    free(rowsNnz);
#ifdef profiling
    std::cout << "static_omp_CSR_RMCL_OneStep(SOMP) time passed " << time_in_mill_now() - now << std::endl;
#endif
}

void static_fair_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {
#ifdef profiling
  QValue now = time_in_mill_now();
#endif
  static_omp_CSR_SpMM(IA, JA, A, nnzA,
      IB, JB, B, nnzB,
      IC, JC, C, nnzC,
      m, k, n, stride);
  int* rowsNnz = (int*)malloc((m + 1) * sizeof(int));
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, stride)
    for (int i = 0; i < m; ++i) {
        QValue *cQValues = C + IC[i]; //-1 for one based IC index
        int *cColInd = JC + IC[i];
        int count = IC[i + 1] - IC[i];
        arrayInflationR2(cQValues, count, cQValues);
        pair<QValue, QValue> maxSum = arrayMaxSum(cQValues, count);
        QValue rmax = maxSum.first, rsum = maxSum.second;
        QValue thresh = computeThreshold(rsum / count, rmax);
        arrayThreshPruneNormalize(thresh, cColInd, cQValues,
            &count, cColInd, cQValues);
        rowsNnz[i] = count;
    }
    omp_matrix_relocation(rowsNnz, m, tid, stride, IC, JC, C, nnzC);
  }
  //matrix_relocation(rowsNnz, m, IC, JC, C, nnzC);
  free(rowsNnz);
#ifdef profiling
    std::cout << "static_fair_CSR_RMCL_OneStep(SFOMP) time passed " << time_in_mill_now() - now << std::endl;
#endif
}
