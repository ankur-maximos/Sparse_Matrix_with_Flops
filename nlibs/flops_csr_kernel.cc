#include <omp.h>
#include <iostream>
#include "cpu_csr_kernel.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include "tools/prefixSum.h"
using namespace std;

/*
 * dynamic_omp_CSR_flops reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void dynamic_omp_CSR_flops(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n,
    int* IC, int& nnzC, int* rowFlops, const int stride) {
#pragma omp for schedule(dynamic, stride)
  for (int i = 0; i < m; ++i) {
    int tmpRowFlops = 0;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      int BrowFlops = IB[j + 1] - IB[j];
      tmpRowFlops += BrowFlops;
    }
    rowFlops[i] = (tmpRowFlops >> 1);
    //rowFlops[i] = (tmpRowFlops);
  }
#pragma omp barrier
  noTileOmpPrefixSum(rowFlops, rowFlops, m);
}

void flops_omp_CSR_SpMM(const int IA[], const int JA[], const Value A[], const int nnzA,
        const int IB[], const int JB[], const Value B[], const int nnzB,
        int* &IC, int* &JC, Value* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
#ifdef profiling
  Value now = time_in_mill_now();
#endif
  IC = (int*)malloc((m + 1) * sizeof(int));
  int* flops = (int*)malloc((m + 1) * sizeof(int));
#ifdef profiling
  printf("Time passed for malloc IC and flops with %lf milliseconds\n", time_in_mill_now() - now);
  now = time_in_mill_now();
#endif
  static int ends[MAX_THREADS_NUM];
#pragma omp parallel firstprivate(stride)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
#ifdef profiling
    Value now = time_in_mill_now();
#endif
    dynamic_omp_CSR_flops(IA, JA, IB, JB, m, n, IC, nnzC, flops, stride);
#ifdef profiling
    printf("Time passed for thread %d dynamic_omp_CSR_flops with %lf milliseconds\n", tid, time_in_mill_now() - now);
#endif
#pragma omp barrier
#pragma omp single
    {
      arrayEqualPartition(flops, m, nthreads, ends);
      //arrayOutput("ends partitions ", stdout, ends, nthreads + 1);
    }
#ifdef profiling
    now = time_in_mill_now();
#endif
    int *iJC = (int*)thread_datas[tid].index;
    bool *xb = thread_datas[tid].xb;
    memset(xb, 0, n);
#ifdef profiling
    printf("Time passed for thread %d memset xb with %lf milliseconds\n", tid, time_in_mill_now() - now);
    now = time_in_mill_now();
#endif
    int low = ends[tid];
    int high = ends[tid + 1];
    for (int i = low; i < high; ++i) {
      IC[i] = cRowiCount(i, IA, JA, IB, JB, iJC, xb);
    }
#ifdef profiling
    printf("Time passed for thread %d nnz cRowiCount with %lf milliseconds\n", tid, time_in_mill_now() - now);
#endif
#pragma omp barrier
    noTileOmpPrefixSum(IC, IC, m);
#pragma omp master
    {
#ifdef profiling
      Value now = time_in_mill_now();
#endif
      nnzC = IC[m];
      JC = (int*)malloc(sizeof(int) * nnzC);
      C = (Value*)malloc(sizeof(Value) * nnzC);
#ifdef profiling
      printf("time passed for malloc JC and C in main thread with %lf milliseconds\n", time_in_mill_now() - now);
#endif
    }
#ifdef profiling
    now = time_in_mill_now();
#endif
    Value *x = thread_datas[tid].x;
    int *index = thread_datas[tid].index;
    memset(index, -1, n * sizeof(int));
#ifdef profiling
    printf("Time passed for thread %d memset index with %lf milliseconds\n", tid, time_in_mill_now() - now);
#endif
#pragma omp barrier
#ifdef profiling
    now = time_in_mill_now();
#endif
    for (int i = low; i < high; ++i) {
      indexProcessCRowI(index,
          IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
          IB, JB, B,
          JC + IC[i], C + IC[i]);
    }
#ifdef profiling
    printf("Time passed for thread %d indexProcessCRowI with %lf milliseconds\n", tid, time_in_mill_now() - now);
#endif
  }
  free(flops);
}

void flops_omp_CSR_SpMM(const int IA[], const int JA[], const Value A[], const int nnzA,
        const int IB[], const int JB[], const Value B[], const int nnzB,
        int* &IC, int* &JC, Value* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {
#ifdef profiling
    Value now = time_in_mill_now();
#endif
    int nthreads = 8;
#pragma omp parallel
#pragma omp master
    nthreads = omp_get_num_threads();
    thread_data_t* thread_datas = allocateThreadDatas(nthreads, n);
    flops_omp_CSR_SpMM(IA, JA, A, nnzA,
        IB, JB, B, nnzB,
        IC, JC, C, nnzC,
        m, k, n, thread_datas, stride);
    freeThreadDatas(thread_datas, nthreads);
#ifdef profiling
    std::cout << "time passed for flops_omp_CSR_SpMM total " <<  time_in_mill_now() - now << std::endl;
#endif
}
