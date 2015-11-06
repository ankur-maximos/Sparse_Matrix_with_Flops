#include "process_args.h"
#include "qrmcl.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include "tools/prefixSum.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

inline int allInOne(int *restrict index,
    const int i, const int IA[], const int JA[], const QValue A[],
    const int IB[], const int JB[], const QValue B[],
    int* restrict iJC, QValue* restrict iC) {
  if (IA[i] == IA[i + 1]) {
    return 0;
  }
  int cp = -1;
  int ap = IA[i];
  int a = JA[ap];
  //printf("Process row i=%d a=%d ap=%d IB[a]=%d IB[a + 1]=%d\n", i, a, ap, IB[a], IB[a + 1]);
  for (int bp = IB[a]; bp < IB[a + 1]; ++bp) {
    int b = JB[bp];
    iJC[++cp] = b;
    index[b] = cp;
    iC[cp] = A[ap] * B[bp];
  }
  //printf("middle finish row i=%d\n", i);
  ++ap;
  for (; ap < IA[i + 1]; ++ap) {
    int a = JA[ap];
    for (int bp = IB[a]; bp < IB[a + 1]; ++bp) {
      int b = JB[bp];
      if(index[b] == -1) {
        iJC[++cp] = b;
        index[b] = cp;
        iC[cp] = A[ap] * B[bp];
      } else {
        iC[index[b]] += A[ap] * B[bp];
      }
    }
  }
  ++cp;
  for(int jp = 0; jp < cp; ++jp) {
    int c = iJC[jp];
    index[c] = -1;
  }
  return cp;
}

struct PCSR {
  int ranges[MAX_THREADS_NUM];
  int *tIC[MAX_THREADS_NUM];
  int *tJC[MAX_THREADS_NUM];
  QValue *tC[MAX_THREADS_NUM];
  int rows, cols;
  int partitions;

  void dispose() {
    for (int tid = 0; tid < partitions; ++tid) {
      free(tIC[tid] + ranges[tid]);
      free(tJC[tid]);
      free(tC[tid]);
    }
  }

  void output(const char* msg) const {
    printf("%s\n", msg);
    for (int tid = 0; tid < partitions; ++tid) {
      for (int i = ranges[tid]; i < ranges[tid + 1]; ++i) {
        for (int cp = tIC[tid][i]; cp < tIC[tid][i + 1]; ++cp) {
          int col = tJC[tid][cp];
          QValue val = tC[tid][cp];
          printf("%d %d %lf\n", i, col, val);
        }
      }
    }
  }

  CSR toCSR() const {
    int nnz = 0;
    CSR csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.rowPtr = (int*)malloc((rows + 1) * sizeof(int));
    //csr.colInd
    for (int tid = 0; tid < partitions; ++tid) {
      int low = ranges[tid];
      int high = ranges[tid + 1];
      nnz += tIC[tid][high] - tIC[tid][low];
      for (int i = ranges[tid]; i < ranges[tid + 1]; ++i) {
        for (int cp = tIC[tid][i]; cp < tIC[tid][i + 1]; ++cp) {
          int col = tJC[tid][cp];
          QValue val = tC[tid][cp];
        }
      }
    }
    csr.nnz = nnz;
    int *JC = (int*)malloc(nnz * sizeof(int));
    QValue *VC = (QValue*)malloc(nnz * sizeof(QValue));
    nnz = 0;
    for (int tid = 0; tid < partitions; ++tid) {
      int low = ranges[tid];
      int high = ranges[tid + 1];
      for (int i = ranges[tid]; i < ranges[tid + 1]; ++i) {
        csr.rowPtr[i] = tIC[tid][i] + nnz;
        for (int cp = tIC[tid][i]; cp < tIC[tid][i + 1]; ++cp) {
          //int col = tJC[tid][cp];
          //QValue val = tC[tid][cp];
          JC[cp + nnz] = tJC[tid][cp];
          VC[cp + nnz] = tC[tid][cp];
        }
      }
      nnz += tIC[tid][high] - tIC[tid][low];
    }
    assert (ranges[partitions] == rows);
    csr.rowPtr[rows] = nnz;
    csr.colInd = JC;
    csr.values = VC;
    return csr;
  }
};

void vomp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
    const int IB[], const int JB[], const QValue B[], const int nnzB,
    PCSR &pc,
    const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
  //IC = (int*)malloc((m + 1) * sizeof(int));
  long* flops = (long*)malloc((m + 1) * sizeof(long));
  int threads_count = 0;
  // omp_lock_t wlock;
  // omp_init_lock(&wlock);
#pragma omp parallel firstprivate(stride)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
#ifdef mprofile
    double dnow = time_in_mill_now();
#endif
    dynamic_omp_CSR_flops(IA, JA, IB, JB, m, n, flops, stride);
#ifdef mprofile
    printf("dflops time=%lf millis for thread=%d\n", time_in_mill_now() - dnow, tid);
#endif
#pragma omp barrier
#pragma omp single
    {
      //arrayOutput("flops", stdout, flops, m + 1);
      const long chunk_size = (flops[m] + nthreads - 1) / nthreads;
      //printf("chunk_size=%ld flops[m]=%ld\n", chunk_size, flops[m]);
      arrayEqualPartition64(flops, m, nthreads, pc.ranges);
      threads_count = nthreads;
      assert (threads_count <= MAX_THREADS_NUM);
    }
    int *index = thread_datas[tid].index;
    memset(index, -1, n * sizeof(int));
    int low = pc.ranges[tid];
    int high = pc.ranges[tid + 1];
    long hflops = flops[high] - flops[low];
    size_t presize = hflops / 3 + k;
    if (presize > (high - low) * n) {
      presize = (high - low) * n;
    }
    //printf("presize=%lu on tid=%d low=%d high=%d\n", presize, tid, low, high);
    int *pIC = (int*)malloc(sizeof(int) * (high - low + 1)) - low;
    int *pJC = (int*)malloc(sizeof(int) * presize);
    QValue *pC = (QValue*)malloc(sizeof(QValue) * presize);
#ifdef mprofile
    double fnow = time_in_mill_now();
#endif
    int tcnnz = 0;
    for (int i = low; i < high; ++i) {
      pIC[i] = tcnnz;
      size_t rowr = std::min(flops[i + 1] - flops[i], (long) k);
      if (tcnnz + rowr > presize) {
        //size_t resize = std::max(tcnnz + std::min(flops[i + 1] - flops[i], k), tcnnz + (flops[high] - flops[i]) / 3);
        presize = std::max(tcnnz + rowr, presize + presize / 2 + 1);
        //printf("Thread %d realloc presize=%lu\n", tid, presize);
        //omp_set_lock(&wlock);
        pJC = (int*)realloc(pJC, presize * sizeof(int)); assert (pJC != NULL);
        pC = (QValue*)realloc(pC, presize * sizeof(QValue)); assert (pC != NULL);
        //omp_unset_lock(&wlock);
      }
      int nnzi = allInOne(index,
          i, IA, JA, A,
          IB, JB, B,
          pJC + tcnnz, pC + tcnnz);
      tcnnz += nnzi;
    }
#ifdef mprofile
    printf("allInOne time=%lf on thread %d\n", time_in_mill_now() - fnow, tid);
#endif
    pIC[high] = tcnnz;
    pc.tIC[tid] = pIC;
    pc.tJC[tid] = pJC;
    pc.tC[tid] = pC;
  }
  free(flops);
  pc.partitions = threads_count;
  pc.rows = m;
  pc.cols = k;
  //omp_destroy_lock(&wlock);
}

void vomp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
    const int IB[], const int JB[], const QValue B[], const int nnzB,
    PCSR &pc,
    const int m, const int k, const int n, const int stride) {
  int nthreads = 8;
#pragma omp parallel
#pragma omp master
  nthreads = omp_get_num_threads();
  thread_data_t* thread_datas = allocateThreadDatas(nthreads, n);
  vomp_CSR_SpMM(IA, JA, A, nnzA,
      IB, JB, B, nnzB,
      pc,
      m, k, n, thread_datas, stride);
  freeThreadDatas(thread_datas, nthreads);
}

PCSR vomp_CSR_SpMM_Wrapper(CSR &A, CSR &B, int stride) {
  PCSR pc;
  vomp_CSR_SpMM(A.rowPtr, A.colInd, A.values, A.nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      pc,
      A.rows, A.cols, B.cols, stride);
  return pc;
}

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  int up = options.maxIters;
  COO cooAt;
  cooAt.readSNAPFile(options.inputFileName, false);
  cooAt.orderedAndDuplicatesRemoving();
  CSR A = cooAt.toCSR();
  cooAt.dispose();
  CSR B = A.deepCopy();
  long flops = A.spmmFlops(B);
  //A.output("A");
  PCSR pc = vomp_CSR_SpMM_Wrapper(A, B, options.stride);
  pc.dispose();
  double csum = 0.0;
  double now;
  for (int i = 0; i < up; ++i) {
    now = time_in_mill_now();
    pc = vomp_CSR_SpMM_Wrapper(A, B, options.stride);
    csum += time_in_mill_now() - now;
    if (i != up - 1)
      pc.dispose();
  }
  std::cout << "time passed for " << up << " times vcsr cpu " << csum / up
    << " GFLOPS=" << flops / (csum / up) / 1e6 << std::endl;
  CSR sC = A.somp_spmm(B, options.stride);
  //pc.output("PCSR");
  CSR C = pc.toCSR();
  bool isSame = sC.isRawEqual(C);
  if (isSame) {
    std::cout << "Same\n";
  } else {
    std::cout << "Diffs\n";
  }
  A.dispose();
  B.dispose();
  pc.dispose();
  //C.output("CSR C");
  C.dispose();
  return 0;
}
