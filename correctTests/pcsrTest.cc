#include "process_args.h"
#include "tools/ntimer.h"
#include "CSR.h"
#include "PCSR.h"
#include <omp.h>

PCSR spmm(const CSR &A, const PCSR &pB, const int stride) {
  PCSR pC(pB.rows, pB.cols, pB.c);
  int nthreads = 8;
#pragma omp parallel
#pragma omp master
  nthreads = omp_get_num_threads();
  thread_data_t* thread_datas = allocateThreadDatas(nthreads, pB.stride());
  for (int b = 0; b < pB.c; ++b) {
    pC.blocks[b] = A.omp_spmm(thread_datas, pB.blocks[b], stride);
  }
  freeThreadDatas(thread_datas, nthreads);
  return pC;
}

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();

  COO cooAt;
  cooAt.readTransposedSNAPFile(options.inputFileName);
  CSR A = rmclInit(cooAt);
  //Mt.output("CSR Mt");
  cooAt.dispose();
  //A.output("CSR A");
  CSR B = A.deepCopy();
  CSR C = A.omp_spmm(B, options.stride);
  PCSR pB(B, 2);
  B.dispose();
  PCSR pC = spmm(A, pB, options.stride);
  //pB.output("PCSR A");
  pC.dispose();
  const int up = 10;
  double sum = 0.0;
  double now;
  for (int i = 0; i < up; ++i) {
    now = time_in_mill_now();
    pC = spmm(A, pB, options.stride);
    sum += time_in_mill_now() - now;
    if (i != up - 1) {
      pC.dispose();
    }
  }
  std::cout << "time passed for " << up << " times omp cpu " << sum / up << std::endl;
  pB.dispose();
  bool isSame = pC.isEqual(C);
  if (isSame) {
    std::cout << "Same\n";
  } else {
    std::cout << "Diffs\n";
  }
  pC.dispose();
  C.dispose();
  return 0;
}
