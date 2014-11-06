#include "process_args.h"
#include "qrmcl.h"
#include "tools/ntimer.h"
//#include "tools/micpower_sample.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkls/mkl_csr_kernel.h"


int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  int up = options.maxIters;
  COO cooAt;
  cooAt.readSNAPFile(options.inputFileName, true);
  //cooAt.makeOrdered();
  printf("nnz before prune=%d", cooAt.nnz);
  cooAt.orderedAndDuplicatesRemoving();
  printf("nnz after prune=%d", cooAt.nnz);
  CSR A = cooAt.toCSR();
  //A.output("A");
  cooAt.dispose();
  CSR B = A.deepCopy();
  up = options.maxIters;
  double now;
  CSR ompC;
  long flops = A.spmmFlops(B);
  printf("IA=%p JA=%p\n", A.rowPtr, A.colInd);
  // Warmup for omp_spmm
  ompC = A.group_spmm(B, options.stride);
  ompC.dispose();
  double sum = 0.0;
  for (int i = 0; i < up; ++i) {
    now = time_in_mill_now();
    ompC = A.group_spmm(B, options.stride);
    sum += time_in_mill_now() - now;
    if (i != up - 1) {
      ompC.dispose();
    }
  }
  std::cout << "time passed for " << up << " times group cpu " << sum / up
    << " GFLOPS=" << flops / (sum / up) / 1e6 << std::endl;
  //ompC.output("group spmm:");
  A.toOneBasedCSR();
  B.toOneBasedCSR();
  CSR C = mkl_spmm(A, B);
  C.toZeroBasedCSR();
  A.dispose();
  B.dispose();
  //C.output("somp spmm:");
  bool isSame = C.isRawEqual(ompC);
  if (isSame) {
    std::cout << "Same\n";
  } else {
    std::cout << "Diffs\n";
  }
  C.dispose();
  ompC.dispose();
  return 0;
}
