#include "process_args.h"
#include "qrmcl.h"
#include "tools/ntimer.h"
//#include "tools/micpower_sample.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  int up = options.maxIters;
  COO cooAt;
  cooAt.readSNAPFile(options.inputFileName, true);
  CSR A = rmclInit(cooAt);
  cooAt.dispose();
  CSR B = A.deepCopy();
  CSR C;
  up = options.maxIters;
  double now;
  CSR ompC;
  long flops = A.spmmFlops(B);
  printf("IA=%p JA=%p\n", A.rowPtr, A.colInd);
  // Warmup for omp_spmm
  ompC = A.omp_spmm(B, options.stride);
  ompC.dispose();
  double sum = 0.0;
  for (int i = 0; i < up; ++i) {
    now = time_in_mill_now();
    ompC = A.omp_spmm(B, options.stride);
    sum += time_in_mill_now() - now;
    if (i != up - 1) {
      ompC.dispose();
    }
  }
  std::cout << "time passed for " << up << " times domp cpu " << sum / up
    << " GFLOPS=" << flops / (sum / up) / 1e6 << std::endl;
  ompC.dispose();
  A.dispose();
  B.dispose();
  return 0;
}
