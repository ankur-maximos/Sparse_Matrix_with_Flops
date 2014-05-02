#include "mkl.h"
#include "process_args.h"
#include "qrmcl.h"
#include "tools/ntimer.h"
#include "tools/micpower_sample.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkls/mkl_csr_kernel.h"

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  int up = options.maxIters;
  COO cooAt;
  cooAt.readTransposedSNAPFile(options.inputFileName);
  CSR A = rmclInit(cooAt);
  cooAt.dispose();
  CSR B = A.deepCopy();
  long flops = A.spmmFlops(B);
  A.toOneBasedCSR();
  B.toOneBasedCSR();

  // Warmup for mkl_spmm
  CSR C = mkl_spmm(A, B);
  C.dispose();

  cout << "mkl threads8 = " << mkl_get_max_threads() << endl;
  double csum = 0.0;
  double now;
  for (int i = 0; i < up; ++i) {
    now = time_in_mill_now();
    C = mkl_spmm(A, B);
    csum += time_in_mill_now() - now;
    if (i != up - 1) {
      C.dispose();
    }
  }
  std::cout << "time passed for " << up << " times mkl cpu " << csum / up
    << " GFLOPS=" << flops / (csum / up) / 1e6 << std::endl;
  //C.toZeroBasedCSR();
  A.dispose();
  B.dispose();
  C.dispose();
  return 0;
}
