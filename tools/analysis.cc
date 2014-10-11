#include "mkl.h"
#include "process_args.h"
#include "qrmcl.h"
#include "tools/ntimer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkls/mkl_csr_kernel.h"
#include "tools/stats.h"

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
  assert (A.rows == A.cols && A.cols == B.cols && B.rows == B.cols);
  CSR C = A.somp_spmm(B, options.stride);
  printf("N= %d\tAnnz= %d Cnnz=%d flops= %ld\n", A.rows, A.nnz, C.nnz, flops);

  vector<int> stats = A.nnzStats();
  outputStats(stats);
  // Warmup for mkl_spmm
  A.dispose();
  B.dispose();
  C.dispose();
  return 0;
}
