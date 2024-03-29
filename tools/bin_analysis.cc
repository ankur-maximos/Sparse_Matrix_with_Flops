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
  cooAt.readSNAPFile(options.inputFileName, false);
  cooAt.orderedAndDuplicatesRemoving();
  CSR At = cooAt.toCSR();

  COO cooA;
  cooA.readSNAPFile(options.inputFileName, false);
  cooA.orderedAndDuplicatesRemoving();
  CSR A = cooA.toCSR();
  //At.output("At");
  //A.output("A");
  cooAt.dispose();
  cooA.dispose();
  long flops = At.spmmFlops(A);
  printf("Oflops=%ld\n",flops);
  //assert (A.rows == A.cols && A.cols == B.cols && B.rows == B.cols);
  CSR C = At.somp_spmm(A, options.stride);
  //C.output("C");
  // Warmup for mkl_spmm
  At.dispose();
  A.dispose();
  C.dispose();
  return 0;
}
