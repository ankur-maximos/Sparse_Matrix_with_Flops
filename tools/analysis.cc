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
  //assert (A.rows == A.cols && A.cols == B.cols && B.rows == B.cols);
  CSR C = At.somp_spmm(A, options.stride);
  //C.output("C");
  printf("sparsity of A=%lf\n", (double)A.nnz / A.rows / A.rows);
  printf("sparsity of C=%lf\n", (double)C.nnz / C.rows / C.rows);
  printf("N= %d\tAnnz= %d Cnnz=%d flops= %ld flops/cnnz=%lf\n", A.rows, A.nnz, C.nnz, flops, ((double)flops) / C.nnz);
  printf("%d & %d & %d & %lf\n", A.rows, A.nnz, C.nnz, flops, ((double)flops) / C.nnz);
  printf("Cnnz/Annz=%lf\n", (double)C.nnz / A.nnz);

  //vector<int> stats = A.nnzStats();
  vector<int> res = flopsStats(A.rowPtr,A.colInd,A.rowPtr,A.colInd,A.rows);
  //vector<int> statsC = C.nnzStats();
  outputStats(res);
  // Warmup for mkl_spmm
  At.dispose();
  A.dispose();
  C.dispose();
  return 0;
}
