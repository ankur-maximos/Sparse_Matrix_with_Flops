#include "mkl.h"
#include "process_args.h"
#include "qrmcl.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkls/mkl_csr_kernel.h"
//#include <google/profiler.h>

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  COO cooAt;
  cooAt.readSNAPFile(options.inputFileName, true);
  CSR A = rmclInit(cooAt);
  cooAt.dispose();
  CSR B = A.deepCopy();
  int *P = randomPermutationVector(A.rows);
  CSR pAPt = A.PMPt(P);
  CSR pBPt = pAPt.deepCopy();
  CSR ompC = A.omp_spmm(B, options.stride);
  double sum = 0.0;
  const int up = 4;
  for (int i = 0; i < up; ++i) {
    double now = time_in_mill_now();
    ompC = A.omp_spmm(B, options.stride);
    sum += time_in_mill_now() - now;
    if (i != up - 1)
    ompC.dispose();
  }
  std::cout << "time passed for " << up << " times omp cpu " << sum / up << std::endl;
  CSR pCPt = pAPt.omp_spmm(pBPt); pCPt.dispose();
  sum = 0.0;
  for (int i = 0; i < up; ++i) {
    double now = time_in_mill_now();
    pCPt = pAPt.omp_spmm(pBPt, options.stride);
    sum += time_in_mill_now() - now;
    if (i != up - 1) {
      pCPt.dispose();
    }
  }
  std::cout << "time passed for " << up << " times pt omp " << sum / up << std::endl;
  CSR C = pCPt.PtMP(P); pCPt.dispose();
  A.dispose();
  B.dispose();
  pAPt.dispose();
  pBPt.dispose();
  C.makeOrdered();
  ompC.makeOrdered();
  bool isSame = C.isEqual(ompC);
  if (isSame) {
    std::cout << "Same\n";
  } else {
    std::cout << "Diffs\n";
  }
  C.dispose();
  ompC.dispose();
  return 0;
}
