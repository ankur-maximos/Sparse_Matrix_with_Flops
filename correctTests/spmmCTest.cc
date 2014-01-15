#include "mkl.h"
#include "process_args.h"
#include "qrmcl.h"
#include "tools/ntimer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkls/mkl_csr_kernel.h"
//#include <google/profiler.h>

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  COO cooAt;
  cooAt.readTransposedSNAPFile(options.inputFileName);
  CSR A = rmclInit(cooAt);
  //Mt.output("CSR Mt");
  cooAt.dispose();
  CSR B = A.deepCopy();
  CSR C;
  double now;
  //ProfilerStart("email.gprofile");
  CSR ompC;
  double sum = 0.0;
  // Warmup for omp_spmm
  ompC = A.omp_spmm(B, options.stride);
  ompC.dispose();
  const int up = 10;
  printf("IA=%p JA=%p\n", A.rowPtr, A.colInd);
  for (int i = 0; i < up; ++i) {
    now = time_in_mill_now();
    ompC = A.omp_spmm(B, options.stride);
    sum += time_in_mill_now() - now;
    if (i != up - 1)
    ompC.dispose();
  }
  std::cout << "time passed for " << up << " times omp cpu " << sum / up << std::endl;
  now = time_in_mill_now();
  C = ompC.deepCopy();
  std::cout << "time passed deepCopy C cpu " << time_in_mill_now() - now << std::endl;
  C.dispose();
  //ProfilerStop();
  A.toOneBasedCSR();
  B.toOneBasedCSR();

  // Warmup for mkl_spmm
  C = mkl_spmm(A, B);
  C.dispose();

  cout << "mkl threads8 = " << mkl_get_max_threads() << endl;
  double csum = 0.0;
  for (int i = 0; i < up; ++i) {
    now = time_in_mill_now();
    C = mkl_spmm(A, B);
    csum += time_in_mill_now() - now;
    if (i != up - 1)
    C.dispose();
  }
  std::cout << "time passed for " << up << " times mkl cpu " << csum / up << std::endl;
  C.toZeroBasedCSR();
  A.dispose();
  B.dispose();
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
