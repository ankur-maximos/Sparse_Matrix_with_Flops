#include "CSR.h"
#include "COO.h"
#include "iostream"
#include "tools/ntimer.h"
#include "omp.h"
#include "cpu_csr_kernel.h"
#include "process_args.h"
#include "qrmcl.h"

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  COO cooAt;
  cooAt.readTransposedSNAPFile(options.inputFileName);
  CSR A = rmclInit(cooAt);
  CSR B = A.deepCopy();
  CSR C = A.omp_spmm(A);
  long flops = A.spmmFlops(A);
  printf("C: flops = %ld\n", flops);
  printf("C: nnz = %d\trows = %d\t nnz/rows = %.2lf\t ", C.nnz, C.rows, (double)C.nnz/C.rows);
  printf("C: flops/nnz=%.2lf\tflops/rows=%.2lf\n", (double)flops/C.nnz, (double)flops/C.rows);
  vector<int> stats = A.multiFlopsStats(B);
  outputStats(stats);
  vector<int> cstats = C.nnzStats();
  outputStats(cstats);
  A.dispose();
  B.dispose();
  C.dispose();
  return 0;
}

