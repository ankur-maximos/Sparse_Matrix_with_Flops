#include "mkl.h"
#include "process_args.h"
#include "qrmcl.h"
#include "tools/ntimer.h"
#include "tools/micpower_sample.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "DenseMatrix.h"
#include "mkls/mkl_csr_kernel.h"

int main(int argc, char *argv[]) {
  int up = 2;
  process_args(argc, argv);
  print_args();
  COO cooAt;
  cooAt.readTransposedSNAPFile(options.inputFileName);
  CSR A = rmclInit(cooAt);
  //Mt.output("CSR Mt");
  cooAt.dispose();
  CSR B = A.deepCopy();
  CSR C;
  double snow = time_in_mill_now();
  for (int i = 0; i < up; ++i) {
     C = A.omp_spmm(B, options.stride);
     if (i != up - 1) {
       C.dispose();
     }
  }
  printf("spmm time per iter pass %lf milliseconds\n", (time_in_mill_now() - snow) / up);
  printf("Before dA init\n");
  //C.output("CSR C"); C.dispose();
  DenseMatrix dA(A); A.dispose();
  //dA.output("DenseA");
  DenseMatrix dB(B); B.dispose();
  double alpha = 1.0, beta = 0.0;
  int m = A.rows, k = A.cols, n = B.cols;
  DenseMatrix dC(m, n);
  printf("Before dgemm call m=k=n=%d\n", n);
  double dnow = time_in_mill_now();
  for (int i = 0; i < up; ++i) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, alpha, dA.values, k, dB.values, n, beta, dC.values, n);
    printf("i=%d done\n", i);
  }
  printf("dense spmm time per iter %lf milliseconds\n", (time_in_mill_now() - snow) / up);
  printf("after dgemm call\n");
  //dC.output("DenseC");

  dA.dispose();
  dB.dispose();
  dC.dispose();
  return 0;
}
