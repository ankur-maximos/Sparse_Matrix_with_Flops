#include "CSR.h"
#include "COO.h"
#include "iostream"
#include "ntimer.h"
#include "omp.h"
#include "cpu_csr_kernel.h"

int main(int argc, char *argv[]) {
  char iname[500] = "matrix_05_05_crg.txt";
  if (argc == 2) {
    strcpy(iname, argv[1]);
  }
  COO cooA(iname);
  //cooA.output("COO");
  cooA.makeOrdered();
  //cooA.output("COO");
  CSR A = cooA.toCSR();
  cooA.dispose();
  //A.output("csr");
  double now = time_in_mill_now();
  CSR C = A.spmm(A);
  //C.output("cpu spmm");
  std::cout << "time passed cpu " << time_in_mill_now() - now << std::endl;
  now = time_in_mill_now();
  int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
  std::cout << "max_threads" << max_threads << std::endl;
  CSR ompC = A.omp_spmm(A);
  //CSR ompC = A.spmm(A);
  std::cout << "time passed omp " << time_in_mill_now() - now << std::endl;
  C.makeOrdered();
  ompC.makeOrdered();
  bool isSame = C.isEqual(ompC);
  if (isSame) {
    std::cout << "Same\n";
  } else {
    std::cout << "Diffs\n";
  }
  long flops = A.spmmFlops(A);
  printf("C: flops = %ld\n", flops);
  printf("C: nnz = %d\trows = %d\t nnz/rows = %.2lf\t ", C.nnz, C.rows, (double)C.nnz/C.rows);
  printf("C: flops/nnz=%.2lf\tflops/rows=%.2lf\n", (double)flops/C.nnz, (double)flops/C.rows);
  A.dispose();
  C.dispose();
  ompC.dispose();

  //std::cout << C.nnz << "\n";
  //C.output("C");
  //A.output("hA");
  //CSR gC;
  //gpu_CSR_SpMM(A, A, gC);
  //tcsr.output("C thrust");
  return 0;
}

