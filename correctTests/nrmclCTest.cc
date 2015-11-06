#include "CSR.h"
#include "COO.h"
#include "iostream"
#include "tools/ntimer.h"
#include "omp.h"
#include "tools/util.h"
#include "tools/key_value_qsort.h"
#include "process_args.h"
#include "gpus/gpu_csr_kernel.h"
#include "qrmcl.h"

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  //double now = time_in_mill_now();
  CSR Mt = RMCL(options.inputFileName, options.maxIters, SEQ);
  //printf("time pass rmcl total = %lf\n", time_in_mill_now() - now);
  double now = time_in_mill_now();
  CSR oMt = RMCL(options.inputFileName, options.maxIters, options.rmclOption);
  printf("time pass %s rmcl total = %lf\n", runOptionsStr[options.rmclOption], time_in_mill_now() - now);
#ifdef debugging
  Mt.output("mt");
  oMt.output("omt");
#endif
  Mt.makeOrdered();
  oMt.makeOrdered();
  bool isSame = Mt.isEqual(oMt);
  if (isSame) {
    std::cout << "Same\n";
  } else {
    std::cout << "Diffs\n";
  }
  Mt.dispose();
  oMt.dispose();
  return 0;
}
