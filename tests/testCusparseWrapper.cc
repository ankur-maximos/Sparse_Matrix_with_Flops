#ifdef enable_gpu
#include "gpus/cusparse_spmm.h"
#include "CSR.h"
#include "qrmcl.h"
#include "process_args.h"
#include "tools/ntimer.h"
#endif

int main(int argc, char *argv[]) {
#ifdef enable_gpu
  process_args(argc, argv);
  print_args();
  COO cooAt;
  cooAt.readTransposedSNAPFile(options.inputFileName);
  CSR A = rmclInit(cooAt);
  cooAt.dispose();
  //A.output("A");
  CSR B = A.deepCopy();
  CSR dA = A.toGpuCSR();
  CSR dB = B.toGpuCSR();
  //gpuOutputCSRWrapper(dA, "dA");
  cusparse_init();
  CSR dC = cusparseSpMMWrapper(dA, dB);
  //gpuOutputCSRWrapper(dA, "dA after");
  dA.deviceDispose();
  dB.deviceDispose();
  CSR hC = dC.toCpuCSR();
  dC.deviceDispose();
  cusparse_finalize("clear cusparse");


  //hC.output("hC no ordered");
  double now = time_in_mill_now();
  CSR ompC = A.spmm(B);
  //CSR ompC = A.omp_spmm(B);
  printf("%lf pass for seq\n", time_in_mill_now() - now);
  hC.makeOrdered();
  ompC.makeOrdered();
  bool flag = hC.isEqual(ompC);
  printf("%s\n", flag ? "Same" : "Differs");
  //ompC.output("ompC");
  //hC.output("hC");
  A.dispose();
  B.dispose();
  hC.dispose();
  ompC.dispose();
#endif
  return 0;
}
