
#include "CSR.h"
#include "gpus/cuda_handle_error.h"
#include "gpus/gpu_csr_kernel.h"
#include "process_args.h"
#include "qrmcl.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include "tools/prefixSum.h"

void test(const COO &cooA, const COO &cooB);

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  COO cooA;
  cooA.readSNAPFile(options.inputFileName, false);
  cooA.orderedAndDuplicatesRemoving();
  test(cooA, cooA);
  cooA.dispose();
  return 0;
}
