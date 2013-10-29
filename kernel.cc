#include "CSR.h"
#include <omp.h>

void rowRMCLIter(const int maxIter, const CSR Mgt, CSR &Mt) {
  CSR newMt;
  double tsum = 0.0;
  //IC = (int*)calloc(m + 1, sizeof(int));
  const int nthreads = 8;
  const int stride = 128;
  thread_data_t* thread_datas = allocateThreadDatas(nthreads, Mgt.rows);
  for (int iter = 0; iter < maxIter; ++iter) {
//#pragma omp parallel firstprivate(stride)
  }
  freeThreadDatas(thread_datas, nthreads);
}
