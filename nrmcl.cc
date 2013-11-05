#include "CSR.h"
#include "COO.h"
#include "iostream"
#include "tools/ntimer.h"
#include "omp.h"
#include "tools/util.h"
#include "tools/key_value_qsort.h"
#include "process_args.h"
#include "gpus/gpu_csr_kernel.h"

void ompRmclIter(const int maxIter, const CSR Mgt, CSR &Mt) {
  CSR newMt;
  double tsum = 0.0;
  int nthreads = 8;
  FILE *fp = NULL;
  static const int cpercents[] = {-30.0, -20.0, -5.0, -0.0, 5.0, 20.0, 30.0, 100.0};
  vector<double> percents(cpercents, cpercents + sizeof(cpercents) / sizeof(int));
  if (options.stats) {
    fp = fopen("percent.stats", "w");
    fprintf(fp, "rows %d\n", Mt.rows);
    arrayOutput("percent\t", fp, percents);
  }
  double now = time_in_mill_now();
  thread_data_t* thread_datas = allocateThreadDatas(nthreads, Mt.cols);
  for (int iter = 0; iter < maxIter; ++iter) {
#ifdef debugging
    Mgt.output("Mgt iter");
    Mt.output("Mt iter");
#endif
    newMt = Mgt.rmclOneStep(Mt, thread_datas);
    if (options.stats) {
      vector<int> counts = Mt.differsStats(newMt, percents);
      char msg[20];
      sprintf(msg, "%d :\t", iter);
      arrayOutput(msg, fp, counts);
    }
    Mt.dispose();
    Mt = newMt;
    printf("iter %d done\n", iter);
  }
  freeThreadDatas(thread_datas, nthreads);
  if (options.stats) {
    fclose(fp);
  }
  cout << "iter finish in " << time_in_mill_now() - now << "\n";
}

void rmclIter(const int maxIter, const CSR Mgt, CSR &Mt) {
  CSR newMt;
  double tsum = 0.0;
  for (int iter = 0; iter < maxIter; ++iter) {
  double now = time_in_mill_now();
    //newMt = Mgt.omp_spmm(Mt);
    newMt = Mgt.spmm(Mt);
    tsum += time_in_mill_now() - now;
  //printf("time pass total = %lf\n", time_in_mill_now() - now);
    //newMt.output("newMt spmm");
    int pos = 0;
    int i;
    for (i = 0; i < newMt.rows; ++i) {
      int count = newMt.rowCount(i);
      double* values = (double*)malloc(count * sizeof(double));
      arrayInflationR2(newMt.values + newMt.rowPtr[i], count, values);
      double rmax = arrayMax(values, count);
      double rsum = arraySum(values, count);
      double thresh = computeThreshold(rsum / count, rmax);
      int* indices = (int*)malloc(count * sizeof(int));
      double nsum = arrayThreshPruneNormalize(thresh, newMt.colInd + newMt.rowPtr[i], values,
          &count, indices, values);
      memcpy(newMt.values + pos, values, count * sizeof(double));
      memcpy(newMt.colInd + pos, indices, count * sizeof(int));
      newMt.rowPtr[i] = pos;
      pos += count;
      free(values);
      free(indices);
    }
    newMt.rowPtr[newMt.rows] = pos;
    newMt.nnz = pos;
    Mt.dispose();
    Mt = newMt;
    printf("iter %d done\n", iter);
  }
  printf("time pass tsum spmm = %lf\n", tsum);
}

CSR rmclInit(COO &cooAt) {
  //cooAt.output("COO");
  cooAt.addSelfLoopIfNeeded();
  cooAt.makeOrdered();
  //cooA.output("COO");
  CSR At = cooAt.toCSR();
  At.averAndNormRowValue();
  return At;
}

CSR ompRMCL(const char iname[], int maxIters) {
  COO cooAt;
  cooAt.readTransposedSNAPFile(iname);
  CSR Mt = rmclInit(cooAt);
  //Mt.output("CSR Mt");
  cooAt.dispose();
  CSR Mgt = Mt.deepCopy();
  //Mt.output("CSR Mgt");
  double now = time_in_mill_now();
  //rmclIter(maxIters, Mgt, Mt);
  ompRmclIter(maxIters, Mgt, Mt);
  printf("time pass iters = %lf\n", time_in_mill_now() - now);
  Mgt.dispose();
  return Mt;
}

CSR RMCL(const char iname[], int maxIters) {
  COO cooAt;
  cooAt.readTransposedSNAPFile(iname);
  CSR Mt = rmclInit(cooAt);
  //Mt.output("CSR Mt");
  cooAt.dispose();
  CSR Mgt = Mt.deepCopy();
  //Mt.output("CSR Mgt");
  double now = time_in_mill_now();
  rmclIter(maxIters, Mgt, Mt);
  //ompRmclIter(maxIters, Mgt, Mt);
  printf("time pass iters = %lf\n", time_in_mill_now() - now);
  return Mt;
}

CSR gpuRMCL(const char iname[], int maxIters) {
  COO cooAt;
  cooAt.readTransposedSNAPFile(iname);
  CSR Mt = rmclInit(cooAt);
  //Mt.output("CSR Mt");
  cooAt.dispose();
  CSR Mgt = Mt.deepCopy();
  //Mgt.output("CSR Mgt");
  double now = time_in_mill_now();
  //rmclIter(maxIters, Mgt, Mt);
  gpuRmclIter(maxIters, Mgt, Mt);
  printf("gpu time pass iters = %lf\n", time_in_mill_now() - now);
  Mgt.dispose();
  return Mt;
}

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  //double now = time_in_mill_now();
  //CSR Mt = RMCL(options.inputFileName, options.maxIters);
  //printf("time pass rmcl total = %lf\n", time_in_mill_now() - now);
  //now = time_in_mill_now();
  CSR oMt = ompRMCL(options.inputFileName, options.maxIters);
  /*printf("time pass omp rmcl total = %lf\n", time_in_mill_now() - now);
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
  }*/
  //Mt.dispose();
  oMt.dispose();
  //CSR gpuMt = gpuRMCL(options.inputFileName, options.maxIters);
  return 0;
}
