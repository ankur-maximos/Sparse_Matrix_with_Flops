//#ifdef enable_gpu
#include "CSR.h"
#include "gpus/cuda_handle_error.h"
#include "gpus/gpu_csr_kernel.h"
#include "process_args.h"
#include "qrmcl.h"
#include "gpus/timer.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include "tools/prefixSum.h"
#include <thrust/device_ptr.h>
//#endif
#include <omp.h>
#include "mkls/mkl_csr_kernel.h"
#include "gpus/cusparse_spmm.h"

CSR sgpuSpMMWrapper(const CSR &dA, const CSR &dB, int *drowIds, int v[4]);

void ompFlops(const int m, const int IA[], const int JA[], const int IB[], long rowFlops[]) {
  const int stride = 512;
#pragma omp for schedule(dynamic, stride)
  for (int i = 0; i < m; ++i) {
    long tmpRowFlops = 0;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      int BrowFlops = IB[j + 1] - IB[j];
      tmpRowFlops += BrowFlops;
    }
    rowFlops[i] = tmpRowFlops;
  }
}

inline int queueId(long x) {
  assert (x > 0);
  if (x == 0) return 0;
  else if (x == 1) return 1;
  int ret = 2;
  long up = 2;
  for (up = 2; ; up *= 2, ++ret) {
    if (x <= up) return ret;
  }
  return -1;
}

vector<int> classifyFlops(const long rowFlops[], const int IA[], const int m, int hqueue[]) {
  //vector<int> vqueue[33];
  vector<int> vqueue[64];
  int maxq = -1;
  for (int i = 0; i < m; ++i) {
    if (rowFlops[i] == 0) continue;
    int q = 0;
    const int acount = IA[i + 1] - IA[i];
    //if (acount >= 64) {
    if (acount >= 128) {
      vqueue[63].push_back(i);
      continue;
    }
    if (acount > 1) q = queueId(rowFlops[i]);
    assert (q < 32);
    //cout << "classify rowId=" << i << " flops=" << rowFlops[i] << " "<< q << endl;
    if (maxq < q) maxq = q;
    vqueue[q].push_back(i);
  }
  //int qup = maxq + 1;
  int qup = 64;
  vector<int> v;
  v.resize(qup + 1);
  v[0] = 0;
  for (int t = 1; t <= qup; ++t) {
    v[t] = v[t - 1] + vqueue[t - 1].size();
  }
  //arrayOutput("v ", stdout, v);
  for (int k = 0; k < qup; ++k) {
    int top = 0;
    for (int i = v[k]; i < v[k + 1]; ++i, ++top) {
      hqueue[i] = vqueue[k][top];
    }
  }
  return v;
}

bool isPartialRawEqual(int s, int e, const int hqueue[], const CSR &hC, const CSR &rC) {
  float* rowVals = (float*)malloc(hC.cols * sizeof(float));
  memset(rowVals, 0, hC.cols * sizeof(float));
  bool flag = true;
  for (int q = s; q < e; ++q) {
    const int rowId = hqueue[q];
    assert (hC.rowPtr[rowId] == rC.rowPtr[rowId]);
    assert (hC.rowPtr[rowId + 1] == rC.rowPtr[rowId + 1]);
    //if (rowId == 49) cout << "row49 hC range from " << hC.rowPtr[rowId] << " to"<< hC.rowPtr[rowId + 1] << endl;
    for (int cp = rC.rowPtr[rowId]; cp < rC.rowPtr[rowId + 1]; ++cp) {
      int col = rC.colInd[cp];
      if (fabs(rC.values[cp]) > 1e-8) {
        rowVals[col] = rC.values[cp];
      }
    }
    for (int cp = hC.rowPtr[rowId]; cp < hC.rowPtr[rowId + 1]; ++cp) {
      int col = hC.colInd[cp];
      //if (fabs(rowVals[col] - hC.values[cp]) > 1e-8) {
      float relativeError = fabs((rowVals[col] - hC.values[cp]) / rowVals[col]);
      if (relativeError >= 0.001 && fabs(hC.values[cp]) > 1e-8)  {
        printf("values[%d, %d] %e should be %e at cp=%d\n", rowId, col, hC.values[cp], rowVals[col], cp);
        flag = false;
        return false;
      }
    }
    for (int cp = rC.rowPtr[rowId]; cp < rC.rowPtr[rowId + 1]; ++cp) {
      int col = rC.colInd[cp];
      if (fabs(rC.values[cp]) > 1e-8) {
        rowVals[col] = 0.0;
      }
    }
  }
  free(rowVals);
  return flag;
}

bool resultsComparison(CSR &hC, CSR &rC, const vector<int> &hv, const int *hqueue) {
  bool isSame = hC.isRelativeEqual(rC, 0.0001);
  std::cout << "hC compare with rC: ";
  if (isSame) {
    std::cout << "Same\n";
  } else {
    std::cout << "Diffs\n";
  }
  std::cout << "rC compare with hC: ";
  isSame = rC.isRelativeEqual(hC, 0.0001);
  if (isSame) {
    std::cout << "Same\n";
  } else {
    std::cout << "Diffs\n";
  }

  if (hv.size() > 0 + 1) {
    printf("Checking a 1 %d\n", hv[0]);
    isPartialRawEqual(hv[0], hv[1], hqueue, hC, rC);
    isPartialRawEqual(hv[0], hv[1], hqueue, rC, hC);
  }
  if (hv.size() > 1 + 1) {
    printf("Checking fp 1 %d\n", hv[2] - hv[1]);
    isPartialRawEqual(hv[1], hv[2], hqueue, hC, rC);
    isPartialRawEqual(hv[1], hv[2], hqueue, rC, hC);
  }
  if (hv.size() > 2 + 1) {
    printf("Checking fp 2 %d\n", hv[3] - hv[2]);
    isPartialRawEqual(hv[2], hv[3], hqueue, hC, rC);
    isPartialRawEqual(hv[2], hv[3], hqueue, rC, hC);
  }
  if (hv.size() > 3 + 1) {
    printf("Checking fp l4 %d\n", hv[4] - hv[3]);
    isPartialRawEqual(hv[3], hv[4], hqueue, hC, rC);
    isPartialRawEqual(hv[3], hv[4], hqueue, rC, hC);
  }
  if (hv.size() > 4 + 1) {
    printf("Checking fp 8 %d\n", hv[5] - hv[4]);
    isPartialRawEqual(hv[4], hv[5], hqueue, hC, rC);
    isPartialRawEqual(hv[4], hv[5], hqueue, rC, hC);
  }
  if (hv.size() > 5 + 1) {
    printf("Checking fp 16 %d\n", hv[6] - hv[5]);
    isPartialRawEqual(hv[5], hv[6], hqueue, hC, rC);
    isPartialRawEqual(hv[5], hv[6], hqueue, rC, hC);
  }
  if (hv.size() > 6 + 1) {
    printf("Checking fp 32 %d\n", hv[7] - hv[6]);
    isPartialRawEqual(hv[6], hv[7], hqueue, hC, rC);
    isPartialRawEqual(hv[6], hv[7], hqueue, rC, hC);
  }
  if (hv.size() > 7 + 1) {
    printf("Checking fp 64 %d\n", hv[8] - hv[7]);
    isPartialRawEqual(hv[7], hv[8], hqueue, hC, rC);
    isPartialRawEqual(hv[7], hv[8], hqueue, rC, hC);
  }
  if (hv.size() > 8 + 1) {
    printf("Checking fp 128 %d\n", hv[9] - hv[8]);
    isPartialRawEqual(hv[8], hv[9], hqueue, hC, rC);
    isPartialRawEqual(hv[8], hv[9], hqueue, rC, hC);
  }
  if (hv.size() > 9 + 1) {
    printf("Checking fp 256 %d\n", hv[10] - hv[9]);
    isPartialRawEqual(hv[9], hv[10], hqueue, hC, rC);
    isPartialRawEqual(hv[9], hv[10], hqueue, rC, hC);
  }
  if (hv.size() > 10 + 1) {
    printf("Checking fp 512 %d\n", hv[11] - hv[10]);
    isPartialRawEqual(hv[10], hv[11], hqueue, hC, rC);
    isPartialRawEqual(hv[10], hv[11], hqueue, rC, hC);
  }
  if (hv.size() > 11 + 1) {
    printf("Checking fp larger than 512 count=%d\n", hv[63] - hv[11]);
    isPartialRawEqual(hv[11], hv[63], hqueue, hC, rC);
    isPartialRawEqual(hv[11], hv[63], hqueue, rC, hC);
  }
  assert (hv.size() == 65);
  printf("Checking >=128 nonzero entries in single A's row  count=%d\n", hv[64] - hv[63]);
  isPartialRawEqual(hv[63], hv.back(), hqueue, hC, rC);
  isPartialRawEqual(hv[63], hv.back(), hqueue, rC, hC);

}

CSR sgpuSpMMWrapper(const CSR &dA, const CSR &dB, int *drowIds, int* dv, const vector<int> &hv);

//CSR rC;
CSR scudaSpMM(const CSR &hA, const CSR &hB) {
  int m = hA.rows;
  int k = hA.cols;
  int n = hB.cols;
  CSR dA = hA.toGpuCSR();
  CSR dB = hB.toGpuCSR();
  timer t;
  long *hflops = (long*)malloc(m * sizeof(long)), *dflops = NULL;
  double now1 = time_in_mill_now();
  ompFlops(m, hA.rowPtr, hA.colInd, hB.rowPtr, hflops);
  double now2 = time_in_mill_now();
  int *dv = NULL;
  int *hqueue = (int*) malloc(m * sizeof(int)), *dqueue = NULL;
  vector<int> hv = classifyFlops(hflops, hA.rowPtr, m, hqueue);
  double now3 = time_in_mill_now();
  printf("time for flops=%lf ms classify=%lf ms\n", now2 - now1, now3 - now2);

  HANDLE_ERROR(cudaMalloc((void**)&dqueue, m * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy((void*)dqueue, (void*) hqueue, m * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void**)&dv, hv.size() * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy((void*)dv, (void*)(&hv[0]), hv.size() * sizeof(int), cudaMemcpyHostToDevice));
  timer t2;
  CSR dC = sgpuSpMMWrapper(dA, dB, dqueue, dv, hv);
  double te = t.milliseconds_elapsed();
  double t2e = t2.milliseconds_elapsed();
  printf("SpGEMM compute pass %lf milliseconds\n", t2e);
  printf("hvsize=%u\n", hv.size());
  printf("SpGEMM rowFlops and classify includes %lf milliseconds\n", te);
  HANDLE_ERROR(cudaFree(dqueue));
  HANDLE_ERROR(cudaFree(dv));
  CSR hC = dC.toCpuCSR();

  dC.deviceDispose();
  // CSR dC2 = cusparseSpMMWrapper(dA, dB);
  // CSR rC = dC2.toCpuCSR();
  // dC2.deviceDispose();
  CSR rC = hA.somp_spmm(hB, 512);
  resultsComparison(hC, rC, hv, hqueue);

  free(hflops); free(hqueue);
  dA.deviceDispose();
  dB.deviceDispose();
  return rC;
}

int main(int argc, char *argv[]) {
  process_args(argc, argv);
  print_args();
  //cusparse_init();
  COO cooAt;
  cooAt.readSNAPFile(options.inputFileName, false);
  printf("cooAt before removing Duplicates rows=%d cols=%d nnz=%d\n", cooAt.rows, cooAt.cols, cooAt.nnz);
  cooAt.orderedAndDuplicatesRemoving();
  printf("cooAt after removing Duplicates rows=%d cols=%d nnz=%d\n", cooAt.rows, cooAt.cols, cooAt.nnz);
  CSR hA = cooAt.toCSR();
  hA.toAbs();
  cooAt.dispose();
  //A.output("A");
  CSR hB = hA.deepCopy();
  //CSR ompC = hA.somp_spmm(hB, options.stride);
  //rC = ompC;
  CSR hC = scudaSpMM(hA, hB);


  hA.dispose();
  hB.dispose();
  //ompC.dispose();
  hC.dispose();
  //ompC.dispose();
  //cusparse_finalize("clear cusparse");
  return 0;
}