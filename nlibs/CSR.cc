/*
 * CSR.cpp
 *
 *  Created on: Oct 3, 2013
 *      Author: niuq
 */
#include "CSR.h"
#include "tools/util.h"
#include "tools/stats.h"
#include "tools/qmalloc.h"
#include "mkls/mkl_csr_kernel.h"
#include <vector>
#include <algorithm>
#include <omp.h>
#ifdef enable_GPU
#include "gpus/cuda_handle_error.h"
#endif
//#include "gpus/gpu_csr_kernel.h"

void CSR::matrixRowReorder(const int* ranks) const {
  int* nrowPtr = (int*)qmalloc((rows + 1) * sizeof(int), __FUNCTION__, __LINE__);
  int* ncolInd= (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  Value* nvalues = (Value*)qmalloc(nnz * sizeof(Value), __FUNCTION__, __LINE__);
  nrowPtr[0] = 0;
  for (int i = 0; i < rows; ++i) {
    int count = rowPtr[ranks[i] + 1] - rowPtr[ranks[i]];
    nrowPtr[i + 1] = nrowPtr[i] + count;
    memcpy(ncolInd + nrowPtr[i], colInd + rowPtr[ranks[i]],
        count * sizeof(int));
    memcpy(nvalues + nrowPtr[i], values + rowPtr[ranks[i]],
        count * sizeof(Value));
  }
  memcpy(rowPtr, nrowPtr, (rows + 1) * sizeof(int));
  memcpy(colInd, ncolInd, nnz * sizeof(int));
  memcpy(values, nvalues, nnz * sizeof(Value));
  free(nrowPtr);
  free(ncolInd);
  free(nvalues);
}

long CSR::spmmFlops(const CSR& B) const {
  long flops = getSpMMFlops(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz, rows, cols, B.cols);
  return flops;
}

std::vector<int> CSR::multiFlopsStats(const CSR& B) const {
  std::vector<int> stats = flopsStats(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz, rows, cols, B.cols);
  return stats;
}


CSR CSR::spmm(const CSR& B) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  Value* C;
  int nnzC;
  sequential_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

void CSR::makeOrdered() {
  for (int i = 0; i < rows; ++i) {
    std::vector<std::pair<int, Value> > rowv;
    for (int jp = rowPtr[i]; jp < rowPtr[i + 1]; ++jp) {
      rowv.push_back(std::make_pair(colInd[jp], values[jp]));
    }
    std::sort(rowv.begin(), rowv.end());
    int iter = 0;
    for (int jp = rowPtr[i]; jp < rowPtr[i + 1]; ++jp, ++iter) {
      colInd[jp] = rowv[iter].first;
      values[jp] = rowv[iter].second;
    }
  }
}

void CSR::averAndNormRowValue() {
  for (int i = 0; i < rows; ++i) {
    int count = rowPtr[i + 1] - rowPtr[i];
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
      values[j] = 1.0 / count;
    }
  }
}

CSR CSR::deepCopy() {
  int* browPtr = (int*)qmalloc((rows + 1) * sizeof(int), __FUNCTION__, __LINE__);
	Value* bvalues = (Value*)qmalloc(nnz * sizeof(Value), __FUNCTION__, __LINE__);
  int* bcolInd = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  memcpy(browPtr, rowPtr, (rows + 1) * sizeof(int));
  memcpy(bvalues, values, nnz * sizeof(Value));
  memcpy(bcolInd, colInd, nnz * sizeof(int));
  CSR B(bvalues, bcolInd, browPtr, rows, cols, nnz);
  return B;
}

CSR CSR::omp_spmm(const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  Value* C;
  int nnzC;
  omp_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::omp_spmm(thread_data_t* thread_datas, const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  Value* C;
  int nnzC;
  int nthreads = 8;
#pragma omp parallel
#pragma omp master
  nthreads = omp_get_num_threads();
  static_omp_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::flops_spmm(const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  Value* C;
  int nnzC;
  flops_omp_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

/* This method returns the norm of A-B. Remember, it assumes
 * that the adjacency lists in both A and B are sorted in
 * ascending order. */
Value CSR::differs(const CSR& B) const {
  Value sum = 0;
  int i, j, k;
  for (i = 0; i < rows; ++i) {
    for (j = rowPtr[i], k = B.rowPtr[i];
        j < rowPtr[i + 1] && k < B.rowPtr[i + 1];) {
      Value a = values[j];
      Value b = B.values[k];
      if (colInd[j] == colInd[k]) {
        sum += (a - b) * (a - b);
        ++j, ++k;
      } else if (colInd[j] < colInd[k]){
        sum += a * a;
        ++j;
      } else {
        sum += b * b;
        ++k;
      }
    }
    for (; j < rowPtr[i + 1]; ++j) {
      sum += values[j] * values[j];
    }
    for (; k < rowPtr[i + 1]; ++k) {
      sum += B.values[k] * B.values[k];
    }
  }
  return sum;
}

vector<int> CSR::nnzStats() const {
  std::vector<int> stats(18, 0);
  for (int i = 0; i < rows; ++i) {
    long stat = rowPtr[i + 1] - rowPtr[i];
    pushToStats(rowPtr[i + 1] - rowPtr[i], stats);
  }
  return stats;
}

CSR CSR::ompRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  Value* C;
  int nnzC;
  omp_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::staticOmpRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  Value* C;
  int nnzC;
  static_omp_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::hybridOmpRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  Value* C;
  int nnzC;
  hybrid_omp_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::staticFairRmclOneStep(const CSR &B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  Value* C;
  int nnzC;
  static_fair_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

//Input A and B matrix are one based index. Output C is also one based index.
CSR CSR::mklRmclOneStep(const CSR &B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  Value* C;
  int nnzC;
  mkl_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::cilkRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  Value* C;
  int nnzC;
  cilk_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

#ifdef enable_GPU
CSR CSR::toGpuCSR() const {
  CSR dA;
  dA.rows = this->rows;
  dA.cols = this->cols;
  dA.nnz = this->nnz;
  cudaMalloc((void**)&dA.rowPtr, sizeof(int) * (rows + 1));
  cudaMemcpy(dA.rowPtr, rowPtr, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dA.colInd, sizeof(int) * nnz);
  cudaMemcpy(dA.colInd, colInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dA.values, sizeof(Value) * nnz);
  cudaMemcpy(dA.values, values, sizeof(Value) * nnz, cudaMemcpyHostToDevice);
  return dA;
}
#endif

#ifdef enable_GPU
CSR CSR::toCpuCSR() const {
  CSR hA;
  hA.rows = this->rows;
  hA.cols = this->cols;
  hA.nnz = this->nnz;
  hA.rowPtr = (int*)qmalloc(sizeof(int) * (rows + 1), __FUNCTION__, __LINE__);
  HANDLE_ERROR(cudaMemcpy(hA.rowPtr, rowPtr, sizeof(int) * (rows + 1), cudaMemcpyDeviceToHost));
  hA.colInd = (int*)qmalloc(sizeof(int) * nnz, __FUNCTION__, __LINE__);
  HANDLE_ERROR(cudaMemcpy(hA.colInd, colInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost));
  hA.values = (Value*)qmalloc(sizeof(Value) * nnz, __FUNCTION__, __LINE__);
  HANDLE_ERROR(cudaMemcpy(hA.values, values, sizeof(Value) * nnz, cudaMemcpyDeviceToHost));
  return hA;
}
#endif

#ifdef enable_GPU
void CSR::deviceDispose() {
  cudaFree(values); values = NULL;
  cudaFree(colInd); colInd = NULL;
  cudaFree(rowPtr); rowPtr = NULL;
}
#endif

vector<int> CSR::differsStats(const CSR& B, const vector<Value> percents) const {
  vector<int> counts(percents.size() + 4, 0);
  const int PINFI = percents.size() + 1;
  const int ZEROS = PINFI + 1;
  const int EQUALS = ZEROS + 1;
  for (int i = 0; i < rows; ++i) {
    int acount = rowPtr[i + 1] - rowPtr[i];
    int bcount = B.rowPtr[i + 1] - B.rowPtr[i];
    if (acount == 0 && bcount > 0) {
      ++counts[PINFI];
    } else if (acount == 0 && bcount == 0) {
      ++counts[ZEROS];
    } else if (acount == bcount) {
      ++counts[EQUALS];
    } else {
      Value percent = (bcount - acount) / (Value)acount;
      int k;
      for (k = 0; k < percents.size(); ++k) {
        if (percent < percents[k]) {
          ++counts[k];
          break;
        }
      }
      if (k == percents.size()) {
        ++counts[percents.size()];
      }
    }
  }
  int countSum = 0;
  for (int k = 0; k < counts.size(); ++k) {
    countSum += counts[k];
  }
  assert(countSum == rows);
  return counts;
}

long long CSR::spMMFlops(const CSR &B) const {
  return getSpMMFlops(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      this->rows, this->cols, B.cols);
}

void CSR::outputSpMMStats(const CSR &B) const {
  long long flops = this->spMMFlops(B);
  CSR C = this->omp_spmm(B, 512);
  int cNnz = C.nnz;
  C.dispose();
  printf("flops=%lld\tcNnz=%d\trows=%d\tflops/rows=%lf cnnz/rows=%lf\n", flops, cNnz, rows, (Value)(flops) / rows, (Value)(cNnz) / rows);
}

CSR CSR::PM(const int P[]) const {
  int* browPtr = (int*)qmalloc((rows + 1) * sizeof(int), __FUNCTION__, __LINE__);
	Value* bvalues = (Value*)qmalloc(nnz * sizeof(Value), __FUNCTION__, __LINE__);
  int* bcolInd = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  browPtr[0] = 0;
  for (int i = 0; i < rows; ++i) {
    int target = P[i];
    int count = rowPtr[target + 1] - rowPtr[target];
    memcpy(bvalues + browPtr[i], values + rowPtr[target], count * sizeof(Value));
    memcpy(bcolInd + browPtr[i], colInd + rowPtr[target], count * sizeof(int));
    browPtr[i + 1] = browPtr[i] + count;
  }
  CSR pM(bvalues, bcolInd, browPtr, rows, cols, nnz);
  return pM;
}

CSR CSR::MP(const int P[]) const {
  int* browPtr = (int*)qmalloc((rows + 1) * sizeof(int), __FUNCTION__, __LINE__);
  memcpy(browPtr, rowPtr, (rows + 1) * sizeof(int));
	Value* bvalues = (Value*)qmalloc(nnz * sizeof(Value), __FUNCTION__, __LINE__);
  memcpy(bvalues, values, nnz * sizeof(Value));
  int* bcolInd = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  for (int i = 0; i < rows; ++i) {
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
      bcolInd[j] = P[colInd[j]];
#ifdef DEBUG
      printf("row %d %d->%d\t", i, colInd[j], P[colInd[j]]);
      printf("%d %d %lf\n", i, bcolInd[j], bvalues[j]);
#endif
    }
  }
  CSR mP(bvalues, bcolInd, browPtr, rows, cols, nnz);
  return mP;
}

CSR CSR::PMPt(const int P[]) const {
  CSR pm = PM(P);
  int *Pt = permutationTranspose(P, rows);
  CSR pMPt = pm.MP(Pt);
  pm.dispose();
  free(Pt);
  return pMPt;
}

CSR CSR::PtMP(const int P[]) const {
  CSR mP = MP(P);
  int *Pt = permutationTranspose(P, rows);
  CSR ptMP = mP.PM(Pt);
  mP.dispose();
  free(Pt);
  return ptMP;
}
