/*
 * CSR.cpp
 *
 *  Created on: Oct 3, 2013
 *      Author: niuq
 */
#include "CSR.h"
#include "tools/util.h"
#include "tools/qmalloc.h"
#include <vector>
#include <algorithm>
#include <omp.h>
#include "gpus/cuda_handle_error.h"
//#include "gpus/gpu_csr_kernel.h"

void CSR::matrixRowReorder(const int* ranks) const {
  int* nrowPtr = (int*)qmalloc((rows + 1) * sizeof(int), __FUNCTION__, __LINE__);
  int* ncolInd= (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  double* nvalues = (double*)qmalloc(nnz * sizeof(double), __FUNCTION__, __LINE__);
  nrowPtr[0] = 0;
  for (int i = 0; i < rows; ++i) {
    int count = rowPtr[ranks[i] + 1] - rowPtr[ranks[i]];
    nrowPtr[i + 1] = nrowPtr[i] + count;
    memcpy(ncolInd + nrowPtr[i], colInd + rowPtr[ranks[i]],
        count * sizeof(int));
    memcpy(nvalues + nrowPtr[i], values + rowPtr[ranks[i]],
        count * sizeof(double));
  }
  memcpy(rowPtr, nrowPtr, (rows + 1) * sizeof(int));
  memcpy(colInd, ncolInd, nnz * sizeof(int));
  memcpy(values, nvalues, nnz * sizeof(double));
  free(nrowPtr);
  free(ncolInd);
  free(nvalues);
}

long CSR::spmmFlops(const CSR& B) const {
  long flops = getSpMMFlops(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz, rows, cols, B.cols);
  return flops;
}

CSR CSR::spmm(const CSR& B) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  double* C;
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
    std::vector<std::pair<int, double> > rowv;
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
	double* bvalues = (double*)qmalloc(nnz * sizeof(double), __FUNCTION__, __LINE__);
  int* bcolInd = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  memcpy(browPtr, rowPtr, (rows + 1) * sizeof(int));
  memcpy(bvalues, values, nnz * sizeof(double));
  memcpy(bcolInd, colInd, nnz * sizeof(int));
  CSR B(bvalues, bcolInd, browPtr, rows, cols, nnz);
  return B;
}

CSR CSR::omp_spmm(const CSR& B) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  double* C;
  int nnzC;
  omp_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

/* This method returns the norm of A-B. Remember, it assumes
 * that the adjacency lists in both A and B are sorted in
 * ascending order. */
double CSR::differs(const CSR& B) const {
  double sum = 0;
  int i, j, k;
  for (i = 0; i < rows; ++i) {
    for (j = rowPtr[i], k = B.rowPtr[i];
        j < rowPtr[i + 1] && k < B.rowPtr[i + 1];) {
      double a = values[j];
      double b = B.values[k];
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

CSR CSR::rmclOneStep(const CSR &B, thread_data_t *thread_datas) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  double* C;
  int nnzC;
  omp_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::toGpuCSR() const {
  CSR dA;
  dA.rows = this->rows;
  dA.cols = this->cols;
  dA.nnz = this->nnz;
  cudaMalloc((void**)&dA.rowPtr, sizeof(int) * (rows + 1));
  cudaMemcpy(dA.rowPtr, rowPtr, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dA.colInd, sizeof(int) * nnz);
  cudaMemcpy(dA.colInd, colInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dA.values, sizeof(double) * nnz);
  cudaMemcpy(dA.values, values, sizeof(double) * nnz, cudaMemcpyHostToDevice);
  return dA;
}

CSR CSR::toCpuCSR() const {
  CSR hA;
  hA.rows = this->rows;
  hA.cols = this->cols;
  hA.nnz = this->nnz;
  hA.rowPtr = (int*)qmalloc(sizeof(int) * (rows + 1), __FUNCTION__, __LINE__);
  HANDLE_ERROR(cudaMemcpy(hA.rowPtr, rowPtr, sizeof(int) * (rows + 1), cudaMemcpyDeviceToHost));
  hA.colInd = (int*)qmalloc(sizeof(int) * nnz, __FUNCTION__, __LINE__);
  HANDLE_ERROR(cudaMemcpy(hA.colInd, colInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost));
  hA.values = (double*)qmalloc(sizeof(double) * nnz, __FUNCTION__, __LINE__);
  HANDLE_ERROR(cudaMemcpy(hA.values, values, sizeof(double) * nnz, cudaMemcpyDeviceToHost));
  return hA;
}

void CSR::deviceDispose() {
  cudaFree(values); values = NULL;
  cudaFree(colInd); colInd = NULL;
  cudaFree(rowPtr); rowPtr = NULL;
}

vector<int> CSR::differsStats(const CSR& B, const vector<double> percents) const {
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
      double percent = (bcount - acount) / (double)acount;
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
