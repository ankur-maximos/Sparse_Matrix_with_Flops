/*
 * CSR.cpp
 *
 *  Created on: Oct 3, 2013
 *      Author: niuq
 */

#include "CSR.h"
#include <vector>
#include <algorithm>

void CSR::matrixRowReorder(const int* ranks) const {
  int* nrowPtr = (int*)malloc((rows + 1) * sizeof(int));
  int* ncolInd= (int*)malloc(nnz * sizeof(int));
  double* nvalues = (double*)malloc(nnz * sizeof(double));
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
  int* browPtr = (int*)malloc(rows * sizeof(int));
	double* bvalues = (double*)malloc(nnz * sizeof(double));
  int* bcolInd = (int*)malloc(nnz * sizeof(int));;
  memcpy(browPtr, rowPtr, rows * sizeof(int));
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

double CSR::differs(const CSR& B) const {
  double sum = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = rowPtr[i], k = B.rowPtr[i];
        j < rowPtr[i + 1] && k < B.rowPtr[i + 1];) {
    }
  }
  return sum;
}
