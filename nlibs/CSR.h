/*
 * CSR.h
 *
 *  Created on: Oct 3, 2013
 *      Author: niuq
 */

#ifndef CSR_H_
#define CSR_H_
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include "cpu_csr_kernel.h"
#ifdef enable_GPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif
#include <vector>
using namespace std;

struct CSR {
public:
/*A real or complex array that contains the non-zero elements of a sparse matrix.
 * The non-zero elements are mapped into the values array using the row-major upper
 * triangular storage mapping described above.*/
	double* values;

/*Element i of the integer array columns is the number of the column that
 * contains the i-th element in the values array.*/
	int* colInd;

/*Element j of the integer array rowIndex gives the index of the element
 * in the values array that is
 * first non-zero element in a row j.*/
	int* rowPtr;
	int rows, cols, nnz;

/* if allocate by malloc, isStatus=1
 * else if allocate by cudaMalloc isStatus=-1
 * else isStatus=0 */
	CSR() {
    this->values = NULL;
    this->colInd = NULL;
    this->rowPtr = NULL;
    this->rows = 0;
    this->cols = 0;
    this->nnz = 0;
	}

 CSR deepCopy();

	CSR(double* values, int* colInd, int* rowPtr, int rows, int cols, int nnz) {
    this->values = values;
    this->colInd = colInd;
    this->rowPtr = rowPtr;
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
	}

  void averAndNormRowValue();
  long spmmFlops(const CSR& B) const;
  std::vector<int> multiFlopsStats(const CSR& B) const;
  vector<int> nnzStats() const;
  CSR spmm(const CSR& B) const;
  CSR omp_spmm(const CSR& B, const int stride = 512) const;
  void output(const char* msg, bool isZeroBased = true) const {
    printf("%s\n", msg);
    if (isZeroBased) {
      for (int i = 0; i < rows; i++) {
        for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
          int col=colInd[j];
          double val=values[j];
          printf("%d\t%d\t%.6lf\n", i, col, val);
        }
      }
    } else {
      for (int i = 0; i < rows; i++) {
        for (int j = rowPtr[i] - 1; j < rowPtr[i+1] - 1; j++) {
          int col=colInd[j];
          double val=values[j];
          printf("%d\t%d\t%.6lf\n", i + 1, col, val);
        }
      }
    }
  }

  void output_int(const char* msg) const {
    printf("%s\n", msg);
    for (int i = 0; i < rows; i++) {
      for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
        int col=colInd[j];
        int val=((int*)values)[j];
        printf("%d\t%d\t%d\n", i, col, val);
      }
    }
  }

  void output_structure(const char* msg) const {
    printf("%s\n", msg);
    for (int i = 0; i < rows; i++) {
      for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
        int col=colInd[j];
        printf("%d\t%d\n", i, col);
      }
    }
  }

  void toOneBasedCSR() {
    for (int i = 0; i < rows; i++) {
      for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
        ++colInd[j];
      }
    }
    for (int i = 0; i < rows + 1; i++) {
      ++rowPtr[i];
    }
  }

  void toZeroBasedCSR() {
    for (int i = 0; i < rows + 1; i++) {
      --rowPtr[i];
    }
    for (int i = 0; i < rows; i++) {
      for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
        --colInd[j];
      }
    }
  }

  void makeOrdered();
  void matrixRowReorder(const int* ranks) const;

  //Both CSR should be called makeOrdered before call isEqual
  bool isEqual(const CSR &B) const {
    bool flag = true;
    if (rows != B.rows) {
      printf("rows = %d\tB_rows = %d\n", rows, B.rows);
      flag = false;
    }
    if (cols != B.cols) {
      printf("cols = %d\tB_cols = %d\n", cols, B.cols);
      flag = false;
    }
    if (nnz != B.nnz) {
      printf("nnz = %d\tB_nnz = %d\n", nnz, B.nnz);
      flag = false;
    }

    for (int i = 0; i < (rows + 1); ++i) {
      if (rowPtr[i] != B.rowPtr[i]) {
        printf("rowPtr[%d] %d\t%d\n", i, rowPtr[i], B.rowPtr[i]);
        flag = false;
        break;
      }
    }

    for (int i = 0; i < nnz; ++i) {
      if (colInd[i] != B.colInd[i]) {
        printf("colInd[%d] %d\t%d\n", i, colInd[i], B.colInd[i]);
        flag = false;
        break;
      }
    }

    for (int i = 0; i < nnz; ++i) {
      if (fabs(values[i] - B.values[i]) > 1e7) {
        printf("values[%d] %lf\t%lf\n", i, values[i], B.values[i]);
        flag = false;
        break;
      }
    }
    return flag;
  }

  void dispose() {
      free(values); values = NULL;
      free(colInd); colInd = NULL;
      free(rowPtr); rowPtr = NULL;
  }

  void deviceDispose();

  /*Default rowInflation gamma is 2*/
  void rowInflationR2(int rowId) const {
    for (int i = rowPtr[rowId]; i < rowPtr[rowId + 1]; ++i) {
      values[i] = values[i] * values[i];
    }
  }

  inline double rowMax(int rowId) const {
    double rmax = 0.0;
    for (int i = rowPtr[rowId]; i < rowPtr[rowId + 1]; ++i) {
      if (rmax < values[i]) {
        rmax = values[i];
      }
    }
    return rmax;
  }

  inline double rowSum(int rowId) const {
    double sum = 0.0;
    for (int i = rowPtr[rowId]; i < rowPtr[rowId + 1]; ++i) {
        sum += values[i];
    }
    return sum;
  }

  inline int rowCount(int rowId) const {
    return rowPtr[rowId + 1] - rowPtr[rowId];
  }

  CSR ompRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const;
  CSR mklRmclOneStep(const CSR &B, const int stride) const;
  CSR staticOmpRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const;
  CSR hybridOmpRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const;
  CSR staticFairRmclOneStep(const CSR &B, const int stride) const;
  CSR cilkRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const;
  double differs(const CSR& B) const;
  vector<int> differsStats(const CSR& B, vector<double> percents) const;
  CSR toGpuCSR() const;
  CSR toCpuCSR() const;
  long long spMMFlops(const CSR& B) const;
  void outputSpMMStats(const CSR& B) const;
  CSR PXM(const int P[]) const;
  CSR MXP(const int P[]) const;
};
#endif /* CSR_CUH_ */
