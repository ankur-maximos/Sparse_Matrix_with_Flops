/*
 * COO.cpp
 *
 *  Created on: Oct 3, 2013
 *      Author: niuq
 */

#include "COO.h"
#include "tools/qmalloc.h"
//#include <tuple>
#include <vector>
#include <algorithm>

COO::COO() {
	cooRowIndex=cooColIndex=NULL;
	cooVal=NULL;
	nnz=0; rows=cols=0;
}

COO::COO(const char fname[]) {
  this->readMatrixMarketFile(fname);
}

COO::COO(const QValue* const cooVal, const int* const cooColIndex,
      const int* const cooRowIndex, const int rows, const int cols, const int nnz) {
  this->rows = rows;
  this->cols = cols;
  this->nnz = nnz;
  this->cooColIndex = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  this->cooRowIndex = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  this->cooVal = (QValue*)qmalloc(nnz * sizeof(QValue), __FUNCTION__, __LINE__);
  memcpy(this->cooColIndex, cooColIndex, nnz * sizeof(int));
  memcpy(this->cooRowIndex, cooRowIndex, nnz * sizeof(int));
  memcpy(this->cooVal, cooVal, nnz * sizeof(QValue));
}

void COO::dispose() {
	free(cooRowIndex); cooRowIndex = NULL;
	free(cooColIndex); cooColIndex = NULL;
	free(cooVal); cooVal = NULL;
}

void COO::readMatrixMarketFile(const char fname[]) {
	mm_read_unsymmetric_sparse(fname, &rows, &cols, &nnz,
            &cooVal, &cooRowIndex, &cooColIndex);
}

void COO::readTransposedSNAPFile(const char fname[]) {
  FILE *fpin;
  if ((fpin = fopen(fname, "r")) == NULL) {
    printf("Failed to open file %s\n", fname);
    exit(-1);
  }
  const int MAX_LINE = 500;
  char line[MAX_LINE];
  do {
    fgets(line, MAX_LINE, fpin);
  } while (line[0] == '#' && !feof(fpin));
  if (feof(fpin)) {
    nnz = 0;
    return;
  }
  sscanf(line, "%d %d", &(this->rows), &(this->nnz));
  this->cols = this->rows;
  cooRowIndex = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  cooColIndex = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  cooVal = (QValue*)qmalloc(nnz * sizeof(QValue), __FUNCTION__, __LINE__);
  int from, to;
  for (int i = 0; i < nnz; ++i) {
    fscanf(fpin, "%d%d", &from, &to);
    //Reverse row and col so that it is transposed.
    cooRowIndex[i] = to;
    cooColIndex[i] = from;
    cooVal[i] = 1.0;
  }
}

void COO::addSelfLoopIfNeeded() {
  assert(rows == cols);
  bool* u = (bool*)calloc(rows, sizeof(bool));
  int count = 0;
  for (int i = 0; i < nnz; ++i) {
    int row = cooRowIndex[i];
    int col = cooColIndex[i];
    if (row == col) {
      u[row] = true;
      count++;
    }
  }
  int needed = rows - count;
  int oldNnz = nnz;
  nnz += needed;
  cooRowIndex = (int*)realloc(cooRowIndex, nnz * sizeof(int));
  cooColIndex = (int*)realloc(cooColIndex, nnz * sizeof(int));
  cooVal = (QValue*)realloc(cooVal, nnz * sizeof(QValue));
  int top = oldNnz;
  for (int i = 0; i < rows; ++i) {
    if (u[i]) {
      continue;
    }
    cooRowIndex[top] = i;
    cooColIndex[top] = i;
    cooVal[top++] = 1.0;
  }
  free(u);
}

void COO::output(const char* msg) {
  printf("%s\n", msg);
  for(int i=0;i<nnz;i++)
  {
    printf("%d %d %lf\n", cooRowIndex[i], cooColIndex[i], cooVal[i]);
  }
  printf("host output end\n");
}

struct COOTuple {
  int rowIndex;
  int colIndex;
  QValue val;
};

bool operator < (const COOTuple &A, const COOTuple &B) {
  return A.rowIndex < B.rowIndex || (A.rowIndex == B.rowIndex && A.colIndex < B.colIndex);
}


COOTuple makeCOOTuple(int rowIndex, int colIndex, QValue val) {
  COOTuple cooTuple;
  cooTuple.rowIndex = rowIndex;
  cooTuple.colIndex = colIndex;
  cooTuple.val = val;
}

void COO::makeOrdered() const {
  typedef COOTuple iid;
  std::vector<iid> v(nnz);
  for (int i = 0; i < nnz; ++i) {
    v[i] = (makeCOOTuple(cooRowIndex[i], cooColIndex[i], cooVal[i]));
  }
  std::sort(v.begin(), v.end());
  for (int i = 0; i < nnz; ++i) {
    cooRowIndex[i] = v[i].rowIndex;
    cooColIndex[i] = v[i].colIndex;
    cooVal[i] = v[i].val;
  }
}

CSR COO::toCSR() const {
  int row = 0;
	int* ocsrRowPtr = (int*)qmalloc(sizeof(int) * (rows + 1), __FUNCTION__, __LINE__);
	memset(ocsrRowPtr, -1, sizeof(int) * (rows + 1));
	for (int t = 0; t < nnz; ++t) {
		while(row < cooRowIndex[t] && row < rows && ocsrRowPtr[row] == -1)
		  ocsrRowPtr[row++] = t;
		if(row == cooRowIndex[t] && ocsrRowPtr[row] == -1)
		  ocsrRowPtr[row++] = t;
	}
	ocsrRowPtr[rows] = nnz;
	int onnz = nnz;
	int* ocsrColInd = (int*)qmalloc(sizeof(int) * onnz, __FUNCTION__, __LINE__);
	QValue* ocsrVals = (QValue*)qmalloc(sizeof(QValue) * onnz, __FUNCTION__, __LINE__);
	memcpy(ocsrColInd, cooColIndex, sizeof(int) * onnz);
	memcpy(ocsrVals, cooVal, sizeof(QValue) * onnz);
	CSR csr(ocsrVals, ocsrColInd, ocsrRowPtr, rows, cols, onnz);
	return csr;
}


