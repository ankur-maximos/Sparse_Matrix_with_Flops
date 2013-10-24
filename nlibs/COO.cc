/*
 * COO.cpp
 *
 *  Created on: Oct 3, 2013
 *      Author: niuq
 */

#include "COO.h"
#include <tuple>
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

COO::~COO() {
	free(cooRowIndex);
	free(cooColIndex);
	free(cooVal);
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
  cooRowIndex = (int*)malloc(nnz * sizeof(int));
  cooColIndex = (int*)malloc(nnz * sizeof(int));
  cooVal = (double*)malloc(nnz * sizeof(double));
  int from, to;
  for (int i = 0; i < nnz; ++i) {
    fgets(line, MAX_LINE, fpin);
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
  cooVal = (double*)realloc(cooVal, nnz * sizeof(double));
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

void COO::makeOrdered() const {
  typedef std::tuple<int, int, double> iid;
  std::vector<iid> v(nnz);
  for (int i = 0; i < nnz; ++i) {
    v[i] = (std::make_tuple(cooRowIndex[i], cooColIndex[i], cooVal[i]));
  }
  std::sort(v.begin(), v.end());
  for (int i = 0; i < nnz; ++i) {
    cooRowIndex[i] = std::get<0>(v[i]);
    cooColIndex[i] = std::get<1>(v[i]);
    cooVal[i] = std::get<2>(v[i]);
  }
}

CSR COO::toCSR() const {
    int row=0;
	int* ocsrRowPtr=(int*)malloc(sizeof(int)*(rows+1));
	memset(ocsrRowPtr,-1,sizeof(int)*(rows+1));
	for(int t=0;t<nnz;t++) {
		while(row<cooRowIndex[t] && row<rows && ocsrRowPtr[row]==-1)
		  ocsrRowPtr[row++]=t;
		if(row==cooRowIndex[t] && ocsrRowPtr[row]==-1)
		  ocsrRowPtr[row++]=t;
	}
	ocsrRowPtr[rows]=nnz;
	int onnz=nnz;
	int* ocsrColInd=(int*)malloc(sizeof(int)*onnz);
	double* ocsrVals=(double*)malloc(sizeof(double)*onnz);
	memcpy(ocsrColInd, cooColIndex, sizeof(int)*onnz);
	memcpy(ocsrVals, cooVal, sizeof(double)*onnz);
	CSR csr(ocsrVals, ocsrColInd, ocsrRowPtr, rows, cols, onnz);
	return csr;
}


