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
