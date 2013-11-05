#include "cusparse_spmm.h"

bool isCusparseInit = false;
cusparseStatus_t status;
cusparseHandle_t handle  = 0;
cusparseMatDescr_t descrA = 0;
cusparseMatDescr_t descrB = 0;
cusparseMatDescr_t descrC = 0;
int cusparse_init(void) {
    /* initialize cusparse library */
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library initialization failed");
        return 1;
    }

    /* create and setup matrix descriptor */
    status = cusparseCreateMatDescr(&descrA);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);

    status = cusparseCreateMatDescr(&descrB);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);

    status = cusparseCreateMatDescr(&descrC);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);

    return 0;
}

void gpuCsrSpMM(const int dIA[], const int dJA[], const double dA[], const int nnzA,
        const int dIB[], const int dJB[], const double dB[], const int nnzB,
        int* &dIC, int* &dJC, double* &dC, int& nnzC,
        const int m, const int k, const int n) {
}

void cusparseXcsrgemmNnzWrapper(const int dIA[], const int dJA[], const int nnzA,
        const int dIB[], const int dJB[], const int nnzB,
        const int m, const int k, const int n,
        int* IC, int& nnzC) {
  cudaMalloc((void**)&IC, sizeof(int) * (m + 1));
  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  status = cusparseXcsrgemmNnz(handle, transA, transB, m, n, k,
      descrA, nnzA, dIA, dJA,
      descrB, nnzB, dIB, dJB,
      descrC, IC, &nnzC);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("CSR Matrix-Matrix multiplication failed");
  }
}

void cusparseDcsrgemmWapper(const int* const dIA, const int dJA[], const double dA[], const int nnzA,
        const int dIB[], const int dJB[], const double dB[], const int nnzB,
        const int* dIC, int* dJC, double* dC, const int nnzC,
        const int m, const int k, const int n) {
  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  status = cusparseDcsrgemm(handle, transA, transB, m, n, k,
      descrA, nnzA,
      dA, dIA, dJA,
      descrB, nnzB,
      dB, dIB, dJB,
      descrC,
      dC, dIC, dJC);

  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("CSR Matrix-Matrix multiplication failed");
  }
}

void cusparse_finalize(const char *msg) {
  CLEANUP(msg);
}
