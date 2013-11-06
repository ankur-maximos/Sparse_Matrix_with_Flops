#ifndef CUSPARSE_SPMM_H_
#define CUSPARSE_SPMM_H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cusparse_v2.h>
#include "timer.h"

#define CLEANUP(s)                                   \
do {                                                 \
    printf ("%s\n", s);                              \
    if (descrA)              cusparseDestroyMatDescr(descrA);\
    if (descrB)              cusparseDestroyMatDescr(descrB);\
    if (descrC)              cusparseDestroyMatDescr(descrC);\
    if (handle)             cusparseDestroy(handle); \
    cudaDeviceReset();          \
    fflush (stdout);                                 \
} while (0)
int cusparse_init(void);
void cusparseXcsrgemmNnzWrapper(const int dIA[], const int dJA[], const int nnzA,
        const int dIB[], const int dJB[], const int nnzB,
        const int m, const int k, const int n,
        int* IC, int& nnzC);
void cusparseDcsrgemmWapper(const int* const dIA, const int dJA[], const double dA[], const int nnzA,
        const int dIB[], const int dJB[], const double dB[], const int nnzB,
        const int* dIC, int* dJC, const double* dC, const int nnzC,
        const int m, const int k, const int n);
void cusparse_finalize(const char *msg);
#endif