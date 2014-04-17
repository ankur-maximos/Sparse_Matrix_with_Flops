#ifndef BCSR_H_
#define BCSR_H_
#include "CSR.h"
#include "tools/util.h"

struct BCSR {
  int r, c;
  int rows, cols;
  int brows, bcols;

	Value* values;
	int* colInd;
	int* rowPtr;
  BCSR(const CSR &csr, const int c, const int r);

  BCSR(int rows, int cols, int r, int c) {
    this->rows = rows;
    this->cols = cols;
    this->r = r; this->c = c;
    brows = (rows + r - 1) / r;
    bcols = (cols + c - 1) / c;
    values = NULL;
    colInd = NULL;
    rowPtr = NULL;
  }

  void output(const char* msg, bool isZeroBased = true) const {
    printf("%s\n", msg);
    arrayOutput("rowPtr", stdout, rowPtr, brows + 1);
    int nnzb = rowPtr[brows];
    arrayOutput("colInd", stdout, colInd, nnzb);
    arrayOutput("values", stdout, values, nnzb * r * c);
    for (int bi = 0; bi < brows; ++bi) {
      for (int bj = rowPtr[bi]; bj < rowPtr[bi + 1]; bj++) {
        int bcol = colInd[bj];
        printf("%d\t%d\n", bi, bcol);
        Value* bvalues = values + bj * r * c;
        for (int ei = 0; ei < r; ++ei) {
          for (int ej = 0; ej < c; ++ej) {
            Value val = bvalues[ei * c + ej];
            printf("%.6lf\t", val);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
  }

  void dispose() {
    free(values);
    free(colInd);
    free(rowPtr);
  }

  bool isEqual(const CSR &B) const;
};
#endif
