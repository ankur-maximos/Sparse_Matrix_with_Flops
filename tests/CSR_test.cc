#include "CSR.h"
#include "COO.h"

int main() {
  int rows = 4, cols = 4, nnz = 5;
  const int rowIndex[] = {0, 1, 2, 3, 3};
  const int colIndex[] = {1, 2, 0, 1, 3};
  const double values[] = {2.0, 3.0, 4.0, 1.0, 5.0};
  COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  CSR csr = coo.toCSR();
  csr.averAndNormRowValue();
  csr.output("csr");
  return 0;
}
