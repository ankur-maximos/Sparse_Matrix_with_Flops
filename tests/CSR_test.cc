#include "CSR.h"
#include "COO.h"

void CSR_PM_Test() {
  int rows = 4, cols = 4, nnz = 5;
  const int rowIndex[] = {0, 1, 2, 3, 3};
  const int colIndex[] = {1, 2, 0, 1, 3};
  const double values[] = {2.0, 3.0, 4.0, 1.0, 5.0};
  COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  CSR M = coo.toCSR();
  coo.dispose();
  M.output("\nM");
  int P[] = {1, 3, 0, 2};
  CSR pM = M.PM(P);
  pM.output("\nPM");
  int prowPtr[] = {0, 1, 2, 3, 4};
  int pcolInd[] = {1, 3, 0, 2};
  double pValues[] = {1.0, 1.0, 1.0, 1.0};
  COO cooP(pValues, pcolInd, prowPtr, 4, 4, 4);
  CSR csrP = cooP.toCSR();
  CSR mPM = csrP.spmm(M);
  M.dispose();
  mPM.makeOrdered();
  pM.makeOrdered();
  bool isSame = mPM.isEqual(pM);
  mPM.output("mPM");
  pM.dispose();
  assert(isSame == true);
  printf("%s Passed\n", __func__);
}

void CSR_MP_Test() {
  int rows = 4, cols = 4, nnz = 5;
  const int rowIndex[] = {0, 1, 2, 3, 3};
  const int colIndex[] = {1, 2, 0, 1, 3};
  const double values[] = {2.0, 3.0, 4.0, 1.0, 5.0};
  COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  CSR M = coo.toCSR();
  coo.dispose();
  int P[] = {1, 3, 0, 2};
  CSR mP = M.MP(P);
#ifdef DEBUG
  M.output("\nM");
  mP.output("\nMP");
#endif
  int prowPtr[] = {0, 1, 2, 3, 4};
  int pcolInd[] = {1, 3, 0, 2};
  double pValues[] = {1.0, 1.0, 1.0, 1.0};
  COO cooP(pValues, pcolInd, prowPtr, 4, 4, 4);
  CSR csrP = cooP.toCSR();
  CSR mMP = M.spmm(csrP);
  M.dispose();
  mMP.makeOrdered();
  mP.makeOrdered();
  bool isSame = mMP.isEqual(mP);
#ifdef DEBUG
  mMP.output("mMP");
#endif
  mP.dispose();
  assert(isSame == true);
  printf("%s Passed\n", __func__);
}

int main() {
  int rows = 4, cols = 4, nnz = 5;
  const int rowIndex[] = {0, 1, 2, 3, 3};
  const int colIndex[] = {1, 2, 0, 1, 3};
  const double values[] = {2.0, 3.0, 4.0, 1.0, 5.0};
  COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  CSR csr = coo.toCSR();
  csr.averAndNormRowValue();
  csr.output("csr");
  CSR_PM_Test();
  CSR_MP_Test();
  return 0;
}
