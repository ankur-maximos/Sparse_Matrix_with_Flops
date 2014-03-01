#include "CSR.h"
#include "COO.h"

void CSR_PXM_Test() {
  int rows = 4, cols = 4, nnz = 5;
  const int rowIndex[] = {0, 1, 2, 3, 3};
  const int colIndex[] = {1, 2, 0, 1, 3};
  const double values[] = {2.0, 3.0, 4.0, 1.0, 5.0};
  COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  CSR M = coo.toCSR();
  coo.dispose();
  M.output("\nM");
  int P[] = {1, 3, 0, 2};
  CSR PM = M.PXM(P);
  PM.output("\nPM");
  int prowPtr[] = {0, 1, 2, 3, 4};
  int pcolInd[] = {1, 3, 0, 2};
  double pValues[] = {1.0, 1.0, 1.0, 1.0};
  COO cooP(pValues, pcolInd, prowPtr, 4, 4, 4);
  CSR csrP = cooP.toCSR();
  CSR mPM = csrP.spmm(M);
  M.dispose();
  mPM.makeOrdered();
  PM.makeOrdered();
  bool isSame = mPM.isEqual(PM);
  mPM.output("mPM");
  PM.dispose();
  assert(isSame == true);
  printf("%s Passed\n", __func__);
}

void CSR_MXP_Test() {
  int rows = 4, cols = 4, nnz = 5;
  const int rowIndex[] = {0, 1, 2, 3, 3};
  const int colIndex[] = {1, 2, 0, 1, 3};
  const double values[] = {2.0, 3.0, 4.0, 1.0, 5.0};
  COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  CSR M = coo.toCSR();
  coo.dispose();
  int P[] = {1, 3, 0, 2};
  CSR MP = M.MXP(P);
#ifdef DEBUG
  M.output("\nM");
  MP.output("\nMP");
#endif
  int prowPtr[] = {0, 1, 2, 3, 4};
  int pcolInd[] = {1, 3, 0, 2};
  double pValues[] = {1.0, 1.0, 1.0, 1.0};
  COO cooP(pValues, pcolInd, prowPtr, 4, 4, 4);
  CSR csrP = cooP.toCSR();
  CSR mMP = M.spmm(csrP);
  M.dispose();
  mMP.makeOrdered();
  MP.makeOrdered();
  bool isSame = mMP.isEqual(MP);
#ifdef DEBUG
  mMP.output("mMP");
#endif
  MP.dispose();
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
  CSR_PXM_Test();
  CSR_MXP_Test();
  return 0;
}
