#include "CSR.h"
#include "COO.h"
#include "BCSR.h"
#include <assert.h>

void testBCSRisEqual() {
  int rows = 4, cols = 4, nnz = 5;
  const int rowIndex[] = {0, 1, 2, 3, 3};
  const int colIndex[] = {1, 1, 0, 1, 3};
  const Value values[] = {2.0, 3.0, 4.0, 1.0, 5.0};
  const int dM[4][4] = {
    {0.0, 2.0, 0.0, 0.0},
    {0.0, 3.0, 0.0, 0.0},
    {4.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 5.0}
  };
  COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  CSR A = coo.toCSR();
  coo.dispose();
  BCSR bA(A, 2, 2);
  bA.output("block A");
  A.output("A");
  bool flag = bA.isEqual(A);
  bA.dispose();
  A.dispose();
  assert(flag == true);
  printf("%s Passed\n", __func__);
}

void testBCSRisEqualBorderCase() {
  int rows = 5, cols = 5, nnz = 7;
  const int rowIndex[] = {0, 1, 1, 2, 3, 3, 4};
  const int colIndex[] = {1, 1, 4, 0, 1, 3, 0};
  const Value values[] = {2.0, 3.0, 3.0, 4.0, 1.0, 5.0, 2.0};
  const int dM[5][5] = {
    {0.0, 2.0, 0.0, 0.0, 0.0},
    {0.0, 3.0, 0.0, 0.0, 3.0},
    {4.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 5.0, 0.0},
    {2.0, 0.0, 0.0, 0.0, 0.0},
  };
  COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  CSR A = coo.toCSR();
  coo.dispose();
  BCSR bA(A, 2, 2);
  bA.output("block A");
  A.output("A");
  bool flag = bA.isEqual(A);
  A.values[6] = 5.0;
  bool flag2 = bA.isEqual(A);
  A.dispose();
  assert(flag == true);
  printf("%s Passed same check\n", __func__);
  assert(flag2 == false);
  printf("%s Passed modified one value check\n", __func__);
  const int browIndex[] = {0, 1, 2, 3, 3, 4};
  const int bcolIndex[] = {1, 1, 0, 1, 3, 0};
  const Value bvalues[] = {2.0, 3.0, 4.0, 1.0, 5.0, 2.0};
  const int bnnz = 6;
  COO bcoo(bvalues, bcolIndex, browIndex, rows, cols, bnnz);
  CSR B = bcoo.toCSR();
  bcoo.dispose();
  bool flag3 = bA.isEqual(B);
  assert(flag3 == false);
  bA.dispose();
  B.dispose();
  printf("%s Passed\n", __func__);
}

int main() {
  testBCSRisEqual();
  testBCSRisEqualBorderCase();
  return 0;
}
