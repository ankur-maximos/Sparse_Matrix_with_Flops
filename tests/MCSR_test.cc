#include "MCSR.h"
#include "tools/util.h"
/*
bool valuePredicate(Value a, Value b) {
  return fabs(a - b) < 1e-8;
}*/

void MCSR_CSR_Constructor_test() {
/*  const Value dvalues[] = {
    1, 2, 0, 0, 0, 0, 0,
    0, 0, 3, 0, 0, 0, 4,
    0, 4, 2, 3, 0, 5, 0,
    0, 0, 0, 0, 2, 0, 0,
    3, 0, 0, 1, 0, 8, 1,
    0, 0, 3, 0, 0, 0, 0,
    0, 2, 0, 0, 1, 0, 3
  };*/
  const int rows = 7, cols = 7;
  /*CSR A;
  A.initWithDenseMatrix(dvalues, rows, cols);
  MCSR mA(A, 1, 2, 4, 4);
  const int rowPtr[] = {0, 0, 1, 2, 3, 7, 8, 11};
  const int colInd[] = {6, 5, 4, 0, 3, 5, 6, 2, 1, 4, 6};
  const Value values[] = {4, 5, 2, 3, 1, 8, 1, 3, 2, 1, 3};
  assert(std::equal(rowPtr, rowPtr + sizeof(rowPtr) / sizeof(int), mA.CSR::rowPtr) == true);
  assert(std::equal(colInd, colInd + sizeof(colInd) / sizeof(int), mA.CSR::colInd) == true);
  assert(std::equal(values, values + sizeof(values) / sizeof(Value), mA.CSR::values) == true);
  const int browPtr[] = {0, 1, 2, 4, 4};
  const int bcolInd[] = {0, 1, 0, 1};
  const Value bvalues[] = {1, 2, 3, 0, 0, 4, 2, 3};
  assert(std::equal(browPtr, browPtr + sizeof(browPtr) / sizeof(int), mA.BCSR::rowPtr) == true);
  assert(std::equal(bcolInd, bcolInd + sizeof(bcolInd) / sizeof(int), mA.BCSR::colInd) == true);
  assert(std::equal(bvalues, bvalues + sizeof(bvalues) / sizeof(Value), mA.BCSR::values, valuePredicate) == true);
  printf("%s Passed\n", __func__);*/
}

int main() {
  //MCSR_CSR_Constructor_test();
  return 0;
}
