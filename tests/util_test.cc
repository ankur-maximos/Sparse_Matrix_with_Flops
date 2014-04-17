#include "tools/util.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

void arrayMaxTest() {
  double values[] = {2.0, 5.0, 4.0, 3.0};
  double mmax = arrayMax(values, sizeof(values)/sizeof(double));
  assert(fabs(mmax - 5.0) <= 1.0e-7);
  printf("%s Passed\n", __func__);
}

void arraySumTest() {
  double values[] = {2.0, 5.0, 4.0, 3.0, -2.0};
  double rsum = arraySum(values, sizeof(values)/sizeof(double));
  assert(fabs(rsum - 12.0) <= 1.0e-7);
  printf("%s Passed\n", __func__);
}

void arrayInflationR2Test() {
  double values[] = {4.0, 3.0, -2.0, 0.0};
  double ovalues[10];
  arrayInflationR2(values, sizeof(values)/sizeof(double), ovalues);
  assert(fabs(ovalues[0] - 16.0) <= 1.0e-7);
  assert(fabs(ovalues[1] - 9.0) <= 1.0e-7);
  assert(fabs(ovalues[2] - 4.0) <= 1.0e-7);
  assert(fabs(ovalues[3] - 0.0) <= 1.0e-7);
  printf("%s Passed\n", __func__);
}

void randomPermutationVectorTest() {
  int* P = randomPermutationVector(5);
  arrayOutput("P5=", stdout, P, 5);
}

void permutationTransposeTest() {
  int P[] = {1, 5, 4, 2, 3, 0};
  int *rPt = permutationTranspose(P, 6);
  int Pt[] = {5,0, 3, 4, 2, 1};
  for (int i = 0; i < 6; ++i) {
    assert(rPt[i] == Pt[i]);
  }
  printf("%s Passed\n", __func__);
}

int main() {
  arrayMaxTest();
  arraySumTest();
  arrayInflationR2Test();
  randomPermutationVectorTest();
  permutationTransposeTest();
  printf("Passed\n");
  return 0;
}
