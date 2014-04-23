#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "tools/key_value_qsort.h"

void testDefault() {
  int keys[] = {2, 5, 9, 10, 1, 8};
  double values[] = {3, 6, 9, 10, 11, 8};
  key_value_qsort<int, double>(keys, values, 6);
  assert(keys[0] == 1); assert(fabs(values[0] - 11.0) <= 1e-7);
  assert(keys[1] == 2); assert(fabs(values[1] - 3.0) <= 1e-7);
  assert(keys[2] == 5); assert(fabs(values[2] - 6.0) <= 1e-7);
  assert(keys[3] == 8); assert(fabs(values[3] - 8.0) <= 1e-7);
  assert(keys[4] == 9); assert(fabs(values[4] - 9.0) <= 1e-7);
  assert(keys[5] == 10); assert(fabs(values[5] - 10.0) <= 1e-7);
  printf("passed key_value_qsort default compare function test\n");
}

template <typename sKey>
bool greaterThanFunction(const sKey &a, const sKey &b) {
  return a > b;
}

void testGreaterSort() {
  int keys[] = {2, 5, 9, 10, 1, 8};
  double values[] = {3, 6, 9, 10, 11, 8};
  key_value_qsort<int, double>(keys, values, 6, &(greaterThanFunction<int>));
  assert(keys[0] == 10); assert(fabs(values[0] - 10.0) <= 1e-7);
  assert(keys[1] == 9); assert(fabs(values[1] - 9.0) <= 1e-7);
  assert(keys[2] == 8); assert(fabs(values[2] - 8.0) <= 1e-7);
  assert(keys[3] == 5); assert(fabs(values[3] - 6.0) <= 1e-7);
  assert(keys[4] == 2); assert(fabs(values[4] - 3.0) <= 1e-7);
  assert(keys[5] == 1); assert(fabs(values[5] - 11.0) <= 1e-7);
  printf("passed key_value_qsort greater compare function test\n");
}

int main() {
  testDefault();
  testGreaterSort();
  return 0;
}
