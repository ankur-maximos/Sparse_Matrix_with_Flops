#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "key_value_qsort.h"

int main() {
  int keys[] = {2, 5, 9, 10, 1, 8};
  double values[] = {3, 6, 9, 10, 11, 8};
  key_value_qsort<int, double>(keys, values, 6);
  assert(keys[0] == 1); assert(fabs(values[0] - 11.0) <= 1e-7);
  assert(keys[1] == 2); assert(fabs(values[1] - 3.0) <= 1e-7);
  assert(keys[2] == 5); assert(fabs(values[2] - 6.0) <= 1e-7);
  assert(keys[3] == 8); assert(fabs(values[3] - 8.0) <= 1e-7);
  assert(keys[4] == 9); assert(fabs(values[4] - 9.0) <= 1e-7);
  assert(keys[5] == 10); assert(fabs(values[5] - 10.0) <= 1e-7);
  printf("passed key_value_qsort\n");
  return 0;
}
