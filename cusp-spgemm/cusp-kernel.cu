#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/multiply.h>
#include "gpus/timer.h"
#include "COO.h"

cusp::coo_matrix<int,float,cusp::host_memory>
coo2cusp_hcoo(const COO &coo) {
  cusp::coo_matrix<int,float,cusp::host_memory> A(coo.rows, coo.cols, coo.nnz);
  for (int i = 0; i < coo.nnz; ++i) {
    A.row_indices[i] = coo.cooRowIndex[i];
    A.column_indices[i] = coo.cooColIndex[i];
    A.values[i] = coo.cooVal[i];
  }
  return A;
}

void test(const COO &cooA, const COO &cooB) {
  //cusp::coo_matrix<int,float,cusp::host_memory> A(4,3,6);

  // initialize matrix entries on host
  /*A.row_indices[0] = 0; A.column_indices[0] = 0; A.values[0] = 10;
  A.row_indices[1] = 0; A.column_indices[1] = 2; A.values[1] = 20;
  A.row_indices[2] = 2; A.column_indices[2] = 2; A.values[2] = 30;
  A.row_indices[3] = 3; A.column_indices[3] = 0; A.values[3] = 40;
  A.row_indices[4] = 3; A.column_indices[4] = 1; A.values[4] = 50;
  A.row_indices[5] = 3; A.column_indices[5] = 2; A.values[5] = 60;

  // A now represents the following matrix
  //    [10  0 20]
  //    [ 0  0  0]
  //    [ 0  0 30]
  //    [40 50 60]*/

  cusp::coo_matrix<int, float, cusp::host_memory> A = coo2cusp_hcoo(cooA);
  cusp::coo_matrix<int, float, cusp::host_memory> B = coo2cusp_hcoo(cooB);
  // copy to the device
  typedef cusp::coo_matrix<int,float,cusp::device_memory> DCOO;
  DCOO dA(A);
  DCOO dB(B);
  //cusp::coo_matrix<int,float,cusp::device_memory> dB(A);
  //cusp::coo_matrix<int,float,cusp::device_memory> dA(A);
  // print the constructed coo_matrix
  //cusp::print(A);
  DCOO dC;
  timer t;
  cusp::multiply(dA, dB, dC);
  printf("cusp time pass %f mills\n", t.milliseconds_elapsed());
}
