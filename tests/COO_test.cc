#include "COO.h"

COO readTransposedSNAPFile() {
  COO coo;
  //coo.readTransposedSNAPFile("tdatas/tdata.snap");
  //coo.output("tdata");
  return coo;
}

void addSelfLoopIfNeededTest() {
  int rows = 4, cols = 4, nnz = 5;
  const int rowIndex[] = {0, 1, 2, 3, 3};
  const int colIndex[] = {1, 2, 0, 1, 3};
  const double values[] = {2.0, 3.0, 4.0, 1.0, 5.0};
  //COO coo(values, colIndex, rowIndex, rows, cols, nnz);
  //coo.addSelfLoopIfNeeded();
  //coo.output("tdata");
}

int main() {
  //readTransposedSNAPFileTest();
  //addSelfLoopIfNeededTest();
  return 0;
}
