#include "CSR.h"
#include "COO.h"
#include "iostream"
#include "ntimer.h"
#include "omp.h"
#include "util.h"

void rmclIter(const int maxIter, const int gamma, const CSR Mgt, CSR &Mt) {
  for (int iter = 0; iter < maxIter; ++iter) {
    CSR newMt = Mgt.spmm(Mt);
    int pos = 0;
    int i;
    for (i = 0; i < newMt.rows; ++i) {
      int count = newMt.rowCount(i);
      double* values = (double*)malloc(count * sizeof(double));
      arrayInflationR2(newMt.values + newMt.rowPtr[i], count, values);
      double rmax = arrayMax(values, count);
      double rsum = arraySum(values, count);
      double thresh = computeThreshold(rsum / count, rmax);
      int* indices = (int*)malloc(count * sizeof(int));
      double nsum = arrayThreshPrune(thresh, &count, indices, values);
      //normalize values vector
      for (int k = 0; k < count; ++k) {
        values[k] /= nsum;
      }
      memcpy(newMt.values + pos, values, count * sizeof(double));
      memcpy(newMt.colInd + pos, indices, count * sizeof(int));
      newMt.rowPtr[i] = pos;
      pos += count;
      free(values);
      free(indices);
    }
    newMt.rowPtr[i + 1] = pos;
  }
}

CSR rmclInit(COO &cooAt) {
  cooAt.addSelfLoopIfNeeded();
  //cooA.output("COO");
  cooAt.makeOrdered();
  //cooA.output("COO");
  CSR At = cooAt.toCSR();
  At.averAndNormRowValue();
  return At;
}

void RMCL(const char iname[]) {
  COO cooAt;
  cooAt.readTransposedSNAPFile(iname);
  CSR Mt = rmclInit(cooAt);
  cooAt.~COO();
  CSR Mgt = Mt.deepCopy();
  rmclIter(1000, 2, Mgt, Mt);
}

int main(int argc, char *argv[]) {
  char iname[500] = "email-Enron.txt";
  if (argc == 2) {
    strcpy(iname, argv[1]);
  }
  return 0;
}
