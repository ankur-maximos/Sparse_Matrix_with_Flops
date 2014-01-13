#ifndef QRMCL_H_
#define QRMCL_H_
#include "CSR.h"
#include "COO.h"

enum RunOptions {SEQ, OMP, GPU, CILK, SOMP, MKL, SFOMP};
static char *runOptionsStr[] = {"SEQ", "OMP", "GPU", "CILK", "SOMP", "MKL", "SFOMP"};

CSR rmclInit(COO &cooAt);
CSR RMCL(const char iname[], int maxIters, RunOptions runOptions);
#endif
