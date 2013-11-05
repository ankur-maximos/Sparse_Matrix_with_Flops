#ifndef QRMCL_H_
#define QRMCL_H_
#include "CSR.h"
#include "COO.h"

enum RunOptions {SEQ, OMP, GPU};

CSR rmclInit(COO &cooAt);
CSR RMCL(const char iname[], int maxIters, RunOptions runOptions);
#endif
