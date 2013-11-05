#ifndef GPU_CSR_KERNEL_H
#define GPU_CSR_KERNEL_H
#include "CSR.h"

void gpuRmclIter(const int maxIter, const CSR Mgt, CSR &Mt);
#endif
