#include "tools/util.h"
#include <stdio.h>

#include <cub/cub.cuh>
#include "gpus/dreduce.cuh"

__device__ double dComputeThreshold(double avg, double max) {
	double ret = MLMCL_PRUNE_A * avg * (1 - MLMCL_PRUNE_B * (max - avg));
	ret = (ret > 1.0e-7) ? ret : 1.0e-7;
	ret = (ret > max) ? max : ret;
	return ret;
}

template <int BLOCK_THREADS>
__device__ int threshPruneNormalize(const int count, const double thresh,
    int iJC[], double iC[]) {
    double psum = 0.0;
    int nonzeros = 0;
    for (int jp = threadIdx.x; jp < count; jp += blockDim.x) {
      double Cjp = iC[jp];
      bool flag = (Cjp >= thresh);
      iC[jp] = (flag ? Cjp : -1.0);
      iJC[jp] = (flag ? iJC[jp] : -1);
      psum += (flag ? Cjp : 0.0);
      nonzeros += (flag ? 1 : 0);
    }
    __syncthreads();
    typedef cub::BlockReduce<double, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double rowPruneSum = BlockReduce(temp_storage).Sum(psum);
    __shared__ double srowPruneSum;
    if (threadIdx.x == 0) {
      srowPruneSum = rowPruneSum;
    }
    __syncthreads();
    typedef cub::BlockReduce<int, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING> IntBlockReduce;
    __shared__ typename IntBlockReduce::TempStorage int_temp_storage;
    int rowPruneCount = IntBlockReduce(int_temp_storage).Sum(nonzeros);
    for (int jp = threadIdx.x; jp < count; jp += blockDim.x) {
      double Cjp = iC[jp];
      int flag = (Cjp > 0.0);
      iC[jp] = (flag ? Cjp / srowPruneSum  : -1.0);
    }
    __shared__ double srowPruneCount;
    if (threadIdx.x == 0) {
      srowPruneCount = rowPruneCount;
    }
    __syncthreads();
    return srowPruneCount;
}

template <int BLOCK_THREADS>
__device__ double inflationR2Thresh(const int count,
    double iC[]) {
    double xmax = -9999.0;
    //double xsum = 0.0;
    for (int jp = threadIdx.x; jp < count; jp += blockDim.x) {
      double xj = iC[jp];
      double xx = xj * xj;
      iC[jp] = xx;
      xmax = (xx > xmax ? xx : xmax);
      //xsum += xx;
    }
    __syncthreads();
    typedef cub::BlockReduce<double, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ double srowSum;
    reduce2<double, BLOCK_THREADS>(iC, &srowSum, count);
    __syncthreads();
    //double rowSum = BlockReduce(temp_storage).Sum(xsum);
    double rowMax = BlockReduce(temp_storage).Reduce(xmax, cub::Max());
    double thresh = dComputeThreshold(srowSum / count, rowMax);
    __shared__ double sthresh;
    if (threadIdx.x == 0) {
      sthresh = thresh;
    }
    __syncthreads();
    return sthresh;
}
