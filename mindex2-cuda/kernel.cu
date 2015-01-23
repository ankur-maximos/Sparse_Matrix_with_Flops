#include "gpus/cuda_handle_error.h"
//#include "gpus/cusparse_spmm.h"
#include "gpus/gpu_csr_kernel.h"
#include "gpus/timer.h"
#include "tools/ntimer.h"
#include "sort_network.cuh"
//#include "large.cuh"
#include "radix_sort.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include "gnnz.cuh"
#include "gspgemm.cuh"
#include <assert.h>

template <int BLOCK_THREADS>
__global__ void sgpu_SpGEMM_olarge(
    const int IA[], const int JA[], const QValue A[],
    const int IB[], const int JB[], const QValue B[],
    const int drowIds[], const int gcount,
    const int m, const int n,
    int IC[], int JC[], QValue C[],
    int *xbs) {
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  int *xb = xbs + blockIdx.x * n;
  __syncthreads();
  for (int q = blockIdx.x; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    //if (threadIdx.x == 0 && rowId == 3) printf("row%d is in olarge\n", rowId);
    const int ICi = IC[rowId];
    int *iJC = JC + ICi;
    QValue *iC = C + ICi;
    for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
      const int a = JA[ap];
      const QValue Aap = A[ap];
      for (int bp = IB[a] + threadIdx.x; bp < IB[a + 1]; bp += blockDim.x) {
        int b = JB[bp];
        if (xb[b] == -1) {
          int pos = atomicAdd(&count, 1);
          iJC[pos] = b;
          iC[pos] = Aap * B[bp];
          xb[b] = pos;
        } else {
          iC[xb[b]] += Aap * B[bp];
        }
        //if (rowId == 3 && b == 247) {
          //printf("A[%d][%d]=%f ap=%d\n", rowId, a, Aap, ap);
          //printf("find C[%d][%d]=%f a=%d Aap=%f B[bp]=%f\n", rowId, b, iC[xb[b]], a, Aap, B[bp]);
          //printf("C[%d][%d]+=A[%d][%d]*B[%d][%d]=%f*%f=%f\n", rowId, b, rowId, a, a, b, Aap, B[bp], iC[xb[b]]);
        //}
      }
      __syncthreads();
    }
    for (int cp = threadIdx.x; cp < count; cp += blockDim.x) {
      int c = iJC[cp];
      xb[c] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      count = 0;
    }
  }
}

__device__ inline void xBlockBrowsProcess(const int  start, const int end, const int aps[],
    const int JA[], const float A[],
    const int IB[], const int JB[], const float B[],
    int *xb, float *x,
    int *countp, int *iJC) {
  for (int at = start; at < end; ++at) {
    const int ap = aps[at];
    const int a = JA[ap];
    const float Aap = A[ap];
    for (int bp = IB[a] + threadIdx.x; bp < IB[a + 1]; bp += blockDim.x) {
      const int b = JB[bp];
      const float cVal = Aap * B[bp];
      if (xb[b] == -1) {
        int pos = atomicAdd(countp, 1);
        iJC[pos] = b;
        x[b] = cVal;
        xb[b] = -2;
      } else {
        x[b] += cVal;
      }
    }
    __syncthreads();
  }
}

template <int WARP_SIZE>
__device__ inline void concurrentXBrowsProcess(const int toffset, const int  start, const int end, const int aps[],
    const int JA[], const float A[],
    const int IB[], const int JB[], const float B[],
    int *xb, float *x,
    int *countp, int *iJC) {
  const int WARPS_PER_BlOCK = blockDim.x / WARP_SIZE;
  int woffset = (toffset % blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  const int warpId = (threadIdx.x / WARP_SIZE + WARPS_PER_BlOCK - woffset) % WARPS_PER_BlOCK;
  const int laneId = threadIdx.x % WARP_SIZE;
  for (int at = start + warpId; at < end; at += WARPS_PER_BlOCK) {
    const int ap = aps[at];
    const int a = JA[ap];
    const float Aap = A[ap];
    for (int bp = IB[a] + laneId; bp < IB[a + 1]; bp += WARP_SIZE) {
      const int b = JB[bp];
      float cVal = Aap * B[bp];
      atomicAdd(x + b, cVal);
      if (atomicCAS(xb + b, -1, -2) == -1) {
        const int pos = atomicAdd(countp, 1);
        iJC[pos] = b;
      }
    }
    //__syncthreads();
  }
  __syncthreads();
}

template <int BLOCK_THREADS>
__global__ void sgpu_SpGEMM_tlarge(
    const int IA[], const int JA[], const float A[],
    const int IB[], const int JB[], const float B[],
    const int drowIds[], const int gcount,
    const int m, const int n,
    int IC[], int JC[], float C[],
    int *xbs, float *xs) {
  __shared__ unsigned keys[BLOCK_THREADS];
  __shared__ int aps[BLOCK_THREADS];
  int *xb = xbs + blockIdx.x * n;
  float *x = xs + blockIdx.x * n;
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  __syncthreads();
  for (int q = blockIdx.x; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    const int ICi = IC[rowId];
    int *iJC = JC + ICi;
    float *iC = C + ICi;
    for (int ap = IA[rowId] + threadIdx.x; __syncthreads_or(ap < IA[rowId + 1]); ap += blockDim.x) {
      int predicate = (ap < IA[rowId + 1]);
      int a = predicate ? JA[ap] : -1;
      keys[threadIdx.x] = predicate ? (IB[a + 1] - IB[a]) : -1;
      aps[threadIdx.x] = ap;
      unsigned le4 = partition_by_bound(keys, aps, 4);
      unsigned le32 = partition_by_bound(keys, aps, 32);
      unsigned le64 = partition_by_bound(keys, aps, 64);
      unsigned total = min(IA[rowId + 1] + threadIdx.x - ap, blockDim.x);
      concurrentXBrowsProcess<1>(0, 0, le4, aps, JA, A, IB, JB, B, xb, x, &count, iJC);
      //int toffset = (le4 + 15) / 16 * 16;
      concurrentXBrowsProcess<16>(0, le4, le32, aps, JA, A, IB, JB, B, xb, x, &count, iJC);
      //toffset = toffset + ((le32 - le4) * 16 + 31) / 32 * 32;
      concurrentXBrowsProcess<32>(0, le32, le64, aps, JA, A, IB, JB, B, xb, x, &count, iJC);
      __syncthreads();
      xBlockBrowsProcess(le64, total, aps, JA, A, IB, JB, B, xb, x, &count, iJC);
    }
    for (int cp = threadIdx.x; cp < count; cp += blockDim.x) {
      int c = iJC[cp];
      xb[c] = -1;
      iC[cp] = x[c];
      x[c] = 0.0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      count = 0;
    }
  }
}


void sgpu_SpGEMM(const CSR &dA, const CSR &dB, int *drowIds, const vector<int> &hv, CSR &dC) {
  int m = dA.rows;
  int n = dB.cols;
  if (hv.size() > 0 + 1 && hv[1] - hv[0] > 0) { // up to fp0
    sgpu_SpGEMM_a1<512><<<128, 512>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[0], hv[1] - hv[0], m, n, dC.rowPtr, dC.colInd, dC.values);
  }
  if (hv.size() > 1 + 1 && hv[2] - hv[1] > 0) { // up to fp1
    const unsigned NTHREADS = 512; const unsigned WARP_SIZE = 1;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[2] - hv[1] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_SpGEMM_fp1<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[1], hv[2] - hv[1], m, n, dC.rowPtr, dC.colInd, dC.values);
  }
  if (hv.size() > 2 + 1 && hv[3] - hv[2] > 0) { // up to fp2
    const unsigned NTHREADS = 512; const unsigned WARP_SIZE = 1;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[3] - hv[2] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_SpGEMM_fp2<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[2], hv[3] - hv[2], m, n, dC.rowPtr, dC.colInd, dC.values);
  }
  if (hv.size() > 3 + 1 && hv[4] - hv[3] > 0) { // up to fp4
    const unsigned NTHREADS = 256; const unsigned WARP_SIZE = 1;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[4] - hv[3] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_SpGEMM_mid<NTHREADS, WARP_SIZE, 7><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[3], hv[4] - hv[3], m, n, dC.rowPtr, dC.colInd, dC.values);
  }
  if (hv.size() > 4 + 1 && hv[5] - hv[4] > 0) { // up to fp8
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 4;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[5] - hv[4] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_SpGEMM_mid<NTHREADS, WARP_SIZE, 17><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[4], hv[5] - hv[4], m, n, dC.rowPtr, dC.colInd, dC.values);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (hv.size() > 5 + 1 && hv[6] - hv[5] > 0) { // up to fp16
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 16;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[6] - hv[5] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_SpGEMM_mid<NTHREADS, WARP_SIZE, 37><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[5], hv[6] - hv[5], m, n, dC.rowPtr, dC.colInd, dC.values);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (hv.size() > 6 + 1 && hv[7] - hv[6] > 0) { // up to fp32
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 32;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[7] - hv[6] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_SpGEMM_mid<NTHREADS, WARP_SIZE, 71><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[6], hv[7] - hv[6], m, n, dC.rowPtr, dC.colInd, dC.values);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (hv.size() > 7 + 1 && hv[8] - hv[7] > 0) { // up to fp64
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 32;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[8] - hv[7] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_SpGEMM_mid<NTHREADS, WARP_SIZE, 139><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[7], hv[8] - hv[7], m, n, dC.rowPtr, dC.colInd, dC.values);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (hv.size() > 8 + 1 && hv[9] - hv[8] > 0) { // up to fp128
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 64;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[9] - hv[8] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_SpGEMM_mid<NTHREADS, WARP_SIZE, 271><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[8], hv[9] - hv[8], m, n, dC.rowPtr, dC.colInd, dC.values);
  }
  if (hv.size() > 9 + 1 && hv[10] - hv[9] > 0) { // up to fp256
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 128;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[10] - hv[9] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_SpGEMM_mid<NTHREADS, WARP_SIZE, 571><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[9], hv[10] - hv[9], m, n, dC.rowPtr, dC.colInd, dC.values);
    HANDLE_ERROR(cudaGetLastError());
  }
  /*if (hv.size() > 10 + 1 && hv[11] - hv[10] > 0) { // up to fp512*/
  /*  const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 128;*/
  /*  const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;*/
  /*  const unsigned NBLOCKS = qmin(65535, (hv[11] - hv[10] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);*/
  /*  sgpu_SpGEMM_mid<NTHREADS, WARP_SIZE, 1187><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[10], hv[11] - hv[10], m, n, dC.rowPtr, dC.colInd, dC.values);*/
  /*  HANDLE_ERROR(cudaGetLastError());*/
  /*}*/

  assert (hv.size() == 65);
  //very large
  if (hv.size() > 10 + 1 && hv[63] - hv[10] > 0) // larger than fp256
  //if (hv.size() > 10 + 1 && hv.back() - hv[10] > 0) // larger than fp256
  //if (hv.size() > 11 + 1 && hv.back() - hv[11] > 0) // larger than fp512
  {
    const int NBLOCKS = 512; const int NTHREADS = 128;
    int *xbs = NULL;
    HANDLE_ERROR(cudaMalloc((void**)&xbs, NBLOCKS * n * sizeof(int)));
    HANDLE_ERROR(cudaMemset(xbs, -1, NBLOCKS * n * sizeof(int)));
    float *xs = NULL; HANDLE_ERROR(cudaMalloc((void**)&xs, NBLOCKS * n * sizeof(float)));
    HANDLE_ERROR(cudaMemset(xs, 0, NBLOCKS * n * sizeof(float)));
    sgpu_SpGEMM_tlarge<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[10], hv[63] - hv[10], m, n, dC.rowPtr, dC.colInd, dC.values, xbs, xs);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaFree(xbs));
    HANDLE_ERROR(cudaFree(xs));
  }

  if (hv[64] - hv[63] > 0) // >=128 non entries a single row of matrix A.
  {
    const int NBLOCKS = 512; const int NTHREADS = 128;
    int *xbs = NULL;
    HANDLE_ERROR(cudaMalloc((void**)&xbs, NBLOCKS * n * sizeof(int)));
    HANDLE_ERROR(cudaMemset(xbs, -1, NBLOCKS * n * sizeof(int)));
    sgpu_SpGEMM_olarge<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[63], hv[64] - hv[63], m, n, dC.rowPtr, dC.colInd, dC.values, xbs);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaFree(xbs));
  }
}

CSR sgpuSpMMWrapper(const CSR &dA, const CSR &dB, int *drowIds, const vector<int> &hv) {
  CSR dC;
  //int *dqueue = NULL; computeDv(dA, dB, &dv, &dqueue);
  gpu_compute_IC(dA, dB, drowIds, hv, dC);
  HANDLE_ERROR(cudaGetLastError());
  thrust::device_ptr<int> dIC = thrust::device_pointer_cast(dC.rowPtr);
  int m = dA.rows;
  //int n = dB.cols;
  thrust::exclusive_scan(dIC, dIC + m + 1, dIC);
  int cNnz = dIC[m];
  HANDLE_ERROR(cudaMalloc((void**)&dC.colInd, cNnz * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dC.values, cNnz * sizeof(QValue)));
  //HANDLE_ERROR(cudaMemset(dC.values, 0, cNnz * sizeof(QValue)));
  sgpu_SpGEMM(dA, dB, drowIds, hv, dC);
  cudaDeviceSynchronize();
  dC.nnz = cNnz;
  dC.rows = dA.rows;
  dC.cols = dB.cols;
  return dC;
}
