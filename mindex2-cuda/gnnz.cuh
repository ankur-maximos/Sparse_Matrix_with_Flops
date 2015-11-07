#include "tryOutCompute.cu"

const int MAX_NBLOCKS = 2048;
const int GBs = 5.6;
const int MAX_SHARED_IC = 512;

int inline qmin(const int a, const int b) {
  if (a < b) return a;
  return b;
}

int inline qmin3(const int a, const int b, const int c) {
  return qmin(qmin(a, b), c);
}

#include "casHash.cuh"


void gpu_compute_stream(const CSR &dA, const CSR &dB, int *drowIds, const vector<int> &hv, int *rowStream, int *colStream, QValue *valueStream, int *dflops) {
  int m = dA.rows;
  int n = dB.cols;
  //printf("\n1 flop count: %d", (hv[3] - hv[2]));
  if (hv.size() > 2 + 1 && hv[3] - hv[2] > 0) { // up to fp1 - bin1
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 1;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[3] - hv[2] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_stream_f1<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[2]-1, dflops + hv[2] - 1,hv[3] - hv[2], m, n, rowStream, colStream, valueStream);
    HANDLE_ERROR(cudaGetLastError());
  }

  //printf("\n2-4 flop count: %d", (hv[4] - hv[3]));
  if (hv.size() > 3 + 1 && hv[4] - hv[3] > 0) { // up to fp[2-4] - bin2
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 2;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[4] - hv[3] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_stream_f4<NTHREADS,WARP_SIZE,WARPS_PER_BLOCK><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[3]-1, dflops + hv[3] - 1, hv[4] - hv[3], m, n, rowStream, colStream, valueStream);
    HANDLE_ERROR(cudaGetLastError());
  }

  if (hv.size() > 4 + 1 && hv[5] - hv[4] > 0) { // up to fp[5-16] - bin3
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 4;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[5] - hv[4] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_stream_f4<NTHREADS,WARP_SIZE,WARPS_PER_BLOCK><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[4] - 1, dflops + hv[4] - 1, hv[5] - hv[4], m, n, rowStream, colStream, valueStream);
    HANDLE_ERROR(cudaGetLastError());
  }

  if (hv.size() > 5 + 1 && hv[6] - hv[5] > 0) { // up to fp[17-64] - bin4
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 16;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[6] - hv[5] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_stream_f4<NTHREADS,WARP_SIZE,WARPS_PER_BLOCK><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[5] - 1, dflops + hv[5] - 1, hv[6] - hv[5], m, n, rowStream, colStream, valueStream);
    HANDLE_ERROR(cudaGetLastError());
  }

  if (hv.size() > 6 + 1 && hv[7] - hv[6] > 0) { // up to fp[65-512] - bin5
    const unsigned NTHREADS = 128;const unsigned WARP_SIZE = 64;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE; 
    const unsigned NBLOCKS = qmin(65535, (hv[7] - hv[6] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_stream_f64<NTHREADS,WARP_SIZE,WARPS_PER_BLOCK><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[6] - 1, dflops + hv[6] - 1, hv[7] - hv[6], m, n, rowStream, colStream, valueStream);
    HANDLE_ERROR(cudaGetLastError());
  }

  if (hv.size() > 7 + 1 && hv[8] - hv[7] > 0) { // up to fp[512-] - bin6
    const unsigned NTHREADS = 128; 
    const unsigned NBLOCKS = qmin(65535,(hv[8] - hv[7]));
    sgpu_stream_higher<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[7] - 1, dflops + hv[7] - 1, hv[8] - hv[7], m, n, rowStream, colStream, valueStream);
    HANDLE_ERROR(cudaGetLastError());
  }
}
