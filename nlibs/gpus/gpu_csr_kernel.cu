//using namespace cub;
//CachingDeviceAllocator  g_allocator;
#include "gpus/gpu_csr_kernel.h"
#include <stdio.h>
#include "tools/util.h"
#include <cub/cub.cuh>
#include "gpus/timer.h"
#include "gpus/cusparse_spmm.h"
#include "tools/ntimer.h"
#include <cuda.h>
#include <cuda_runtime.h>

//__device__ int gpu_cRowiCount

__global__ void gpu_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n,
    int* IC,
    bool *xbs, int *iJCs) {
  bool *xb = xbs + blockIdx.x * n;
  int *iJC = iJCs + blockIdx.x * n;
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  __syncthreads();
  for (int i = blockIdx.x; i < m; i += gridDim.x) {
    IC[i] = 0;
    for (int vp = IA[i]; vp < IA[i + 1]; ++vp) {
      int v = JA[vp];
      for (int kp = IB[v] + threadIdx.x; kp < IB[v + 1]; kp += blockDim.x) {
        int k = JB[kp];
        if (xb[k] == false) {
          iJC[atomicAdd(&count, 1)] = k;
          xb[k] = true;
        }
      }
      __syncthreads();
    }
    for (int jp = threadIdx.x; jp < count; jp += blockDim.x) {
      int j = iJC[jp];
      xb[j] = false;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      IC[i] = count;
      count = 0;
    }
  }
}

void gpuRmclIter(const int maxIter, const CSR Mgt, CSR &Mt) {
  const int NBLOCKS = 512;
  int m = Mgt.rows;
  int n = Mt.cols;
  int *IC=(int*)calloc(m+1, sizeof(int));
  bool* xb=(bool*)calloc(n, sizeof(bool));
  int nnzC = 0;
  int nthreads = 8;
  thread_data_t* thread_datas = allocateThreadDatas(nthreads, Mt.cols);
  double now = time_in_mill_now();
  omp_CSR_IC_nnzC_Wrapper(Mgt.rowPtr, Mgt.colInd, Mt.rowPtr, Mt.colInd, Mgt.rows, Mt.cols, thread_datas,
            IC, nnzC);
  printf("Time passed for omp %lf\n", time_in_mill_now() - now);
  freeThreadDatas(thread_datas, nthreads);
  now = time_in_mill_now();
  sequential_CSR_IC_nnzC(Mgt.rowPtr, Mgt.colInd, Mt.rowPtr, Mt.colInd, Mgt.rows, Mt.cols, xb,
            IC, nnzC);
  printf("Time passed for sequential %lf\n", time_in_mill_now() - now);
  //arrayOutput("cpu ic", stdout, IC, m + 1);
  int *counts = (int*)calloc(m + 1, sizeof(int));
  for (int i = 0; i < m; ++i) {
    counts[i] = IC[i + 1] - IC[i];
  }
    //arrayOutput("counts", stdout, counts, m);
  CSR dMgt = Mgt.toGpuCSR();
  CSR dMt = Mt.toGpuCSR();
  CSR dNewMt;
  bool *xbs = NULL;
  int *iJCs = NULL;
  now = time_in_mill_now();
  timer t3;
  cudaMalloc((void**)&xbs, NBLOCKS * n * sizeof(bool)); cudaMemset(xbs, 0, NBLOCKS * n * sizeof(bool));
  cudaMalloc((void**)&iJCs, NBLOCKS * n * sizeof(int));
  cudaMalloc((void**)&dNewMt.rowPtr, NBLOCKS * (m + 1) * sizeof(int));
  //for (int iter = 0; iter < maxIter; ++iter) {
  timer t;
  gpu_CSR_IC_nnzC<<<NBLOCKS, 128>>>(dMgt.rowPtr, dMgt.colInd, dMt.rowPtr, dMt.colInd,
        m, n, dNewMt.rowPtr, xbs, iJCs);
  cudaDeviceSynchronize();
  double timeUsed = t.milliseconds_elapsed();
  printf("Time used in gpu = %lf\n", timeUsed);
  printf("Total Time passed for gpu with cudaMalloc %lf\n", t3.milliseconds_elapsed());
    //int *IC = (int*)malloc((m + 1) * sizeof(int));
    cudaMemcpy(IC, dNewMt.rowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Begin output array\n");
    int flag = memcmp(IC, counts, m * sizeof(int));
    if (flag == 0) {
      printf("Same\n");
    } else {
      printf("Differs\n");
    }
    //int m = dMgt.rows;
    int k = dMgt.cols;
    //int n = Mt.cols;
    int* dICs = NULL;
    cusparse_init();
    timer t2;
    cusparseXcsrgemmNnzWrapper(dMgt.rowPtr, dMgt.colInd, Mgt.nnz,
        dMt.rowPtr, dMt.colInd, Mt.nnz,
        m, k, n,
        dICs, nnzC);
  double timeUsed2 = t2.milliseconds_elapsed();
  cudaMemcpy(IC, dICs, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  //arrayOutput("cusparse ICs", stdout, IC, m);
  //for (int i = 0; i < m; ++i) {
    //IC[i] = IC[i + 1] - IC[i];
  //}
  printf("Time used in cusparse gpu = %lf\n", timeUsed2);
    flag = memcmp(IC, counts, m * sizeof(int));
    if (flag == 0) {
      printf("cusparse Same\n");
    } else {
      printf("cusparse Differs\n");
    }
    cudaFree(xbs);
    cudaFree(iJCs);
    cudaFree(dNewMt.rowPtr);
    //cusparse_finalize("clear up cusparse");
    //arrayOutput("gpu ic", stdout, IC, m + 1);
  //}
}

void gpu_CSR_RMCL_OneStep(const int IA[], const int JA[], const double A[], const int nnzA,
        const int IB[], const int JB[], const double B[], const int nnzB,
        int* &IC, int* &JC, double* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas) {
    //JC = (int*)malloc(sizeof(int) * nnzC);
    //C = (double*)malloc(sizeof(double) * nnzC);
    //int *dIC = NULL, *dJC = NULL;
    //double *dC = NULL;
    //cudaMalloc((void**)&dIC, (m + 1) * sizeof(int));
    //cudaMemcpy(dIC, IC, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMalloc((void**)&dIC, (m + 1) * sizeof(int));
}
