//#ifdef enable_gpu
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
#include <assert.h>
//#endif

int inline qmin(const int a, const int b) {
  if (a < b) return a;
  return b;
}

__device__ inline int dqueueID(long x) {
  if (x == 0) return 0;
  else if (x == 1) return 1;
  else if (x > 1024) return 12;
  int ret = 2;
  int up = 2;
  for (up = 2; ; up *= 2, ++ret) {
    if (x <= up) return ret;
  }
}

template<int BLOCK_THREADS>
__global__ void printFlops(const int IA[],const int JA[], const int IB[], const int drowids[], const int gcount, unsigned int row_count_array[]) {
	for(int k = blockIdx.x;k<gcount;k+=gridDim.x) {
		int rowId = drowids[k];
		int endRow = IA[rowId + 1];
		for(int i = threadIdx.x+IA[rowId];i<endRow;i+=blockDim.x) {
			int a = JA[i];
			long flops = IB[a+1] - IB[a];
			int index = dqueueID(flops);
			atomicAdd(&row_count_array[index],1);			
		}
	}     	
}

void count_row_flops(const CSR &dA, const CSR &dB, int *drowIds, const vector<int> &hv) {
	unsigned int *drow_count_array;
	unsigned int hrow_count_array[13];
	HANDLE_ERROR(cudaMalloc((void**)&drow_count_array, 13 * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	
	int blocks = qmin(65535, hv[2] - hv[1]);
	int threads = 256;
	printf("Binwise distribution of per element for bin 1 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[1], hv[2] - hv[1], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);

	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	 blocks = qmin(65535, hv[3] - hv[2]);
	 threads = 256;
	printf("Binwise distribution of per element for bin 2 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[2], hv[3] - hv[2], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);

	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	 blocks = qmin(65535, hv[4] - hv[3]);
	 threads = 256;
	printf("Binwise distribution of per element for bin 3 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[3], hv[4] - hv[3], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);

	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	blocks = qmin(65535, hv[5] - hv[4]);
	threads = 256;
	printf("Binwise distribution of per element for bin 4 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[4], hv[5] - hv[4], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);

	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	blocks = qmin(65535, hv[6] - hv[5]);
	threads = 256;
	printf("Binwise distribution of per element for bin 5 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[5], hv[6] - hv[5], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);

	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	blocks = qmin(65535, hv[7] - hv[6]);
	threads = 256;
	printf("Binwise distribution of per element for bin 6 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[6], hv[7] - hv[6], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);

	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	blocks = qmin(65535, hv[8] - hv[7]);
	threads = 256;
	printf("Binwise distribution of per element for bin 7 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[7], hv[8] - hv[7], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);

	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	blocks = qmin(65535, hv[9] - hv[8]);
	threads = 256;
	printf("Binwise distribution of per element for bin 8 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[8], hv[9] - hv[8], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);

	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	blocks = qmin(65535, hv[10] - hv[9]);
	threads = 256;
	printf("Binwise distribution of per element for bin 9 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[9], hv[10] - hv[9], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);

	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	blocks = qmin(65535, hv[11] - hv[10]);
	threads = 256;
	printf("Binwise distribution of per element for bin 10 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[10], hv[11] - hv[10], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);	
	
	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	blocks = qmin(65535, hv[12] - hv[11]);
	threads = 256;
	printf("Binwise distribution of per element for bin 11 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[11], hv[12] - hv[11], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);
	
	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	blocks = qmin(65535, hv[64] - hv[63]);
	threads = 256;
	printf("Binwise distribution of per element for bin 12 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[63], hv[64] - hv[63], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);

	HANDLE_ERROR(cudaMemset(drow_count_array,0,13 * sizeof(unsigned int)));
	blocks = qmin(65535, hv[1] - hv[0]);
	threads = 256;
	printf("Binwise distribution of per element for bin 0 \n");
	printFlops<256><<<blocks,threads>>>(dA.rowPtr,dA.colInd,dB.rowPtr,drowIds + hv[0], hv[1] - hv[0], drow_count_array);	
	HANDLE_ERROR(cudaMemcpy((void*) hrow_count_array, (void*) drow_count_array, 13 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaGetLastError());
	for(int i=1;i<13;i++)
	printf("count %d : %lu \n" , i, hrow_count_array[i]);
	HANDLE_ERROR(cudaFree(drow_count_array));
} 
