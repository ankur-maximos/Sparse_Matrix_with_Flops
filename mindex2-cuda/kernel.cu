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
#include <vector>

#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <thrust/copy.h> 
#include <thrust/fill.h> 
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
//#include <thrust/make_zip_iterator.h>
//#include <thrust/sort_by_key.h>
//#include <thrust/make_tuple.h>
#include <thrust/tuple.h>

#include "moderngpu.cuh"

using namespace std;
using namespace mgpu;

const int FLOPS_SORT = 1024;


struct pack{
template <typename Tuple>
  __device__ __host__ int64 operator()(const Tuple &t){
    return ( static_cast<int64>( thrust::get<0>(t) ) << 32 ) | thrust::get<1>(t);
  }
/*  __device__ __host__ int32 operator()(const Tuple &t){
    return ( ( thrust::get<0>(t) ) << 16 ) | thrust::get<1>(t);
  }
*/};

struct unpack{
  __device__ __host__ thrust::tuple<int,int> operator()(int64 p){
    int d = static_cast<int>(p >> 32);
    int s = static_cast<int>(p & 0xffffffff);
    return thrust::make_tuple(d, s);
  }
/*
  __device__ __host__ thrust::tuple<int,int> operator()(int32 p){
    int d = static_cast<int>(p >> 16);
    int s = static_cast<int>(p & 0xffff);
    return thrust::make_tuple(d, s);
  }
*/};



template<int BLOCK_THREADS> 
__global__ void compute_sorting_pointers(int flops_sort_arr[], int r_c_size, int segment_size, int* rowStream) { 
     int tid = threadIdx.x + blockIdx.x * blockDim.x;	
     int flop;
     for(int i=tid; i<segment_size ; i+=blockDim.x * gridDim.x) {   
	flop = FLOPS_SORT * (i+1); 
        //printf("flop : %d", flop);
        if(flop >= r_c_size) { flops_sort_arr[i] = -1;}
	int cur_row = rowStream[flop-1];  
	int next_row = rowStream[flop]; 
        printf("cur_row %d next_row %d\n ", cur_row, next_row); 
	while(flop<r_c_size  && next_row == cur_row) {  
		flop++;  
		cur_row = next_row;  
		next_row = rowStream[flop];  
	}
        if(flop >= r_c_size) 
        flops_sort_arr[i] = -1;
 	else 
	flops_sort_arr[i] = flop;	
    } 
} 

/*
void sort_data(int size,thrust::devicevector<int> d_rows,thrust::devicevector<int> d_cols,thrust::devicevector<int> d_vals){
	thrust::device_vector<int64> tmp(size);
// Pack (day, site) pairs into 64-bit integers.
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(d_rows.begin(), d_cols.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(d_rows.end(), d_cols.end())),
		tmp.begin(),
		pack());

// Sort using the 64-bit integers as keys.
	thrust::sort_by_key(tmp.begin(), tmp.end(), d_vals.begin());

// Unpack (row,cols) pairs from 64-bit integers.
	thrust::transform(
		tmp.begin(),
		tmp.end(),
		thrust::make_zip_iterator(thrust::make_tuple(d_rows.begin(), d_cols.begin())),
		unpack());
}
*/
template<class T>
__device__ void partition_by_bit(unsigned *keys, T* values, unsigned bit, int end) {
  unsigned int i = threadIdx.x;
  unsigned int size = blockDim.x;
  unsigned k_i = keys[i];
  T v_i = values[i];
  unsigned int p_i = (k_i >> bit) & 1;
  keys[i] = p_i;
  __syncthreads();
  unsigned int T_before = plus_scan(keys);
  unsigned int T_total  = keys[size-1];
  unsigned int F_total  = size - T_total;
  __syncthreads();
  if (p_i) {
    keys[T_before - 1 + F_total] = k_i;
    values[T_before - 1 + F_total] = v_i;
  } else {
    keys[i - T_before] = k_i;
    values[i - T_before] = v_i;
  }
}

__global__ void create_flags(const int d_cols_sorted[], const float d_vals[], const int d_prefix_sum_flags[], int flags[], int size){	
	int num_threads = blockDim.x;
	int tid = threadIdx.x;
	int start = (blockIdx.x)*size;
	int end = (blockIdx.x+1)*size;
	if(tid == 0 && start == 0){
		flags[0] = 1;
		tid += num_threads;
	} 
	for(int i = start + tid; i<end;i+=num_threads){
		if(d_cols_sorted[i] != d_cols_sorted[i-1]){
			flags[i] = 1;
			//printf("i == %d flags[i] = %d\n",i, flags[i]);
		}
	}
}


void SegSortPairs(CudaContext& context,  int size_stream, int* rowStream, int* colStream, QValue* valueStream, int* segment_head_offsets, int total_num_segments_bins_1_to_6) {

  thrust::device_ptr<int> drowStream(rowStream);
  thrust::device_ptr<int> dcolStream(colStream);
  thrust::device_vector<int64_t> rows_cols_packed(size_stream);
  thrust::transform(
    thrust::make_zip_iterator(thrust::make_tuple(drowStream, dcolStream)),
    thrust::make_zip_iterator(thrust::make_tuple(drowStream+size_stream, dcolStream+size_stream)),
    rows_cols_packed.begin(),
    pack());

  int64_t *keys_device = thrust::raw_pointer_cast(rows_cols_packed.data());

  printf("\n\nSEG-SORT PAIRS DEMONSTRATION:\n\n");

  // Use CudaContext::GenRandom to generate 100 random integers between 0 and
  // 9.
  int N1 = size_stream;
  //MGPU_MEM(int) keys = context.GenRandom<int>(N1, 0, 99);

  MGPU_MEM(int64_t) keys = context.Malloc(keys_device,N1);

  // Fill values with ascending integers.
  MGPU_MEM(int) values = context.FillAscending<int>(N1, 0, 1);

  // Define 10 segment heads (for 11 segments in all).
  // const int NumSegs = total_num_segments_bins_1_to_6;
  // const int SegHeads[NumSegs] = { 4, 19, 22, 56, 61, 78, 81, 84, 94, 97 };
  MGPU_MEM(int) segments = context.Malloc(segment_head_offsets, total_num_segments_bins_1_to_6);

  //printf("Input keys:\n");
  //PrintArray(*keys, "%4d", 10);

  //printf("\nSegment heads:\n");
  //PrintArray(*segments, "%4d", 10);

  // Sort within segments.
  SegSortPairsFromIndices(keys->get(), values->get(), N1, segments->get(),total_num_segments_bins_1_to_6, context);

  printf("\nSorted data (segment heads are marked by *):\n");
  //PrintArrayOp(*keys, FormatOpMarkArray(" %c%2d", segment_head_offsets, total_num_segments_bins_1_to_6), 10);

  printf("\nSorted indices (segment heads are marked by *):\n");
 // PrintArrayOp(*values, FormatOpMarkArray(" %c%2d", segment_head_offsets, total_num_segments_bins_1_to_6), 10);

}



CSR sgpuSpMMWrapper(const CSR &dA, const CSR &dB, int *drowIds, const vector<int> &hv,int *dflops) {
  CSR dC;
  int *rowStream,*colStream;
  QValue *valueStream;
  int m = dA.rows;
  int *flops = new int[1];
  HANDLE_ERROR(cudaMemcpy(flops,dflops+m,4,cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMalloc((void**)&rowStream, *flops * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&colStream, *flops * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&valueStream,*flops * sizeof(QValue))); 
  

  //printf("total flops :%d\n",*flops); 
  
  gpu_compute_stream(dA, dB, drowIds, hv, rowStream, colStream, valueStream,dflops);
  HANDLE_ERROR(cudaGetLastError());
 
  /* 
  printing for checking correctness

  int *hrowStream = NULL;
  int *hcolStream = NULL;
  hrowStream = (int*) malloc(*flops * sizeof(int));
  hcolStream = (int*) malloc(*flops * sizeof(int));
  HANDLE_ERROR(cudaMemcpy(hrowStream,rowStream,*flops * sizeof(int),cudaMemcpyDeviceToHost));  
  HANDLE_ERROR(cudaMemcpy(hcolStream,colStream,*flops * sizeof(int),cudaMemcpyDeviceToHost));
  for(int i=0;i<(*flops);i++) {
	printf("row : %d ,col : %d \n",hrowStream[i],hcolStream[i]);
  } */
  
  int total_num_segments = ((*flops + FLOPS_SORT - 1)/FLOPS_SORT) - 1; // represents the size of the array
  //printf("total number of segments : %d\n" , total_num_segments);
  //printf("total flops : %d\n" , (hv[7] - hv[2]));
  int *flops_sort_arr;
  if(total_num_segments > 0) {
    HANDLE_ERROR(cudaMalloc((void**)&flops_sort_arr, total_num_segments * sizeof(int)));
    const int BLOCK_THREADS = 256;
    const unsigned NBLOCKS = qmin(65535, (m + BLOCK_THREADS - 1) / BLOCK_THREADS);
    compute_sorting_pointers<BLOCK_THREADS><<<NBLOCKS,BLOCK_THREADS>>>(flops_sort_arr, *flops, total_num_segments, rowStream);
  }
  
  /*
  printing for checking correctness
  int *hflops_sort_arr = NULL;
  hflops_sort_arr = (int*) malloc(total_num_segments * sizeof(int));   
  HANDLE_ERROR(cudaMemcpy(hflops_sort_arr, flops_sort_arr, total_num_segments * sizeof(int), cudaMemcpyDeviceToHost));
  for(int i=0;i<total_num_segments;i++) {
 	printf("pointer :%d\n" , hflops_sort_arr[i]);
  } 
  */
 
  // Pack (day, site) pairs into 64-bit integers.
  //thrust::device_ptr<int> dvalueStream(valueStream);
  
  //sort_data(*flops,drowStream,dcolStream,dvalueStream);
  //TODO handle segment array size as 0 and for -1, Also finally this method should final CSR
  //ContextPtr context = CreateCudaDevice(0);

  //SegSortPairs(*context, *flops, rowStream, colStream, valueStream, flops_sort_arr, total_num_segments);

  //uncomment later
  //create_flags<<<NBLOCKS, NTHREADS>>>(d_cols_sorted, d_vals, d_prefix_sum_flags, d_flags, N/NBLOCKS);

  /* 
  thrust::device_ptr<int> dIC = thrust::device_pointer_cast(dC.rowPtr);
  thrust::exclusive_scan(dIC, dIC + m + 1, dIC);
  int cNnz = dIC[m];
  printf("total number of nnz %d", cNnz);
  HANDLE_ERROR(cudaMalloc((void**)&dC.colInd, cNnz * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dC.values, cNnz * sizeof(QValue)));
  // performing the computation in matrix -- kernel.cu
  //sgpu_SpGEMM(dA, dB, drowIds, hv, dC, cNnz, temp_C_128_id, temp_C_128_val, temp_C_256_id, temp_C_256_val, temp_C_512_id, temp_C_512_val, temp_C_1024_id, temp_C_1024_val);
  cudaDeviceSynchronize();
  dC.nnz = cNnz;
  dC.rows = dA.rows;
  dC.cols = dB.cols; */
  return dC;
}
