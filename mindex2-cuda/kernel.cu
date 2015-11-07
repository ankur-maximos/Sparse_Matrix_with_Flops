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

#include <thrust/count.h>

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

#include <inttypes.h>

#define ENABLE_DEBUG 0
#define ENABLE_PRINT 0

using namespace std;
using namespace mgpu;

//variable that gives average size of segment?
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
  
  //printf("cur_row %d next_row %d\n ", cur_row, next_row); 

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

__global__ void get_segment_heads_for_CSR_reduction(const int64_t d_row_cols_sorted[], const QValue d_vals[], int segment_heads_for_CSR_reduction[], int size){
  int num_threads = blockDim.x;
  int tid = threadIdx.x;
  int start = (blockIdx.x)*size;
  int end = (blockIdx.x+1)*size;
  /*if(tid == 0 && start == 0){
    segment_heads_for_CSR_reduction[0] = 1;
    tid += num_threads;
  } */
  for(int i = start + tid; i<end;i+=num_threads){
    if(d_row_cols_sorted[i] != d_row_cols_sorted[i-1]){
      segment_heads_for_CSR_reduction[i] = i;
      //printf("i == %d flags[i] = %d\n",i, flags[i]);
    }
  }
}

__global__ void create_flags(const int64_t d_row_cols_sorted[], const QValue d_vals[], const int d_prefix_sum_flags[], int flags[], int size){
	int num_threads = blockDim.x;
	int tid = threadIdx.x;
	int start = (blockIdx.x)*size;
	int end = (blockIdx.x+1)*size;
	if(tid == 0 && start == 0){
		flags[0] = 1;
		tid += num_threads;
	} 
	for(int i = start + tid; i<end;i+=num_threads){
		if(d_row_cols_sorted[i] != d_row_cols_sorted[i-1]){
			flags[i] = 1;
			//printf("i == %d flags[i] = %d\n",i, flags[i]);
		}
	}
}


void SegSortPairs(CudaContext& context,  int size_stream, int* rowStream, int* colStream, QValue* valueStream, int* segment_head_offsets, int NumSegs) {

  thrust::device_ptr<int> drowStream(rowStream);
  thrust::device_ptr<int> dcolStream(colStream);
  thrust::device_vector<int64_t> rows_cols_packed(size_stream);
  thrust::transform(
    thrust::make_zip_iterator(thrust::make_tuple(drowStream, dcolStream)),
    thrust::make_zip_iterator(thrust::make_tuple(drowStream+size_stream, dcolStream+size_stream)),
    rows_cols_packed.begin(),
    pack());

  //TODO: Decide whether to delete rowstream and colstream or whether to just reuse this memory in the unpacked rows and cols? 
  //Maybe could consider deleting and recreating because sizeof(rowstream) == total #flops and size of final unpacked rowstream == total #nnzs in C. There could be a huge diff between the 2

  //int64_t *keys_device = thrust::raw_pointer_cast(rows_cols_packed.data());

  thrust::host_vector<int64_t> rows_cols_packed_host = rows_cols_packed;

  int64_t *keys_host = &rows_cols_packed_host[0];


  MGPU_MEM(int64_t) mgpu_rows_cols_packed = context.Malloc(keys_host,size_stream);
  MGPU_MEM(QValue) values = context.Malloc(valueStream, size_stream);
  MGPU_MEM(int) segments = context.Malloc(segment_head_offsets, NumSegs);

#if ENABLE_DEBUG
  printf("\n\nSEG-SORT PAIRS STARTING:\n\n");
  cout<<"ROWS_COLS_PACKED (KEYS):\n";
  for(int i=0;i<rows_cols_packed_host.size();i++){
    cout<<keys_host[i]<<" ";
  }
  cout<<endl;

  cout<<"VALUES: \n";
  PrintArray(*values, "%9f", 10);

  cout<<"total_num_segments"<<NumSegs<<endl;
  printf("Input keys:\n");
  HANDLE_ERROR(cudaMemcpy(keys_host,mgpu_rows_cols_packed->get(),size_stream * sizeof(int64_t),cudaMemcpyDeviceToHost));  
  for(int i=0;i<size_stream;i++){
    cout<<keys_host[i]<<" ";
  }
  cout<<endl;

  printf("\nSegment heads:\n");
  //PrintArray(*segments, "%4d", 10);
#endif

  // Sort within segments.
  SegSortPairsFromIndices(mgpu_rows_cols_packed->get(), values->get(), size_stream, segments->get(),NumSegs, context);

#if ENABLE_DEBUG
  printf("\nSorted keys :\n");
  HANDLE_ERROR(cudaMemcpy(keys_host,mgpu_rows_cols_packed->get(),size_stream * sizeof(int64_t),cudaMemcpyDeviceToHost));  
  for(int i=0;i<rows_cols_packed_host.size();i++){
    cout<<keys_host[i]<<" ";
  }
  cout<<endl;

  printf("\nSorted values :\n");
  PrintArray(*values, "%9f", 10);
#endif

  MGPU_MEM(int64_t) mgpu_reduced_rows_cols_packed = context.Malloc<int64_t>(size_stream);
  MGPU_MEM(QValue) mgpu_reduced_vals = context.Malloc<QValue>(size_stream);
  
  //reduce on row_cols_packed

  //numSegments = number of unique r,c values 
  int numSegments;
  ReduceByKey(mgpu_rows_cols_packed->get(), values->get(), size_stream,
    QValue(0.0), mgpu::plus<QValue>(), mgpu::equal_to<int64_t>(), mgpu_reduced_rows_cols_packed->get(),
    mgpu_reduced_vals->get(), &numSegments, (int*)0, context);

#if ENABLE_DEBUG
  printf("\nReduced keys:\n");
  //PrintArray(*keysDestDevice, numSegments, "%4f", 10);
  HANDLE_ERROR(cudaMemcpy(keys_host, mgpu_reduced_rows_cols_packed->get(), numSegments * sizeof(int64_t), cudaMemcpyDeviceToHost));  
  for(int i=0;i<numSegments;i++){
    cout<<keys_host[i]<<" ";
  }
  cout<<endl;


  printf("\nReduced values:\n");
  PrintArray(*mgpu_reduced_vals, numSegments, "%4f", 10);
#endif

  //unpack keysDestDevice into rows and cols
  int64_t* d_reduced_rows_cols_packed = mgpu_reduced_rows_cols_packed->get();

  thrust::device_ptr<int64_t> thrust_reduced_rows_cols_packed(d_reduced_rows_cols_packed);

  thrust::transform(
    thrust_reduced_rows_cols_packed,
    thrust_reduced_rows_cols_packed+numSegments,
    thrust::make_zip_iterator(thrust::make_tuple(drowStream, dcolStream)),
    unpack());

#if ENABLE_PRINT  
  int *hrowStream = NULL;
  int *hcolStream = NULL;
  QValue* hvalStream = NULL;

  hrowStream = (int*) malloc(numSegments * sizeof(int));
  hcolStream = (int*) malloc(numSegments * sizeof(int));
  hvalStream = (QValue*) malloc(numSegments * sizeof(QValue));

  HANDLE_ERROR(cudaMemcpy(hrowStream,rowStream,numSegments * sizeof(int),cudaMemcpyDeviceToHost));  
  HANDLE_ERROR(cudaMemcpy(hcolStream,colStream,numSegments * sizeof(int),cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(hvalStream,mgpu_reduced_vals->get(), numSegments * sizeof(QValue),cudaMemcpyDeviceToHost));

  for(int i=0;i<(numSegments);i++) {
    printf("row : %d ,col : %d, val : %f\n",hrowStream[i],hcolStream[i],hvalStream[i]);
  }

  
#endif
  //need to handle bin 7
}


template <typename T>
struct is_odd : public thrust::unary_function<T,bool>
{
    __host__ __device__
    bool operator()(T x)
    {
        return x % 2;
    }
};


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
 
   
 // printing for checking correctness
#if ENABLE_DEBUG

  int *hrowStream = NULL;
  int *hcolStream = NULL;
  QValue* hvalStream = NULL;
  hrowStream = (int*) malloc(*flops * sizeof(int));
  hcolStream = (int*) malloc(*flops * sizeof(int));
  hvalStream = (QValue*) malloc(*flops * sizeof(QValue));
  HANDLE_ERROR(cudaMemcpy(hrowStream,rowStream,*flops * sizeof(int),cudaMemcpyDeviceToHost));  
  HANDLE_ERROR(cudaMemcpy(hcolStream,colStream,*flops * sizeof(int),cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(hvalStream,valueStream,*flops * sizeof(QValue),cudaMemcpyDeviceToHost));

  for(int i=0;i<(*flops);i++) {
	  printf("row : %d ,col : %d , val: %f\n",hrowStream[i],hcolStream[i],hvalStream[i]);
  } 
#endif
  
  int total_num_segments = ((*flops + FLOPS_SORT - 1)/FLOPS_SORT) - 1; // represents the size of the array
  printf("total number of segments : %d\n" , total_num_segments);
  //printf("total flops : %d\n" , (hv[7] - hv[2]));
  int *d_segment_heads;
  if(total_num_segments > 0) {
    HANDLE_ERROR(cudaMalloc((void**)&d_segment_heads, total_num_segments * sizeof(int)));
    const int BLOCK_THREADS = 256;
    const unsigned NBLOCKS = qmin(65535, (m + BLOCK_THREADS - 1) / BLOCK_THREADS);
    compute_sorting_pointers<BLOCK_THREADS><<<NBLOCKS,BLOCK_THREADS>>>(d_segment_heads, *flops, total_num_segments, rowStream);
  }
  
  /*
  printing for checking correctness
  */
  //prints the segment heads

  int *h_segment_heads = NULL;
  h_segment_heads = (int*) malloc(total_num_segments * sizeof(int));   
  HANDLE_ERROR(cudaMemcpy(h_segment_heads, d_segment_heads, total_num_segments * sizeof(int), cudaMemcpyDeviceToHost));

  int idx_of_last_minus1;
  for(idx_of_last_minus1=total_num_segments-1 ; idx_of_last_minus1>0 && (h_segment_heads[idx_of_last_minus1] == -1); idx_of_last_minus1--){
  }
  cout<<"idx_of_last_minus1 = "<<idx_of_last_minus1<<endl;

#if ENABLE_DEBUG
  for(int i=0;i<total_num_segments;i++) {
 	  printf("segment head %d :%d\n" ,i, h_segment_heads[i]);
  }
#endif

/* flops array - prefix sum of flops per row after rows are sorted in ascending order of flops
    Note that flops[0] will always be 0. Also, flops array stores all the rows that have 0 flops also. 

*/
//  d_segment_heads contains my segment heads
/*
    hv array - contains info about starting indices of each bin in flops array. There are at most 7 bins in total and total size of hv is at most 9 (first 2 elements are dummy). Starting index of Bin #i (where i ranges from 1..7) is located in hv[i+1]. This value will point to the index in flops array containing the first non-zero value (flops is sorted so starting elements may be 0 - see above info about flops array for more details).

hv[8] -> index where bin 7 starts in flops array

*/
  // Pack (day, site) pairs into 64-bit integers.
  //thrust::device_ptr<int> dvalueStream(valueStream);
  
  //sort_data(*flops,drowStream,dcolStream,dvalueStream);
  
  //TODO handle segment array size as 0 and for -1, Also finally this method should final CSR

  ContextPtr context = CreateCudaDevice(0);
 
  //reducing total_num_segments by 1 as the last element in d_segment_heads contains -1
  //total_num_segments = idx_of_last_minus1 + 1
  SegSortPairs(*context, *flops, rowStream, colStream, valueStream, d_segment_heads, idx_of_last_minus1+1);

  //get_segment_heads_for_CSR_reduction(const int64_t d_row_cols_sorted[], const QValue d_vals[], int segment_heads_for_CSR_reduction[], int size);

  //Iterator values_end = thrust::remove_if(values.begin(), values.end(), is_odd<int>());

  
  // since the values after values_end are garbage, we'll resize the vector
  //values.resize(values_end - values.begin());

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
