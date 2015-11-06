#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include "CSR.h"
#include "gpus/cuda_handle_error.h"

void outputDeviceLongArray(long *dflops, int m) {
  long *hflops = (long*)malloc(m * sizeof(long));
  HANDLE_ERROR(cudaMemcpy((void*)hflops, (void*) dflops, m * sizeof(long), cudaMemcpyDeviceToHost));
  printf("gpu hflops: ");
  for (int i = 0; i < m; ++i) {
    printf("%ld ", hflops[i]);
  }
  printf("\n");
  free(hflops);
}

void outputDeviceIntArray(const char *msg, int *dia, int m) {
  int *hia = (int*)malloc(m * sizeof(int));
  HANDLE_ERROR(cudaMemcpy((void*)hia, (void*)dia, m * sizeof(int), cudaMemcpyDeviceToHost));
  printf("%s", msg);
  for (int i = 0; i < m; ++i) {
    printf("%d ", hia[i]);
  }
  printf("\n");
  free(hia);
}

int inline qmin(const int a, const int b) {
  if (a < b) return a;
  return b;
}

__device__ inline int dqueueId(long x) {
  if(x == 0) return 1;
  else if (x == 1) return 2;
  else if (x > 512) return 7;
  else if (x > 64 && x <= 512) return 6;
  else if (x > 16 && x <= 64) return 5;
  else if (x > 4 && x<=16) return 4;
  else return 3; 
}

/*
__device__ inline int dqueueId(long x) {
  //assert (x > 0);
  //if (x == 0) return 0;
  if (x == 1) return 1;
  else if (x > 1024) return 63;
  else if (x > 256 && x <= 512) return 10;
  else if (x > 512 && x <= 1024) return 11;
  int ret = 2;
  int up = 2;
  for (up = 2; ; up *= 2, ++ret) {
    if (x <= up) return ret;
  }
  //return -1;
} */

template <int BLOCK_THREADS>
__global__ void gcomputeFlops(const int m, const int* dIA, const int *dJA, const int* dIB,
     int *drowIds,int *dflopIds) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < m; i += blockDim.x * gridDim.x) {
    long tmpRowFlops = 0;
    for (int ap = dIA[i]; ap < dIA[i + 1]; ++ap) {
      int a = dJA[ap];
      int BrowFlops = dIB[a + 1] - dIB[a];
      tmpRowFlops += BrowFlops;
    }
    //dflops[i] = tmpRowFlops;
    //if (q == 0) assert (dIA[i + 1] - dIA[i] == 1);
    //__syncthreads();
    
    dflopIds[i] = tmpRowFlops;
    drowIds[i] = i;
  }
}

/* kernel for assigning binids for rows */
template <int BLOCK_THREADS>
__global__ void gcomputeBinId(const int m, const int* dflopIds,short *dbinIds) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int q = 0;
    for(int i=tid;i<m;i+=blockDim.x*gridDim.x) {
	q = dqueueId(dflopIds[i]);
	dbinIds[i] = q;
    }
}

thrust::device_vector<int> computeHistogram(short *sortedKeys, int m) {
  thrust::device_ptr<short> sortedKeys_dptr(sortedKeys);
  int num_bins = sortedKeys_dptr[m - 1] + 1;
  thrust::device_vector<int> histogram;
  histogram.resize(num_bins + 1);
  histogram[0] = 0;
  thrust::counting_iterator<int> search_begin(0);
  thrust::upper_bound(sortedKeys_dptr, sortedKeys_dptr + m,
                      search_begin, search_begin + num_bins,
                      histogram.begin() + 1);
  return histogram;
}

/* Classification of flops to bins */
std::vector<int> gpuFlopsClassify(const CSR &dA, const CSR &dB, int **drowIdsp, int **dflopId ) { 
  const int m = dA.rows;
  int *dflopIds = NULL, *drowIds = NULL;
  short *dbinIds = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&dflopIds, (m+1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&drowIds, m * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dbinIds, (m+1) * sizeof(short)));
 
  //unsigned long long int *d_totflops;
  //HANDLE_ERROR(cudaMalloc((void**)&d_totflops, sizeof(unsigned long long int)));
  //long *dflops = NULL;
  //HANDLE_ERROR(cudaMalloc((void**)&dflops, m * sizeof(long)));
  HANDLE_ERROR(cudaMemset(dflopIds, 0, 4));
  const unsigned BLOCK_THREADS = 256;
  const unsigned NBLOCKS = qmin(65535, (m + BLOCK_THREADS - 1) / BLOCK_THREADS);
  gcomputeFlops<BLOCK_THREADS><<<NBLOCKS, BLOCK_THREADS>>>(m, dA.rowPtr, dA.colInd, dB.rowPtr, drowIds, dflopIds+1);

  std::vector<int> v;
  thrust::device_ptr<int> dflopIds_dptr(dflopIds);
  thrust::device_ptr<int> drowIds_dptr(drowIds);
  //thrust::device_ptr<short> dbinIds_dptr(dbinIds);
  thrust::stable_sort_by_key(dflopIds_dptr+1, dflopIds_dptr + m + 1, drowIds_dptr);
  gcomputeBinId<BLOCK_THREADS><<<NBLOCKS, BLOCK_THREADS>>>(m + 1, dflopIds, dbinIds);
  thrust::inclusive_scan(dflopIds_dptr+1,dflopIds_dptr+m+1,dflopIds_dptr+1);
  //int *flops = new int[1];
  //HANDLE_ERROR(cudaMemcpy(flops,dflopIds+m,4,cudaMemcpyDeviceToHost));

  int *hflopIds = NULL; 
  int *hrowIds = NULL;
  short *hbinIds = NULL;
  hflopIds = (int *)malloc((m+1) * sizeof(int));
  hrowIds = (int*)malloc(m * sizeof(int));
  hbinIds = (short*)malloc((m+1) * sizeof(short));
  HANDLE_ERROR(cudaMemcpy(hflopIds,dflopIds,(m+1) * sizeof(int),cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(hrowIds,drowIds, m * sizeof(int),cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(hbinIds,dbinIds, (m+1) * sizeof(short),cudaMemcpyDeviceToHost));

  printf("flops array is : \n");
  for(int i=0;i<=m;i++) {
	printf("%d ",*(hflopIds+i));
  }
  printf("\n");
  printf("row array is : \n");
  for(int i=0;i<m;i++) {
	printf("%d ",*(hrowIds+i));
  }
  printf("\n");

  for(int i = 0;i<=m ;i++) {
     	printf("%hd ",*(hbinIds+i));
  }

  printf("\n");

  thrust::device_vector<int> dhist = computeHistogram(dbinIds, m+1);
  *drowIdsp = drowIds;
  *dflopId = dflopIds;
  v.resize(dhist.size());
  thrust::copy(dhist.begin(), dhist.end(), v.begin());
  
  /*if (v.size() < 9) {
    int osize = v.size();
    int back = v.back();
    v.resize(9);
    for (int i = osize; i < 9; ++i) {
      v[i] = back;
    }
  }  */
  for(int i=0;i<v.size();i++) {
  	printf("hv :%d \n", v[i]);
  }
  return v;
}
