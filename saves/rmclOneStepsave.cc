
/*CSR CSR::rmclOneStep(const CSR &B, thread_data_t *thread_datas) const {
  CSR C;
  C.rowPtr = (int*)calloc(rows + 1, sizeof(int));
  int* cColIndV[8];
  double* cValuesV[8];
  int offsets[9];
#pragma omp parallel
  {
      omp_CSR_IC_nnzC(rowPtr, colInd, B.rowPtr, B.colInd, rows, B.cols, thread_datas, C.rowPtr, C.nnz);
#pragma omp barrier
      int threadId = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      double *x = thread_datas[threadId].x;
      bool *xb = thread_datas[threadId].xb;
      const int rowsPerThread = (rows + nthreads) / nthreads;
      int rowStart = threadId * rowsPerThread;
      int rowEnd = std::min((threadId + 1) * rowsPerThread, rows);
      const int threadCnnz = C.rowPtr[rowEnd] - C.rowPtr[rowStart];
      cColIndV[threadId] = (int*)malloc(sizeof(int) * threadCnnz + LEVEL1_DCACHE_LINESIZE);
      cValuesV[threadId] = (double*)malloc(sizeof(double) * threadCnnz + LEVEL1_DCACHE_LINESIZE);
#ifdef debugging
      printf("First omp barrier with thread %d threadCnnz=%d rowStart=%d rowEnd=%d\n", threadId, threadCnnz, rowStart, rowEnd);
#endif
#pragma omp barrier
      int *cColInd = cColIndV[threadId];
      double *cValues = cValuesV[threadId];
      int cOffset = 0;
      for (int i = rowStart; i < rowEnd; ++i) {
        int count = C.rowCount(i);
          int cCount = processCRowI(x, xb,
              rowPtr[i + 1] - rowPtr[i], colInd, values, //A
              B.rowPtr, B.colInd, B.values, //B
              cColInd + cOffset, cValues + cOffset); //C
          arrayInflationR2(cValues + cOffset, count, cValues);
          pair<double, double> maxSum = arrayMaxSum(cValues, count);
          double rmax = maxSum.first;
          double rsum = maxSum.second;
          double thresh = computeThreshold(rsum / count, rmax);
          arrayThreshPruneNormalize(thresh, cColInd, cValues,
              &count, cColInd + cOffset, cValues + cOffset);
          C.rowPtr[i] = count;
          cOffset += count;
      }
      offsets[threadId + 1] = cOffset;
#pragma omp barrier
#pragma omp single
      {
        offsets[0] = 0;
        for (int i = 0; i < nthreads; ++i) {
          offsets[i + 1] += offsets[i];
        }
        int nnz = offsets[nthreads];
        C.colInd = (int*)malloc(nnz * sizeof(int));
        C.values = (double*) malloc(nnz * sizeof(double));
      }
#pragma omp barrier
      const int threadCnnzStay = offsets[threadId + 1] - offsets[threadId];
      memcpy(C.colInd + offsets[threadId], cColIndV[threadId], threadCnnzStay * sizeof(int));
      memcpy(C.values + offsets[threadId], cValuesV[threadId], threadCnnzStay * sizeof(double));
#pragma omp barrier
      free(cColIndV[threadId]);
      free(cValuesV[threadId]);
#pragma omp barrier
  }
  return C;
}*/
