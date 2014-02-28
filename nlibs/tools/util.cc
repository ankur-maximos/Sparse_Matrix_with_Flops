#include "util.h"
#include <algorithm>

double computeThreshold(double avg, double max) {
	double ret = MLMCL_PRUNE_A * avg * (1 - MLMCL_PRUNE_B * (max - avg));
	ret = (ret > 1.0e-7) ? ret : 1.0e-7;
	ret = (ret > max) ? max : ret;
	return ret;
}

double arrayMax(const double values[], const int count) {
  double rmax = 0.0;
  for ( int i = 0; i < count; ++i) {
    if (rmax < values[i]) {
      rmax = values[i];
    }
  }
  return rmax;
}

pair<double, double> arrayMaxSum(const double values[], const int count) {
  double rmax = 0.0;
  double rsum = 0.0;
  for ( int i = 0; i < count; ++i) {
    if (rmax < values[i]) {
      rmax = values[i];
    }
    rsum += values[i];
  }
  return make_pair(rmax, rsum);
}

double arraySum(const double *restrict values, const int count) {
  double rsum = 0.0;
  for (int i = 0; i < count; ++i) {
    rsum += values[i];
  }
  return rsum;
}

void arrayInflationR2(const double *restrict ivalues, const int count, double *restrict ovalues) {
  for (int i = 0; i < count; ++i) {
    ovalues[i] = ivalues[i] * ivalues[i];
  }
}

double arrayThreshPruneNormalize(const double thresh, const int rindices[], const double rvalues[],
    int* count, int indices[], double values[]) {
	//int* indicesToRetain = (int*)malloc(sizeof(int) * (*count));
	int i, j;
	double sum = 0;
	for (i = 0, j = 0; i < *count; ++i) {
		if (values[i] >= thresh) {
			sum += rvalues[i];
			//indicesToRetain[j++] = i;
      indices[j] = rindices[i];
      values[j++] = rvalues[i];
		}
	}
  //normalize
	for (i = 0; i < j; ++i) {
		//indices[i] = rindices[indicesToRetain[i]];
		//values[i] = rvalues[indicesToRetain[i]] / sum;
		values[i] = values[i] / sum;
	}
	*count = j;
	//free(indicesToRetain);
	return sum;
}

void arrayOutput(const char *msg, FILE* fp, const double datas[], int len) {
  fprintf(fp, "%s", msg);
  for (int i = 0; i < len; ++i) {
    fprintf(fp, "%e ", datas[i]);
  }
  fprintf(fp, "\n");
  fflush(fp);
}

void arrayOutput(const char *msg, FILE* fp, const int datas[], int len) {
  fprintf(fp, "%s", msg);
  for (int i = 0; i < len; ++i) {
    fprintf(fp, "%d ", datas[i]);
  }
  fprintf(fp, "\n");
  fflush(fp);
}

void arrayOutput(const char *msg, FILE* fp, const vector<int> &datas) {
  fprintf(fp, "%s", msg);
  for (int i = 0; i < datas.size(); ++i) {
    fprintf(fp, "%d ", datas[i]);
  }
  fprintf(fp, "\n");
  fflush(fp);
}

void arrayOutput(const char *msg, FILE* fp, const vector<double> &datas) {
  fprintf(fp, "%s", msg);
  for (int i = 0; i < datas.size(); ++i) {
    fprintf(fp, "%lf ", datas[i]);
  }
  fprintf(fp, "\n");
  fflush(fp);
}

//len is the length of counts array
void prefixSumToCounts(const int prefixSum[], const int len, int *counts) {
  for (int i = 0; i < len; ++i) {
    counts[i] = prefixSum[i + 1] - prefixSum[i];
  }
}

void arrayEqualPartition(int prefixSum[], const int n, const int nthreads, int ends[]) {
  const int chunk_size = (prefixSum[n] + nthreads - 1) / nthreads;
  ends[0] = 0;
  for (int i = 0, now = 0; i < nthreads - 1; ++i) {
    const int target = std::min((i + 1) * chunk_size, prefixSum[n]);
    int* begin = prefixSum + now;
    int* upper = std::upper_bound(begin, prefixSum + n + 1, target);
    ends[i + 1] = std::max((int)(upper - prefixSum - 1), now + 1);
    ends[i + 1] = std::min(ends[i + 1], n);
    now = ends[i + 1];
  }
  ends[nthreads] = n;
}

void randomPermutationVector(int* &P, int len) {
  srand(time(NULL));
  P = (int*)malloc(len * sizeof(int));
  for (int i = 0; i < len; ++i) {
    int pos = rand() % (i + 1);
    P[i] = P[pos];
    P[pos] = i;
  }
}
