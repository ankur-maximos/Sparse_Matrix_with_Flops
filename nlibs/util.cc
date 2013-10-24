#include "util.h"

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

double arraySum(const double values[], const int count) {
  double rsum = 0.0;
  for (int i = 0; i < count; ++i) {
    rsum += values[i];
  }
  return rsum;
}

void arrayInflationR2(const double ivalues[], const int count, double ovalues[]) {
  for (int i = 0; i < count; ++i) {
    ovalues[i] = ivalues[i] * ivalues[i];
  }
}

double arrayThreshPrune(const double thresh, int* count, int indices[], double values[]) {
	int* indicesToRetain = (int*)malloc(sizeof(int) * (*count));
	int i, j;
	double sum = 0;
	for (i = 0, j = 0; i < *count; ++i) {
		if (values[i] >= thresh) {
			sum += values[i];
			indicesToRetain[j++] = i;
		}
	}
	for (i = 0; i < j; ++i) {
		indices[i] = indices[indicesToRetain[i]];
		values[i] = values[indicesToRetain[i]];
	}
	*count = j;
	free(indicesToRetain);
	return sum;
}

