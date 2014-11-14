#ifndef PROCESS_ARGS_H
#define PROCESS_ARGS_H

#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "qrmcl.h"

struct Options {
  bool calcChange = false;
  int maxIters = 5;
  char inputFileName[200];
  bool stats = false;
  int stride = 512;
  int ptile = 2;
  RunOptions rmclOption;
  int br = 2, bc = 8;
};

extern Options options;

int process_args(int argc, char **argv);
void print_args();
#endif
