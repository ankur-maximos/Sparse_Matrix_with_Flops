SOURCES=cusparse_spmm.cu
OBJS=$(SOURCES:%.cu=%.o)
EXE=$(SOURCES:%.cu=%.x)
CFLAGS=-arch sm_20 -I/home/niuq/code/gpu/cusplibrary
CC=nvcc
all:$(EXE) 
%.x:%.o
	$(CC) $^ -o $@ -lcusparse
%.o:%.cu
	$(CC) $(CFLAGS) $^ -c -g -O3
CFLAGS:
	@echo $(CFLAGS)
LDFLAGS:
	@echo $(LDFLAGS)
clean:
	rm -rf $(OBJS) $(EXE)

