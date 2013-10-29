CFLAGS=-std=c++0x -g -openmp -fast #-Dprofiling
#-static-intel 
SOURCES=$(wildcard *.cc)
OBJS=$(shell echo $(SOURCES) | sed s/.cc/.o/g)
EXE=spmm.x
#VPATH := ..:.
all:$(EXE)

CC=icpc
$(EXE):$(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS) -openmp

%.o:%.cc 
	$(CC) $(CFLAGS) $^ -c 
clean:
	rm -rf *.o *.x
