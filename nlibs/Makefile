include ../Makefile.in

INCLUDES = -I. 

CFLAGS = $(COPTIONS) $(OPTFLAGS) $(CINCLUDES)
CUFLAGS = $(CUOPTFLAGS) $(CUINCLUDES) -I.
LD = $(CC) -L. 

#SOURCES=$(wildcard *.cc)
SOURCES=$(shell find . -name "*.cc" -o -name "*.cu")
OBJS=$(shell echo $(SOURCES) | sed "s/\.cc/\.o/g" | sed "s/\.cu/\.o/g")
HEADERS=$(shell find . -name "*.h")
%.o: %.cc $(HEADERS) ../Makefile.in
	@echo $(CFLAGS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cu ../Makefile.in $(shell find . -name "*.cuh")
	$(NVCC) $(CUFLAGS) $< -c -o $@

$(LIBNAME): $(OBJS) $(HEADERS)
	$(AR) $@ $(OBJS)
	$(RANLIB) $@

clean:
	rm -f *.o gpus/*.o tools/*.o
	rm -f $(LIBNAME)

