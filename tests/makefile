include ../Makefile.in
LIBDIR:=../$(LIBDIR)
NRMCL=$(LIBDIR)/$(LIBNAME)
LDFLAGS=$(NRMCL) -openmp $(LDOPTIONS)
CFLAGS = $(COPTIONS) $(OPTFLAGS) $(CINCLUDES)
CFLAGS += -I$(LIBDIR)

SOURCES=$(wildcard *.cc)
OBJS=$(shell echo $(SOURCES) | sed s/.cc/.o/g)
EXES=$(shell echo $(SOURCES) | sed s/.cc/.x/g)
all:$(EXES)

CC=icpc
%.x:%.o $(NRMCL)
	@echo $(LDFLAGS)
	$(CC) -o $@ $< $(LDFLAGS)

$(NRMCL):$(shell find $(LIBDIR) -name '*.cc' -o -name '*.h' -o -name '*.cu')
	(cd $(LIBDIR); make)

%.o:%.cc 
	$(CC) $(CFLAGS) $^ -c 

clean:
	rm -rf *.o *.x

diskclean:
	rm -rf *.o *.x
	make -C $(LIBDIR) clean

test:
	for exe in $(shell ls *.x); \
	do \
		echo $${exe}; \
		./$${exe}; \
	done
