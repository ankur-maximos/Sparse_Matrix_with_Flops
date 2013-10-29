include Makefile.in
NRMCL=$(LIBDIR)/$(LIBNAME)
LDFLAGS=$(NRMCL) -openmp
CFLAGS = $(COPTIONS) $(OPTFLAGS) $(INCLUDES)
CFLAGS += -Inlibs

SOURCES=$(wildcard *.cc)
OBJS=$(shell echo $(SOURCES) | sed s/.cc/.o/g)
EXE=nrmcl.x
#VPATH := ..:.
all:$(EXE)

CC=icpc
$(EXE):$(OBJS) $(NRMCL)
	$(CC) -o $@ $^ $(LDFLAGS)

$(NRMCL):$(shell find $(LIBDIR) -name '*.cc' -o -name '*.h')
	(cd $(LIBDIR); make)

%.o:%.cc 
	$(CC) $(CFLAGS) $^ -c 
clean:
	rm -rf *.o *.x
	make -C $(LIBDIR) clean
