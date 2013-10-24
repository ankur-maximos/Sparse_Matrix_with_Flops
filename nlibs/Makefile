include ../Makefile.in

INCLUDES = -I. 

CFLAGS = $(COPTIONS) $(OPTFLAGS) $(INCLUDES)
LD = $(CC) -L. 

SOURCES=$(wildcard *.cc)
OBJS=$(shell echo $(SOURCES) | sed s/.cc/.o/g)

%.o: %.cc *.h
	@echo $(CFLAGS)
	$(CC) $(CFLAGS) -c $<

$(LIBNAME): $(OBJS) *.h 
	$(AR) $@ $(OBJS)
	$(RANLIB) $@

clean:
	rm -f *.o ; rm -f $(LIBNAME)

