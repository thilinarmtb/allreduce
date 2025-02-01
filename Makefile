FC ?= mpif90
FFLAGS = -O3
CC ?= mpicc
CFLAGS = -O3 -Wall -std=c99

SRCS=$(wildcard *.f)
TARGETS=$(SRCS:.f=)

.PHONY: format all clean

.SUFFIXES:

all: $(TARGETS)

%: %.f %.o ; $(FC) $(FFLAGS) -o $@ $^

%.o: %.c ; $(CC) $(CFLAGS) -c $< -o $@

format:
	clang-format -i *.c

clean:
	rm -f $(TARGETS)
