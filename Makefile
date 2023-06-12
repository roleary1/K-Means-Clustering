# the compiler: gcc for C program, define as g++ for C++
CC = g++
NVCC = nvcc

UTCS_OPT ?= -O3

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS  =

TARGET = kmeans

all: $(TARGET)

# Link command:
$(TARGET): $(TARGET).o kmeans_kernel.o
	$(NVCC) $(CFLAGS) -o kmeans kmeans.o kmeans_kernel.o -lboost_program_options

# Compilation commands:
kmeans_kernel.o: kmeans_kernel.cu
	$(NVCC) $(CFLAGS) -o kmeans_kernel.o -c kmeans_kernel.cu

$(TARGET).o: kmeans.cpp
	$(NVCC) $(CFLAGS) -o kmeans.o -c kmeans.cpp -lboost_program_options

clean:
	$(RM) $(TARGET)
