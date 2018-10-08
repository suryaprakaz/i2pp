###################
# rgb2grayscale   #
###################

# Location of CUDA Toolkit
CUDA_PATH      := /usr/local/cuda

# Host machine default compiler
HOST_COMPILER  := g++

# NVIDIA C Compiler compiler path
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# OpenCV CFLAG
CCFLAGS        := `pkg-config --cflags opencv`

# OpenCV Linker FLAG
LDFLAGS        := `pkg-config --libs opencv`

#####################################################################

all: build

build: rgb2grayscale

rgb2grayscale.o: rgb2grayscale.cu
	$(NVCC) $(CCFLAGS) -o $@ -c $<

rgb2grayscale: rgb2grayscale.o
	$(NVCC) $(LDFLAGS) -o $@ $+

clean:
	rm -f rgb2grayscale.o rgb2grayscale
