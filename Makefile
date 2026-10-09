# Makefile - CUDA RSA case study
#   make                 build ./rsa_cuda  (default GPU arch: sm_61 = GeForce MX250 / Pascal)
#   make ARCH=sm_75      e.g. Tesla T4 (Google Colab), sm_86 = RTX 30xx, sm_89 = RTX 40xx
#   make run             demo + verification + benchmarks
#   make regs            show per-kernel register / spill usage reported by ptxas

NVCC     ?= nvcc
ARCH     ?= sm_61
CCBIN    ?=
NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
HOSTOPT  := -Xcompiler -O3,-march=native,-fopenmp
LDFLAGS  := -lgomp
ifneq ($(CCBIN),)
  NVFLAGS += -ccbin $(CCBIN)
endif

SRC := src/rsa_cuda.cu
DEP := src/bigint.cuh src/rsa_key.h

# The fully unrolled 64-limb carry chains are deep enough to overflow the
# default 8 MB stack of NVIDIA's front end (cicc -> "Segmentation fault"),
# so the stack limit is raised for the compiler process first.
STACK := ulimit -s unlimited 2>/dev/null || ulimit -s $$(ulimit -H -s);

rsa_cuda: $(SRC) $(DEP)
	$(STACK) $(NVCC) $(NVFLAGS) $(HOSTOPT) $(SRC) -o $@ $(LDFLAGS) $(EXTRA_LDFLAGS)

src/rsa_key.h:
	python3 tools/keygen.py

regs: $(SRC) $(DEP)
	$(STACK) $(NVCC) $(NVFLAGS) -Xptxas -v -c $(SRC) -o /dev/null 2>&1 | grep -E "Compiling|registers|spill"

run: rsa_cuda
	./rsa_cuda all

clean:
	rm -f rsa_cuda

.PHONY: regs run clean
