CUDA_PATH=home/amir/
INCLUDE_CUDA_RUNTIME_API=${CUDA_PATH}/cuda/tools/cudart
INCLUDE_CUDA=${CUDA_PATH}/cuda/import -I ${CUDA_PATH}/cuda/apps/cutil/inc -I ${CUDA_PATH}/cuda/tools/coop
INCLUDE_CUCODE=${CUDA_PATH}/cucode/src
INCLUDE_SAMPLES=${CUDA_PATH}/samples/common/inc

LIBS=

ARCH=sm_35

all: whatever
	nvcc  -I${INCLUDE_CUCODE} -I ${INCLUDE_CUDA_RUNTIME_API} -I ${INCLUDE_CUDA} -I ${INCLUDE_SAMPLES} -c code.cu -o code.o -arch $(ARCH)  -rdc=true -Xcompiler -ggdb
	nvcc  -I${INCLUDE_CUCODE} -I ${INCLUDE_CUDA_RUNTIME_API} -I ${INCLUDE_CUDA} -I ${INCLUDE_SAMPLES}  code.o whatever1.o whatever2.o test.cpp -arch $(ARCH) -o runme -Xcompiler -ggdb -lcufft

whatever:
	nvcc  -I ${INCLUDE_CUDA_RUNTIME_API} -I ${INCLUDE_CUDA} -I ${INCLUDE_SAMPLES} -c whatever1.cu -rdc=true -arch $(ARCH)
	nvcc  -I ${INCLUDE_CUDA_RUNTIME_API} -I ${INCLUDE_CUDA} -I ${INCLUDE_SAMPLES} -c whatever2.cu -rdc=true -arch $(ARCH)

clean:
	rm -f *.o runme

valgrind:
	valgrind --tool=callgrind --collect-jumps=yes --dump-instr=yes a.out
