ARCH=sm_35

all:
	nvcc reduce.cu -o reduce.out -arch $(ARCH)  -rdc=true -Xcompiler -ggdb -std=c++11
	
clean:
		rm -f *.out	