NVCC = nvcc

.PHONY: clean

all: build

build: bvh

bvh: bvh.o main.o tri_contact.o
	$(NVCC) -o $@ $+

%.o: %.cu
	$(NVCC) $*.cu -dc $*.o 

clean:
	rm *.o