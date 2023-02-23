include ../Makefile.inc

SRC_CU          = $(shell find ./ -name '*.cu')

all: AddVBiasTransposePlugin.so

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

AddVBiasTransposePlugin.so: $(SRC_CU:.cu=.o)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: clean
clean:
	rm -rf ./*.so ./*.o ./*.d ./*.trt ./temp* ./*.nsys-rep ./*.sqlite

.PHONY: test
test:
	clear
	make clean
	make all
	cp ./AddVBiasTransposePlugin.so ../
	nsys nvprof python testAddVBiasTransposePluginOnnx.py

