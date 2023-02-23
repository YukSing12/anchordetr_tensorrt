include ../Makefile.inc

SRC_CU          = $(shell find ./ -name '*.cu')

all: MaskedSoftmaxPlugin.so

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

MaskedSoftmaxPlugin.so: $(SRC_CU:.cu=.o)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: clean
clean:
	rm -rf ./*.so ./*.o ./*.d ./*.trt ./temp* ./*.nsys-rep ./*.sqlite

.PHONY: test
test:
	clear
	make clean
	make all
	cp ./MaskedSoftmaxPlugin.so ../
	nsys nvprof python testMaskedSoftmaxPluginOnnx.py

