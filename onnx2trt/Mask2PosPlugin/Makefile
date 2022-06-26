include ../Makefile.inc

SRC_CU          = $(shell find ./ -name '*.cu')

all: Mask2PosPlugin.so

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

Mask2PosPlugin.so: $(SRC_CU:.cu=.o)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: clean
clean:
	rm -rf ./*.so ./*.o ./*.d ./*.trt ./temp* ./*.nsys-rep ./*.sqlite

.PHONY: test
test:
	clear
	make clean
	make all
	cp ./Mask2PosPlugin.so ../
	python testMask2PosPluginOnnx.py

