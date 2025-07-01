CC_FILES=$(shell find ./ -name "*.cu")
EXE_FILES=$(CC_FILES:.cu=)

all:$(EXE_FILES)

%:%.cu
	nvcc -o $@ $< -O3 -gencode arch=compute_90a,code=sm_90a -gencode arch=compute_89,code=sm_89  -std=c++17 -I/home/chenyidong/gemm_opt/csrc -I/home/chenyidong/gemm_opt/cutlass/include -Xcompiler -fdiagnostics-show-template-tree --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt -lcublas -lineinfo -DNDEBUG --expt-relaxed-constexpr -DCUTE_USE_PACKED_TUPLE=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -lineinfo

clean:
	rm -rf $(EXE_FILES)


