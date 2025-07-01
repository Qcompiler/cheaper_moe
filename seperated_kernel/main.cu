

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include<cuda_fp16.h>

 
#include <cuda_runtime.h>
 



template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using FragB = Vec<half2, 2>;


template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}


__device__ inline FragB dequant_no_zero_point(int q) {
  const int LO = 0x000f000f; // Mask for lower 4 bits
  const int HI = 0x00f000f0; // Mask for higher 4 bits
  const int EX = 0x64006400; // Extended constant for LOP3
  const int MUL = 0x2c002c00; // Scaling multiplier constant
  const int ADD = 0xd480d480;
  const int SUB = 0x64086408;

  // Unpack low and high 4-bit integers
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  // Perform scaling
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );

  return frag_b;
}

__global__ void test(){

      int b_quant = -128; 
      int b_quant_shift = b_quant >> 8;
      FragB frag_b0 = dequant_no_zero_point(b_quant);
      FragB frag_b1 = dequant_no_zero_point(b_quant_shift);

       half2* mat_h1 = (half2*)&frag_b0[0];
       half2* mat_h2 = (half2*)&frag_b0[1];
       half2* mat_h3 = (half2*)&frag_b0[0];
       half2* mat_h4 = (half2*)&frag_b1[1];

        printf("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t",
        __half2float(frag_b0[0].x),
        __half2float(frag_b0[0].y),
        __half2float(frag_b1[0].x),
        __half2float(frag_b1[0].y),
        __half2float(frag_b0[1].x),
        __half2float(frag_b1[1].x),
        
        __half2float(frag_b0[1].y),
        
        __half2float(frag_b1[1].y));

}
int main(){

    test<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}