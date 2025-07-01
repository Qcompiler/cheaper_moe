
#include <cub/cub.cuh>

#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>


template <typename InputType, typename OutputType, int32_t TILE_M, int32_t TILE_N, int32_t BLOCK_SIZE>
__global__ void cudaCoreGemm(InputType const* __restrict__ act, InputType const* __restrict__ weight, float alpha,
    OutputType* __restrict__ output, int32_t m, int32_t n, int32_t k)
{
    using VecType = int4;
    static constexpr int32_t kStepK = static_cast<int32_t>(128 / (8 * sizeof(InputType)));
    static constexpr int32_t kTileK = kStepK * BLOCK_SIZE;
    auto tileIdM = static_cast<int32_t>(blockIdx.x * TILE_M);
    auto tileIdN = static_cast<int32_t>(blockIdx.y * TILE_N);
    auto tid = static_cast<int32_t>(threadIdx.x);
    float tile_a[kStepK], tile_w[TILE_N * kStepK];
    float acc[TILE_M * TILE_N];

    static_assert(kStepK % 4 == 0);
 
    using Converter = cutlass::NumericArrayConverter<float, cutlass::uint4b_t, 4>;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;
    static constexpr int32_t kCvtCount = static_cast<int32_t>(sizeof(VecType) / sizeof(CvtSrcType));

#pragma unroll
    for (int32_t i = 0; i < TILE_M * TILE_N; ++i)
    {
        acc[i] = 0;
    }
    act += tileIdM * k;
    weight += tileIdN * k;
    output += tileIdM * n + tileIdN;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    for (int32_t idxK = tid * kStepK; idxK < k; idxK += kTileK)
    {
        for (int32_t i = 0; i < TILE_N; ++i)
        {
            auto tile_w_quantized = reinterpret_cast<VecType const*>(weight + i * k + idxK)[0];
#pragma unroll
            for (int32_t cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx)
            {
                reinterpret_cast<CvtResType*>(tile_w)[i * kCvtCount + cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_w_quantized)[cvtIdx]);
            }
        }
#pragma unroll
        for (int32_t i = 0; i < TILE_M; ++i)
        {
            auto tile_a_quantized = reinterpret_cast<VecType const*>(act + i * k + idxK)[0];
#pragma unroll
            for (int32_t cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx)
            {
                reinterpret_cast<CvtResType*>(tile_a)[cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_a_quantized)[cvtIdx]);
            }
#pragma unroll
            for (int32_t j = 0; j < TILE_N; ++j)
            {
#pragma unroll
                for (int32_t l = 0; l < kStepK; ++l)
                {
                    acc[i * TILE_N + j] = fma(tile_a[l], tile_w[j * kStepK + l], acc[i * TILE_N + j]);
                }
            }
        }
    }

    typedef cub::WarpReduce<float> WarpReduce;

    static constexpr int32_t kWarpSize = 32;
    static constexpr int32_t kWarpNum = BLOCK_SIZE / kWarpSize;
    int32_t warpId = tid / kWarpSize, laneId = tid % kWarpSize;
    __shared__ float shmem[TILE_M * TILE_N * kWarpNum];
    __shared__ typename WarpReduce::TempStorage tempStorage[kWarpNum];
#pragma unroll
    for (int32_t mi = 0; mi < TILE_M; ++mi)
    {
#pragma unroll
        for (int32_t ni = 0; ni < TILE_N; ++ni)
        {
            float val = WarpReduce(tempStorage[warpId]).Sum(acc[mi * TILE_N + ni]);
            if (laneId == 0)
            {
                shmem[mi * TILE_N + ni + warpId * TILE_M * TILE_N] = val;
            }
        }
    }
    __syncthreads();
    for (int32_t ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE)
    {
        int32_t mid = ii / TILE_N, nid = ii % TILE_N;
        float val = 0;
#pragma unroll
        for (int32_t jj = 0; jj < kWarpNum; ++jj)
        {
            val += shmem[jj * TILE_M * TILE_N + ii];
        }
        output[mid * n + nid] = static_cast<OutputType>(val * alpha);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}