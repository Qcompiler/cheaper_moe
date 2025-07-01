#pragma once
#include "cutlass/numeric_types.h"

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.001;
    data[i] = (T)v;
  }
}

template <>
void gen_rand_data<cutlass::bfloat16_t>(cutlass::bfloat16_t *data, int n) {
    for (int i = 0; i < n; ++i) {
        float v = (rand() % 200 - 100) * 0.001;
        data[i] = cutlass::bfloat16_t(v);
    }
}

template <typename T>
T quantize(float value, float amax) {
    CUTLASS_NOT_IMPLEMENTED();
}

template <typename T>
float dequantize(T value, float scale) {
    CUTLASS_NOT_IMPLEMENTED();
}

template <typename T>
float rescale(float amax) {
    CUTLASS_NOT_IMPLEMENTED();
}

template <typename T_S>
T_S from_float(float value) {
    return static_cast<T_S>(value);
}

template <>
int8_t quantize<int8_t>(float value, float amax) {
    return static_cast<int8_t>(roundf32(value / amax * 127));
}

template <>
cutlass::float_e4m3_t quantize<cutlass::float_e4m3_t>(float value, float amax) {
    return cutlass::float_e4m3_t::from_float(value / amax * 448);
}

template <>
cutlass::float_e5m2_t quantize<cutlass::float_e5m2_t>(float value, float amax) {
    return cutlass::float_e5m2_t::from_float(value / amax * 57344);
}

template <>
float dequantize<int8_t>(int8_t value, float scale) {
    return static_cast<float>(value) * scale;
}

template <>
float dequantize<cutlass::float_e4m3_t>(cutlass::float_e4m3_t value, float scale) {
    return cutlass::float_e4m3_t::to_float(value) * scale;
}

template <>
float dequantize<cutlass::float_e5m2_t>(cutlass::float_e5m2_t value, float scale) {
    return cutlass::float_e5m2_t::to_float(value) * scale;
}

template <>
float rescale<int8_t>(float amax) {
    return amax / 127.0;
}

template <>
float rescale<cutlass::float_e4m3_t>(float amax) {
    return amax / 448.0;
}

template <>
float rescale<cutlass::float_e5m2_t>(float amax) {
    return amax / 57344.0;
}

template <>
cutlass::half_t from_float<cutlass::half_t>(float value) {
    return cutlass::half_t(value);
}

template <>
cutlass::bfloat16_t from_float<cutlass::bfloat16_t>(float value) {
    return cutlass::bfloat16_t(value);
}

template <typename T, typename T_S>
inline void host_quantization(void *Iptr, void *Optr, void *O_Sptr, int m, int n, int col_block_size, int row_block_size) {
    T_S *data = (T_S*)Iptr;
    T *out = (T*)Optr;
    float *Sout = (float*)O_Sptr;
    assert(m % col_block_size == 0);
    assert(n % row_block_size == 0);

    int col_block_num = m / col_block_size;
    int row_block_num = n / row_block_size;

    for (int col_block_idx = 0; col_block_idx < col_block_num; ++col_block_idx)
        for (int row_block_idx = 0; row_block_idx < row_block_num; ++row_block_idx) {
            float amax = 1e-8;
            for (int i = 0; i < col_block_size; ++i)
                for (int j = 0; j < row_block_size; ++j) {
                    int m_idx = col_block_idx * col_block_size + i;
                    int n_idx = row_block_idx * row_block_size + j;
                    amax = std::max(
                        amax,
                        fabsf32(
                            static_cast<float>(
                                *(data + m_idx * n + n_idx)
                            )
                        )
                    );
                }
            for (int i = 0; i < col_block_size; ++i)
                for (int j = 0; j < row_block_size; ++j) {
                    int m_idx = col_block_idx * col_block_size + i;
                    int n_idx = row_block_idx * row_block_size + j;
                    float value = static_cast<float>(*(data + m_idx * n + n_idx));
                    *(out + m_idx * n + n_idx) = quantize<T>(value, amax);
                }
            *(Sout + col_block_idx * row_block_num + row_block_idx) = rescale<T>(amax);
        }
}

template <typename T, typename T_S>
inline void host_dequantization(void *Iptr, void *I_Sptr, void *Optr, int m, int n, int col_block_size, int row_block_size) {
    assert (m % col_block_size == 0);
    assert (n % row_block_size == 0);
    T_S *out = (T_S*)Optr;
    T *data = (T*)Iptr;
    float *Sdata = (float*)I_Sptr;
    // int col_block_num = m / col_block_size;
    int row_block_num = n / row_block_size;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            int scale_col_idx = i / col_block_size;
            int scale_row_idx = j / row_block_size;
            float scale = *(Sdata + scale_col_idx * row_block_num + scale_row_idx);
            *(out + i * n + j) = from_float<T_S>(dequantize<T>(
                *(data + i * n + j),
                scale
            ));
        }
}

template <typename T>
inline void inplace_transpose(void *Iptr, int m, int n) {
    T *data = (T*)Iptr;
    T *tmp = (T*)malloc(sizeof(T) * m * n);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            *(tmp + j * m + i) = *(data + i * n + j);
        }
    
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            *(data + i * m + j) = *(tmp + i * m + j);
        }
    
    free((void*)tmp);
}
