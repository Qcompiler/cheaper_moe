#include <pybind11/pybind11.h>

#include<vector>
#include <torch/extension.h>


torch::Tensor mixgemm_sparse_fp16_dense_weight(int M, int N, int K,  int num_ind,
                            torch::Tensor & A_, torch::Tensor &B_,  
                            torch::Tensor & ind_);
torch::Tensor mixgemmforward_dynamic(int M, int N, int K, 
                            torch::Tensor & A_, torch::Tensor &w_, torch::Tensor &s_,
                            int batch, int seq_len,
                            torch::Tensor &fp_w_, torch::Tensor & ind_, int num_ind);

void gemv(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y );
void gemv_int4(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    torch::Tensor scaling
                                   );
void gemv_int4_fp16_mix(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    torch::Tensor scaling,
                                    torch::Tensor weight_cache,
                                    torch::Tensor ind,
                                    int n_outliers
                                   );
void gemv_int4_fp16_mix_sm90(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    torch::Tensor scaling,
                                    torch::Tensor weight_cache,
                                    torch::Tensor ind,
                                    int n_outliers
                                   );
void gemv_int4_fp16_mix_sm90_new(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    torch::Tensor scaling,
                                    torch::Tensor weight_cache,
                                    torch::Tensor ind,
                                    int n_outliers
                                   );
void gemv_int4_sm90(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    torch::Tensor scaling
                                   );
torch::Tensor mixgemmforward_direct(int M, int N, int K, 
                            torch::Tensor & A_, 
                            torch::Tensor & scale_A_,
                            torch::Tensor &w_, torch::Tensor &s_,
                            int batch, int seq_len );

void zero_copy(torch::Tensor & cpu_A,   torch::Tensor & input_A);
torch::Tensor mixgemmforward_direct_with_scaling(int M, int N, int K, 
                            torch::Tensor & A_, 
                            torch::Tensor & A_scaling, 
                            torch::Tensor &w_, 
                            torch::Tensor &s_,
                            int batch, int seq_len );
void int8quant(int M, int K, torch::Tensor & A_,  torch::Tensor &s_,
               torch::Tensor & quant_out,
               torch::Tensor & fp_activation_tmp,
               torch::Tensor & ind_,
               int num_ind);

void find_zeros(torch::Tensor & cpu_A,   torch::Tensor & input_A, int bs, int seq, int hidden, torch::Tensor & last_input);

void reuse_output(torch::Tensor & cpu_A,   torch::Tensor & input_A, int bs, int seq, int hidden, torch::Tensor & last_input);


torch::Tensor FindRowScaleF32(  const torch::Tensor &x,  torch::Tensor &scaleRow,
                         int rows, int cols, int bit) ;

// mixgemm.dequant(q_weight, scales, rows, cols, bit, groupsize = groupsize) 
// n * k 
torch::Tensor dequant(  const torch::Tensor & q_weight,  torch::Tensor &scales,
                         int rows, int cols, int bit, int groupsize,
                          unsigned int block_dim_x,
                                   unsigned int block_dim_y) ;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8quant", &int8quant,
        "int8quant");
    m.def("zero_copy", &zero_copy,
        "zero_copy");
    m.def("find_zeros",&find_zeros,"find_zeros");
    m.def("reuse_output",&reuse_output,"reuse_output");

    m.def("gemv", &gemv,
        "gemv");
    m.def("gemv_int4", &gemv_int4,
        "gemv_int4");
    m.def("gemv_int4_fp16_mix", &gemv_int4_fp16_mix,
        "gemv_int4_fp16_mix");
    m.def("gemv_int4_sm90", &gemv_int4_sm90,
        "gemv_int4_sm90");
    m.def("gemv_int4_fp16_mix_sm90", &gemv_int4_fp16_mix_sm90,
        "gemv_int4_fp16_mix_sm90");

    m.def("gemv_int4_fp16_mix_sm90_new", &gemv_int4_fp16_mix_sm90_new,
        "gemv_int4_fp16_mix_sm90_new");

    m.def("mixgemmforward_direct", &mixgemmforward_direct,
        "mixgemmforward_direct");        
    m.def("mixgemmforward_dynamic", &mixgemmforward_dynamic,
        "mixgemmforward_dynamic");  
    m.def("mixgemm_sparse_fp16_dense_weight", &mixgemm_sparse_fp16_dense_weight,
        "mixgemm_sparse_fp16_dense_weight");  
    m.def("mixgemmforward_direct_with_scaling", &mixgemmforward_direct_with_scaling,
        "mixgemmforward_direct_with_scaling");  
        
    m.def("FindRowScaleF32", &FindRowScaleF32,
        "FindRowScaleF32");  
        
    // mixgemm.dequant(q_weight, scales, bit, groupsize = groupsize)

     m.def("dequant", &dequant,
        "dequant");     
}