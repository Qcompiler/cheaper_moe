import torch
import mixgemm


M = 16
N = 8
K = 16
for K in [8, 16]:
    n_outliers = 8
    a = torch.randn(M, K, dtype = torch.float16).cuda()
    b = torch.randn(N, K, dtype = torch.float16).cuda()
    ind = torch.ones((n_outliers, 1), dtype = torch.int32).cuda()


    # torch::Tensor mixgemm_sparse_fp16_dense_weight(int M, int N, int K,  int num_ind,
    #                             torch::Tensor & A_, torch::Tensor &B_,  
    #                             torch::Tensor & ind_){
    out = mixgemm.mixgemm_sparse_fp16_dense_weight(M, N, K, n_outliers, a, b, ind)

    grand = torch.mm(a, b.T)
    # print(a[0:16,0:2])
    # print(b[0:2,:])
    # print(grand)
    # print(out)
    print((grand - out).abs().max())
    assert ((grand - out).abs().max() <= 0.01)