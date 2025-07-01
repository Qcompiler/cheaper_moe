import torch
import mixgemm

a = torch.zeros((1,1), dtype = torch.int32, pin_memory = True)

b = torch.ones((1,1), device = "cuda", dtype = torch.float16)
mixgemm.zero_copy(a, b)

print(a)


b = torch.ones((1,1), device = "cuda", dtype = torch.float16) * 10
mixgemm.zero_copy(a, b)

print(a)

# nice job! it worked!
# srun -N 1 --gres=gpu:4090:1 python  test_pinn.py
# tensor([[0]], dtype=torch.int32)
# tensor([[1]], dtype=torch.int32)