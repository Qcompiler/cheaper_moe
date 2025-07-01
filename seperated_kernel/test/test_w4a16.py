import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import time

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def w4a16_matmul_kernel(
    # 指向矩阵的指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 步长
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """W4A16矩阵乘法核函数，其中权重(W)是4-bit，激活(A)是16-bit"""
    
    # 程序ID
    pid = tl.program_id(axis=0)
    
    # 计算这个程序应该处理的C矩阵块
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 创建块的起始坐标
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # A和B矩阵的内存指针
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] // 8 * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 累加器初始化为0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载A矩阵的块(16-bit激活)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 加载B矩阵的块(4-bit权重)
        b_packed = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 解包4-bit权重
        b = (b_packed >> ((offs_k[:, None] % 8) * 4)) & 0xF
        
        # 转换为有符号整数
        b = b - 8  # 假设使用zero_point=8的有符号4-bit
        
        # 矩阵乘法
        accumulator += tl.dot(a, b)
        
        # 移动指针到下一个K块
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk
    
    # 将结果转换为fp16
    c = accumulator.to(tl.float16)
    
    # 定义C矩阵的内存块指针
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # 存储结果
    tl.store(c_ptrs, c, mask=c_mask)

def w4a16_matmul(a, b):
    # 检查维度
    assert a.shape[1] == b.shape[0] * 8, "维度不匹配"
    assert a.is_cuda and b.is_cuda, "矩阵必须在GPU上"
    
    M, K = a.shape
    _, N = b.shape
    
    # 分配输出矩阵
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # 1D启动内核
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    w4a16_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c

def pack_4bit(w):
    """将权重打包为4-bit"""
    assert w.dtype == torch.int8, "权重必须是int8"
    assert w.shape[1] % 8 == 0, "K维度必须是8的倍数"
    
    # 确保值在[-8,7]范围内
    w = torch.clamp(w, -8, 7)
    
    # 转换为无符号4-bit [0,15]
    w = (w + 8).to(torch.uint8)
    
    # 打包为4-bit (每两个int8打包为一个uint8)
    w_packed = torch.zeros((w.shape[0], w.shape[1] // 8), dtype=torch.uint8, device=w.device)
    
    for i in range(8):
        w_packed |= (w[:, i::8] << (4 * i))
    
    return w_packed
def benchmark_matmul(M, N, K, dtype=torch.float16):
    # 创建随机矩阵
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    
    # FP16基准
    def fp16_matmul():
        return torch.matmul(a, b)
    
    # W4A16基准
    def w4a16_matmul_bench():
        # 创建4-bit权重 (模拟量化)
        b_int8 = torch.randint(-8, 8, (K, N), device='cuda', dtype=torch.int8)
        b_packed = pack_4bit(b_int8)

        print(b_packed.shape, b_packed.dtype)
        return w4a16_matmul(a, b_packed)
    
    # 预热
    for _ in range(10):
        fp16_matmul()
        w4a16_matmul_bench()
    
    # 测量FP16性能
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        fp16_matmul()
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / 100
    
    # 测量W4A16性能
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        w4a16_matmul_bench()
    torch.cuda.synchronize()
    w4a16_time = (time.time() - start) / 100
    
    return fp16_time, w4a16_time

def run_benchmarks():
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    
    fp16_times = []
    w4a16_times = []
    speedups = []
    
    for M, N, K in sizes:
        fp16_time, w4a16_time = benchmark_matmul(M, N, K)
        fp16_times.append(fp16_time)
        w4a16_times.append(w4a16_time)
        speedup = fp16_time / w4a16_time
        speedups.append(speedup)
        
        print(f"Size: {M}x{N}x{K}")
        print(f"FP16 time: {fp16_time:.6f}s")
        print(f"W4A16 time: {w4a16_time:.6f}s")
        print(f"Speedup: {speedup:.2f}x")
        print("-" * 40)
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    x = [f"{s[0]}x{s[1]}x{s[2]}" for s in sizes]
    plt.plot(x, fp16_times, label='FP16', marker='o')
    plt.plot(x, w4a16_times, label='W4A16', marker='o')
    plt.xlabel('Matrix Size (MxNxK)')
    plt.ylabel('Execution Time (s)')
    plt.title('FP16 vs W4A16 Matrix Multiplication Performance')
    plt.legend()
    plt.grid()
    plt.show()
    
    # 绘制加速比
    plt.figure(figsize=(10, 6))
    plt.bar(x, speedups)
    plt.xlabel('Matrix Size (MxNxK)')
    plt.ylabel('Speedup (x)')
    plt.title('W4A16 Speedup over FP16')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_benchmarks()