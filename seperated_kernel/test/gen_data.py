import torch
import numpy as np
import torch.nn as nn
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')

def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()
class Layer(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1, tile = 16):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError('`infeatures` must be divisible by 128 and `outfeatures` by 256.')
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError('`infeatures` must be divisible by `groupsize`.')
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer('B', torch.empty((self.k // tile, self.n * tile // 8), dtype=torch.int))
        self.register_buffer('s', torch.empty((self.k // groupsize, self.n), dtype=torch.half))
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        self.register_buffer('workspace', torch.zeros(self.n // 128 * 16, dtype=torch.int), persistent=False)

    def forward(self, A):
        C = torch.empty(A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device)
        mul(A.view((-1, A.shape[-1])), self.B, C.view((-1, C.shape[-1])), self.s, self.workspace)
        return C

    def pack(self, linear, scales, tile):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """ 
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')
        maxq = 2 ** 4 - 1
        s = scales.t()
        w = linear.weight.data.t()
        #print(w.shape)
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        
        # print(w.shape)
        res = res.cpu().numpy().astype(np.uint32)
        # print(res.shape)

        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        # print(q.shape)
        # exit()
        # print("target shape")
        # print(self.B.shape)
        # exit()
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)



class MyLayer(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1, tile = 1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        # if groupsize not in [-1, 128, outfeatures]:
        #     raise ValueError('Only groupsize -1 and 128 are supported.')
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError('`infeatures` must be divisible by 128 and `outfeatures` by 256.')
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError('`infeatures` must be divisible by `groupsize`.')
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer('B', torch.empty((self.n * tile // 8 , self.k // tile), dtype=torch.int))
        self.register_buffer('s', torch.empty((self.k // groupsize, self.n), dtype=torch.half))

    def pack(self, linear, scales, tile):

        
        
        k = self.k
        
        interleave = []
        for i in range(k//8):
            out = [0, 2, 4, 6, 1, 3, 5, 7]
            for j in range(8):
                out[j] = out[j] + 8 * i
            interleave += out
        interleave = np.array(interleave)
        w = linear
        
        res = w[:,interleave]
      

      
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
   
        res = res.cpu().numpy().astype(np.uint32)

        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)

        self.B[:, :] = q.to(self.B.device)
        # self.s[:, :] = s.to(self.s.device)





def gen_quant4_my(n, k, w, groupsize=-1,  tile = 1):
    if groupsize == -1:
        groupsize = k   
    maxq = 2 ** 4 - 1  # 4-bit量化，最大值15
    n, k = w.shape  # 原始权重矩阵形状

    # 计算需要的组数（向上取整）
    num_groups = (k + (groupsize-1)) // groupsize  # 等价于 math.ceil(k / 128)

    # 填充权重矩阵，使k能被128整除
    padded_k = num_groups * groupsize
    if k % groupsize != 0:
        w_padded = torch.nn.functional.pad(w, (0, padded_k - k))
    else:
        w_padded = w

    # 将权重矩阵重塑为 (n, num_groups, 128)
    w_reshaped = w_padded.reshape(n, num_groups, groupsize)
    # 计算每个组的缩放因子s (n, num_groups, 1)
    s = torch.max(torch.abs(w_reshaped), dim=2, keepdim=True)[0]
    s *= 2 / maxq  # 缩放因子范围调整
    # 量化过程
    linear = torch.clone(w_reshaped)
    linear = torch.round(linear / s).int()
    linear += (maxq + 1) // 2  # 添加零点偏移
    linear = torch.clamp(linear, 0, maxq)
    # 将量化的权重和缩放因子重塑回原始形状
    linear = linear.reshape(n, -1)[:, :k]  # 移除填充并恢复原始k维度
    s = s.reshape(n, -1).contiguous()  # 缩放因子形状为 (n, num_groups)  

    # Workaround to test some special cases that are forbidden by the API
    layer = MyLayer(k, n, groupsize=groupsize, tile = tile)

    layer.k = k
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((n // tile , k  * tile // 8), dtype=torch.int, device=DEV)
    layer.pack(linear, s.t(), tile = tile)
    q = layer.B



    return q, s

maxq = 2 ** 4 - 1
def gen_quant4_my_reshape(n, k, w, groupsize=-1,  tile = 1):
   
    
    #print(w)
    s = torch.max(torch.abs(w), 1, keepdim=True)[0]
    s *= 2 / maxq
 
  
    #print(torch.max(torch.abs(ref), 0, keepdim=True)[0])
    #exit()

    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n).to(torch.float16).cuda()
    linear.weight.data.copy_(w)

    
    # Workaround to test some special cases that are forbidden by the API
    layer = MyLayer(256, 256, groupsize=groupsize, tile = tile)
    if groupsize == -1:
        groupsize = k
    layer.k = k
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((n // tile , k  * tile // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((k // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack_rehshape_2(linear, s.t(), tile = tile)
    q = layer.B

    s = layer.s
    return None, q, s


def gen_quant4(m, n, w, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // tile, n * tile // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t(), tile)
    q = layer.B
    s = layer.s
    return ref, q, s


def reshape_activation(weight_cache):
    w = torch.clone(weight_cache)
    n, k = w.shape
    res = w.reshape(-1, 1)

    interleave = []
    for i in range( int(n * k / ( 1 * 2 * k) ) ):
        out = []
        for j in range(0, k):
            out.append(j +     i * 2 * k)
            out.append(j + k + i * 2 * k)

        interleave += out
    interleave = np.array(interleave)
    res = res[interleave, :].contiguous()
    res = res.reshape(n, k) 
    return res     


def reshape_activation_new(weight_cache):
    w = torch.clone(weight_cache)
    n, k = w.shape
    res = w.reshape(-1, 1)

    interleave = []
    for i in range( int(n * k / ( 1 * 4 * k) ) ):
        out = []
        for j in range(0, k):
            out.append(j +         i * 4 * k)
            out.append(j +     k + i * 4 * k)
            out.append(j + 2 * k + i * 4 * k)
            out.append(j + 3 * k + i * 4 * k)

        interleave += out
    interleave = np.array(interleave)
    res = res[interleave, :].contiguous()
    res = res.reshape(n, k) 
    return res     



def reshape_weight(weight_cache):
    w = torch.clone(weight_cache)
    n, k = w.shape
    res = w.reshape(-1, 2)

    interleave = []
    for i in range( int(n   / (  2 ) ) ):
        out = []
        for j in range(0, int( k / 2 ) ):
            out.append(     j +          i *  k)
            out.append( int (j + k / 2 + i *  k) )

        interleave += out
    interleave = np.array(interleave)
    res = res[interleave, :].contiguous()
    res = res.reshape(n, k) 
    return res   


def reshape_weight_new(weight_cache):
    w = torch.clone(weight_cache)
    n, k = w.shape
    res = w.reshape(-1, 2)  # 转换为half2格式
    
    interleave = []
    for group in range(int(n / 4)):  # 4行一组处理
        for pos in range(int(k / 2)):  # 每次处理1个half2(2个half)
            # 按顺序处理组内4行的相同位置
            for row in range(4):
                original_row = group * 4 + row
                half2_pos = pos + original_row * (k // 2)
                interleave.append(half2_pos)
    
    interleave = np.array(interleave)
    res = res[interleave, :].contiguous()
    res = res.reshape(n, k)
    return res

    w = torch.clone(weight_cache)
    n, k = w.shape
    res = w.reshape(-1, 2)


    interleave = []
    for i in range( int(n   / (  4 ) ) ):
        out = []
        for j in range(0, int( k / 2 ) ):
            out.append(      j +             i *  k)
            out.append( int (j +     k / 2 + i *  k) )
            out.append( int (j + 2 * k / 2 + i *  k) )
            out.append( int (j + 3 * k / 2 + i *  k) )

        interleave += out
    interleave = np.array(interleave)
    res = res[interleave, :].contiguous()
    res = res.reshape(n, k) 
    return res     


