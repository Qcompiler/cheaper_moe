from opencompass.models import HuggingFacewithChatTemplate

import torch
from mixquant.modules.linear import MixLinear_GEMM
torch.nn.Linear = MixLinear_GEMM

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepseek-v2-lite-chat-hf-MixQ',
        path='/home/dataset/quant4/DeepSeek-V2-Lite-Chat',
        max_out_len=4096,
        batch_size=32,
        model_kwargs=dict(
             fuse_layers=True,
             mix = True,
        ),
        run_cfg=dict(num_gpus=1),
    )
]
