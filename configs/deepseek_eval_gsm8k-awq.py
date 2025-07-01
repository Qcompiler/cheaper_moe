
# eval_gsm8k.py
from mmengine.config import read_base

with read_base():
    # Select a dataset list
    from .datasets.gsm8k.gsm8k_0shot_gen_a58960 import gsm8k_datasets as datasets
    
from opencompass.models import HuggingFacewithChatTemplate
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='DeepSeek-V2-Lite-Chat-AWQ',
        path='/home/dataset/DeepSeek-V2-Lite-Chat-AWQ',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]
