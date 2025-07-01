
# eval_gsm8k.py
from mmengine.config import read_base

with read_base():
    # Select a dataset list
    from .datasets.gsm8k.gsm8k_0shot_gen_a58960 import gsm8k_datasets as datasets
    # Select an interested model
    # from .models.hf_llama.hf_llama3_8b_instruct import models
    
from opencompass.models import HuggingFacewithChatTemplate
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Qwen2.5-7B-Instruct-AWQ',
        path='/home/dataset/Qwen2.5-7B-Instruct-AWQ',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        model_kwargs=dict(
            trust_remote_code=True,
            quantization = 'awq',
        ),
    )
]
