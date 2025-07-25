
# eval_gsm8k.py
from mmengine.config import read_base

with read_base():
    # Select a dataset list
    from .datasets.gsm8k.gsm8k_0shot_gen_a58960 import gsm8k_datasets as datasets
    # Select an interested model
    from .models.hf_llama.hf_llama3_8b_instruct import models
    
from opencompass.models import HuggingFacewithChatTemplate
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf',
        path='/home/dataset/Meta-Llama-3.1-8B-Instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        model_kwargs=dict(
            trust_remote_code=True,
        ),
    )
]
