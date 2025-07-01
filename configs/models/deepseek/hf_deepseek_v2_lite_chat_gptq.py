from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepseek-v2-lite-chat-hf-gptq',
        path='/home/dataset/DeepSeek-V2-Lite-GPTQ',
        max_out_len=4096,
        batch_size=32,
        model_kwargs=dict(
            torch_dtype="torch.bfloat16",
        ),
        run_cfg=dict(num_gpus=1),
    )
]
