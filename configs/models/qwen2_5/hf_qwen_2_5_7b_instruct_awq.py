from opencompass.models import HuggingFaceBaseModel


from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Qwen2.5-7B-Instruct-AWQ',
        path='/home/dataset/Qwen2.5-7B-Instruct-AWQ',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(
            max_model_len = 10240 * 2,
            trust_remote_code=True,
            quantization = 'awq',
        ),
    )
]
