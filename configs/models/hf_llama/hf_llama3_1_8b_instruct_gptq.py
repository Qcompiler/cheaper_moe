from opencompass.models import HuggingFacewithChatTemplate

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3_1-8b-instruct-hf-GPTQ-INT4',
        path='/home/dataset/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
        model_kwargs=dict(
            trust_remote_code=True,
            quantization = 'gptq',
        ),
    )
]
