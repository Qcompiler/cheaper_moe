import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig



import argparse

parser = argparse.ArgumentParser(description='这是一个示例程序，用于演示 argparse 的基本用法')

parser.add_argument('--quant', type = str,  help='quant')
args = parser.parse_args()

quant = args.quant

if quant == 'gptq':
    model_name = "/home/dataset/DeepSeek-V2-Lite-gptq"

if quant == 'awq':
    model_name = "/home/dataset/DeepSeek-V2-Lite-Chat-AWQ"

if quant == 'fp16':
    model_name = "/home/dataset/DeepSeek-V2-Lite-Chat"


if quant == 'mixq4':
    model_name = "/home/dataset/quant4/DeepSeek-V2-Lite-Chat"

# model_name = "/home/dataset/quant4/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if "gptq" in model_name.lower():

    from gptqmodel import GPTQModel
    model = GPTQModel.load(model_name,  trust_remote_code=True)
else:
    if "quant4" in model_name.lower():
        print("quant4 loading")
        # exit()
        from mixquant import AutoForCausalLM

        model = AutoForCausalLM.from_quantized(
            model_name, model_name, fuse_layers=True,
            mix = True
        )
        # model = AutoForCausalLM.from_pretrained(model_name, 
        #                                              trust_remote_code=True, 
        #                                              device_map = "auto")
        model = model.to("cuda")

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map = "auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.max_new_tokens = 1024
messages = [
    {"role": "user", "content": "Write a piece of quicksort code in C++"}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=32)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)