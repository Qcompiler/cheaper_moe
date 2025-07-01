import sys

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.

max_tokens = 32
sampling_params = SamplingParams(temperature=0.8, top_p=0.95 , 
                                 max_tokens=max_tokens, min_tokens=max_tokens)

# Create an LLM.
import os

# please use Qcompiler to quant model !

#llm = LLM(model="/dev/shm/tmp/Llama-2-7b")
pwd="/home/dataset/quant4/"
# pwd ="/home/dataset/"

# model = 'Qwen2.5-7B-Instruct'
model = 'DeepSeek-V2-Lite-Chat'
llm = LLM(model=pwd + model ,gpu_memory_utilization = 0.95, 
          max_model_len = 10240, trust_remote_code=True, 
          quantization = "mixq4bit")
# llm = LLM(model=pwd + model ,gpu_memory_utilization = 0.95, 
#           max_model_len = 10240, trust_remote_code=True, 
#           quantization = None)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")