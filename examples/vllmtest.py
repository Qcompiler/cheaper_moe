from vllm import LLM, SamplingParams


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

model = "/home/dataset/"+ "DeepSeek-V2-Chat-0628"

if 'awq' in model.lower():
    print("run awq model")
    llm = LLM(model=model, quantization='awq', trust_remote_code=True)
elif 'gptq' in model.lower():
    llm = LLM(model=model, quantization='gptq', trust_remote_code=True)
else:
    llm = LLM(model=model,  trust_remote_code=True)
# 
# llm = LLM(model="/home/dataset/Deepseek-V2-Lite-GPTQ2", quantization = 'gptq', trust_remote_code=True)
# modelscope download --model TechxGenus-MS/DeepSeek-V2-Lite-Chat-AWQ
# modelscope download --model OPEA/DeepSeek-V2-Lite-int4-sym-inc
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")