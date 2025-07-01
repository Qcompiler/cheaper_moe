VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
--model /home/dataset/Llama-3-8B  --served-model-name eval_math --trust_remote_code --port 8801



curl http://0.0.0.0:8801/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "eval_math",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'