set -ex

export DATASET_SOURCE=ModelScope

rm -rf output/default/*
export HF_ENDPOINT=https://hf-mirror.com

datasets=(
    math_gen_5e8458
    
    lambada_gen_217e11
    hellaswag_gen_6faab5
    
    # ceval_gen_2daf24
    # race_gen_69ee4f
    # winogrande_gen_458220
    # piqa_gen_1194eb
    # wikitext_2_raw_ppl_752e2a
    # # olymmath_llm_judeg_gen
    # gsm8k_0shot_gen_a58960
    # internal_humaneval_gen_d2537e

    # mmlu_gen_23a9a9
    # olymmath_llmverify_gen_97b203
    
)

for dataset in "${datasets[@]}"; do
   

    # srun -N 1  --pty --gres=gpu:H100:1  -p Long opencompass    --datasets   ${dataset} --models  hf_llama3_1_8b_instruct  -a vllm
    # srun -N 1  --pty --gres=gpu:H100:1  -p Long opencompass    --datasets   ${dataset} --models  hf_llama3_1_8b_instruct_gptq  -a vllm
    # srun -N 1  --pty --gres=gpu:H100:1  -p Long opencompass    --datasets   ${dataset} --models  hf_llama3_1_8b_instruct_awq  -a vllm
    srun -N 1  --pty --gres=gpu:4090:1  -p Long opencompass  \
      --datasets   ${dataset} --models  hf_llama3_1_8b_instruct_mixq  -a vllm &

    sleep 10
done