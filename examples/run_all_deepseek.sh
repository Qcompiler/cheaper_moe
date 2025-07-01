set -ex

export DATASET_SOURCE=ModelScope

# rm -rf output/default/*
export HF_ENDPOINT=https://hf-mirror.com

datasets=(
    # math_gen_5e8458
    
    # lambada_gen_217e11
    # hellaswag_gen_6faab5
    
    # ceval_gen_2daf24
    # race_gen_69ee4f  
    # winogrande_gen_458220
    # piqa_gen_1194eb
    # wikitext_2_raw_ppl_752e2a
    # olymmath_llm_judeg_gen
    gsm8k_0shot_gen_a58960
    internal_humaneval_gen_d2537e

    # mmlu_gen_23a9a9
    # olymmath_llmverify_gen_97b203
 
    
)

for dataset in "${datasets[@]}"; do
   
    # sleep 10
    # srun -N 1  --pty --gres=gpu:H100:1  -p Long opencompass  --work-dir /home/chenyidong/output/output/deepseek_output  \
    # --datasets   ${dataset} --models  hf_deepseek_v2_lite_chat   &
    # # sleep 10
    srun -N 1  --pty --gres=gpu:H100:1  -p Long opencompass  --work-dir /home/chenyidong/output/output/deepseek_output  \
    --datasets   ${dataset} --models  hf_deepseek_v2_lite_chat_awq &
    # # sleep 10
    # # srun -N 1  --pty --gres=gpu:H100:1  -p Long opencompass  --work-dir qwen_output  \
    #  --datasets   ${dataset} --models  hf_qwen_2_5_7b_instruct_gptq
    # srun -N 1 --export=ALL --pty --gres=gpu:H100:1  -p Long opencompass    --work-dir /home/chenyidong/output/output/deepseek_output \
    #  --datasets   ${dataset} --models  hf_deepseek_v2_lite_chat_mixq &


done