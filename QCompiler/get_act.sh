export PYTHONPATH=$(pwd)
# CMD=" srun -N 1  --pty --gres=gpu:H100:1 python "
CMD=" srun -N 1  --pty --gres=gpu:4090:1 python "
set -x

base=/home/dataset

model=( $1 )
$CMD examples/smooth_quant_get_act.py  --model-name  ${base}/${model}  \
        --output-path ${PYTHONPATH}/act_scales/${model}.pt \
         --dataset-path ${base}/mixqdata/val.jsonl.zst 

