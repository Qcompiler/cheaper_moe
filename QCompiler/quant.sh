# CMD=" srun -N 1  --pty --gres=gpu:H100:1 python "
CMD=" srun -N 1  --pty --gres=gpu:4090:1 python "
set -x

export PYTHONPATH=/home/chenyidong/newstart/QComplier
models=( $1 )
path=/home/dataset
for bit in   $2
  do
    for model in "${models[@]}"
            do
            echo ${model}
            ${CMD} \
              examples/basic_quant_mix.py  \
            --model_path ${path}/${model} \
            --quant_file ${path}/quant${bit}/${model} --w_bit ${bit}
    done
done


