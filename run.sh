

srun -N 1  --pty --gres=gpu:4090:1  python examples/test.py --quant fp16


export PYTHONPATH=/home/chenyidong/cheaper_moe/QCompiler:$PYTHONPATH
srun -N 1  --pty --gres=gpu:4090:1  python examples/test.py --quant mixq4


srun -N 1  --pty --gres=gpu:4090:1  python examples/test_dense.py --quant mixq4