

srun -N 1  --pty --gres=gpu:4090:1  python examples/test.py --quant fp16


export PYTHONPATH=/home/chenyidong/newstart/QCompiler:$PYTHONPATH
srun -N 1  --pty --gres=gpu:4090:1  python examples/test.py --quant mixq4