```
source load.sh
conda create python==3.11 -n moe
```

## vllm 可以帮助你安装几乎所有的依赖

```
pip install vllm==0.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

cd QCompiler/EETQ
python setup.py install

cd QCompiler/quantkernel
python setup.py install


cd seperated_kernel
mkdir build
cd build
cmake ..
make -j
cd ..
python setup.py install 


cd seperated_kernel/gemm_opt
python setup.py install 

pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.51  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash_attn
```

## 量化模型

cd QCompiler
bash get_act.sh Qwen2.5-7B-Instruct
bash quant.sh Qwen2.5-7B-Instruct  4

## 运行实例

run.sh


## 下一步工作
w3a8 outliers
修改方法
w2a8 outliers
