First of all, build the seperated so file by
```
mkdir build
cd build
cmake ..
make
```

After that, build the python bindings by:
```
python setup.py install
```

export the library path by

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PATH_OF_ROOT}/build/kernel
```

