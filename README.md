# YSDA-CPU-inference
### Quantized inference on CPU (int8 / int4 / mixed precision )      
The aim of this project is to investigate whether the int8 architecture can provide acceleration compared to the fp16/fp32 architecture (in particular, there must be good INT8 computing structures for this to be profitable)       
You may find more info in [project presentation](Quantized_GEMM.pdf)
<!-- You may find more info in [project presentation](https://alexeybelkov.github.io/YSDA-CPU-inference/tree/pytorch/Quantized_GEMM.pdf) -->

### C++ config      
In this branch we used directly builded libtorch as in [Building libtorch using CMake](https://github.com/pytorch/pytorch/blob/main/docs/libtorch.rst#building-libtorch-using-cmake)        
We builded it in Debug mode, to do this the one needs to run the following commands in **/cpp** folder. 
> [!WARNING] 
> Overall build will require a little less than 23 GB of disk space and about 14 GB of CPU RAM    
```shell
git clone -b main --recurse-submodule https://github.com/pytorch/pytorch.git
mkdir pytorch-build
cd pytorch-build
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Debug -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch
cmake --build . --target install
```
Then, also in **/cpp** folder run      
```shell
mkdir build
cd build
cmake ..
make
```
       
### Usefull links   
- [FBGEMM: Enabling High-Performance Low-Precision Deep Learning Inference](https://arxiv.org/pdf/2101.05615.pdf)   
- [Intra-operator parallelism settings in PyTorch](https://github.com/pytorch/pytorch/issues/19001)
- [PyTorch Benchmark](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [PyTorch Numeric Suite](https://pytorch.org/docs/stable/torch.ao.ns._numeric_suite.html) and [guide on it](https://github.com/pytorch/tutorials/blob/main/prototype_source/numeric_suite_tutorial.py)
- [PyTorch Numeric Suite FX](https://pytorch.org/docs/stable/torch.ao.ns._numeric_suite_fx.html#torch-ao-ns-numeric-suite-fx)
- [Pareto-Optimal Quantized ResNet Is Mostly 4-bit](https://arxiv.org/abs/2105.03536#:~:text=Quantization%20has%20become%20a%20popular,without%20changing%20the%20network%20size.)
- [Introducing Quantized Tensor](https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor)
- [How to optimize GEMM](https://github.com/flame/how-to-optimize-gemm)
- [Eigen library](https://gitlab.com/libeigen/eigen)

### Additioanl topic-related papers from experienced researchers

- [Intriguing Properties of Quantization at Scale](https://arxiv.org/abs/2305.19268)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Few-Bit Backward: Quantized Gradients of Activation Functions for Memory Footprint Reduction](https://arxiv.org/abs/2202.00441)
- [And the Bit Goes Down: Revisiting the Quantization of Neural Networks ](https://openreview.net/forum?id=rJehVyrKwH)
