# YSDA-CPU-inference
### Quantized inference on CPU (int8 / int4 / mixed precision )      
The aim of this project is to investigate whether the int8 architecture can provide acceleration compared to the fp16/fp32 architecture (in particular, there must be good INT8 computing structures for this to be profitable)       

### C++ config      
To be able to compile & run C++ code in **/cpp** folder, you need to add libtorch as in [this guide](https://pytorch.org/cppdocs/installing.html#minimal-example)       
       
### Usefull links      

- [Intra-operator parallelism settings in PyTorch](https://github.com/pytorch/pytorch/issues/19001)
- [PyTorch Benchmark](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [PyTorch Numeric Suite](https://pytorch.org/docs/stable/torch.ao.ns._numeric_suite.html) and [guide on it](https://github.com/pytorch/tutorials/blob/main/prototype_source/numeric_suite_tutorial.py)
- [PyTorch Numeric Suite FX](https://pytorch.org/docs/stable/torch.ao.ns._numeric_suite_fx.html#torch-ao-ns-numeric-suite-fx)