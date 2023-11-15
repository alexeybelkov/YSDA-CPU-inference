#include <execinfo.h>
#include <ATen/ATen.h>
#include <iostream>
#include <torch/torch.h>

int main() {
    // mm -> mm::respatch -> /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch-build/aten/src/ATen/Operators_3.cpp:3896 -> 
    // -> /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch-build/aten/src/ATen/RegisterCPU.cpp:8640 -> 
    // -> /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:1383 -> 
    // ->  /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:1513 -> 
    // ??? /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/CPUBlas.cpp:161 -> 
    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:58

    at::Tensor weight = at::rand({64, 256}, at::requires_grad(false));
    at::Tensor input = at::rand({128, 256}, at::requires_grad(false));
    std::cout << "PRE MATMUL\n";
    // at::Tensor output = at::linear(input, weight); // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:277
    // at::Tensor output = at::matmul(input, weight);
    std::cout << "POST MATMUL\n";
}  