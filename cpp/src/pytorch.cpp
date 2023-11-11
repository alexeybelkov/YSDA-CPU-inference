#include <execinfo.h>
#include <ATen/ATen.h>
#include <iostream>
#include <torch/torch.h>

int main() {
    // mm -> mm::respatch -> /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch-build/aten/src/ATen/Operators_3.cpp:3896
    at::Tensor weight = at::rand({32, 16}, at::requires_grad(false));
    // weight.set_requires_grad(false);
    at::Tensor input = at::rand({4, 32}, at::requires_grad(false));
    input.set_requires_grad(false);
    std::cout << "PRE MATMUL\n";
    at::Tensor output = at::matmul(input, weight);
    std::cout << "POST MATMUL\n";
}  