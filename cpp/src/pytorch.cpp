#include <execinfo.h>
#include <ATen/ATen.h>
#include <iostream>
#include <torch/torch.h>

void see_backtrace() {
    
    at::Tensor weight = at::rand({32, 16});
    weight.set_requires_grad(false);
    at::Tensor input = at::rand({4, 32});
    input.set_requires_grad(false);
    std::cout << "PRE MATMUL\n";
    at::Tensor output = at::matmul(input, weight);
    std::cout << "POST MATMUL\n";
}

int main() {

    // at::Tensor weight = at::rand({32, 16});
    // weight.set_requires_grad(false);
    // at::Tensor input = at::rand({4, 32});
    // input.set_requires_grad(false);

    // at::Tensor output = at::matmul(input, weight);  // at::linear -> transpose ->  at::matmul // in transpose it just swap
    // std::cout << output.size(0) << std::endl;
    see_backtrace();
}  