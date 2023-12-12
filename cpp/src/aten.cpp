#include <ATen/ATen.h>
#include <iostream>
#include <torch/torch.h>

int main() {
    at::Tensor weight = at::rand({64, 256}, at::requires_grad(false));
    at::Tensor input = at::rand({128, 256}, at::requires_grad(false));
    at::Tensor output = at::linear(input, weight);
    std::cout << output.size(0);
}

