#include <ATen/ATen.h>
#include <iostream>
#include <torch/torch.h>

int main() {
    at::Tensor weight = at::rand({128, 256, 3, 3, 3});
    at::Tensor input = at::rand({1, 256, 64, 64, 64});
    // at::Tensor weight = at::rand({1024, 512});
    // at::Tensor input = at::rand({1, 512});
    at::Tensor output;
    output = at::conv3d(input, weight);
    // auto x = at::rand({64, 64}), y = at::rand({64, 64}), z = at::rand({64, 64});
    // output = at::native::linear(x, y, z);
    std::cout << output[0][0];
}