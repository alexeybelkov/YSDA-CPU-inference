#include <ATen/ATen.h>
#include <iostream>
#include <torch/torch.h>

int main() {
    torch::Tensor weight = torch::randn({128, 256, 3, 3, 3});
    torch::Tensor input = torch::randn({1, 256, 4, 4, 4});
    at::conv3d(input, weight);
}