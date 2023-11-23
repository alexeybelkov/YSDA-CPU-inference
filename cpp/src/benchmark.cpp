#include <ATen/ATen.h>
#include <iostream>
#include <torch/torch.h>
#include <chrono>
#include <torch/script.h>

int main() {

    size_t num_trials = 128;
    std::cout << "at::get_num_interop_threads: " << at::get_num_interop_threads() << " at::get_num_threads " << at::get_num_threads() << std::endl;
    at::set_num_interop_threads(1);
    at::set_num_threads(1);
    std::cout << "at::get_num_interop_threads: " << at::get_num_interop_threads() << " at::get_num_threads " << at::get_num_threads() << std::endl;

    torch::Tensor weight = torch::rand({256, 512}, torch::requires_grad(false));
    torch::Tensor input = torch::rand({256, 512}, torch::requires_grad(false));

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_trials; ++i)
        auto output = torch::linear(input, weight);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
    std::cout << "torch::linear " << duration / num_trials << std::endl;

    torch::jit::Module linear = torch::jit::load("../../gitignore/linear.pt");
    std::vector<torch::jit::IValue> jit_input = {input};
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_trials; ++i)
        auto output = linear.forward(jit_input);
    stop = std::chrono::high_resolution_clock::now();
    duration = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
    std::cout << "jit::linear " << duration / num_trials << std::endl;

    torch::jit::Module x86_linear = torch::jit::load("../../gitignore/x86_linear.pt");
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_trials; ++i)
        auto output = x86_linear.forward(jit_input);
    stop = std::chrono::high_resolution_clock::now();
    duration = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
    std::cout << "jit::x86_linear " << duration / num_trials << std::endl;

    torch::jit::Module fbgemm_linear = torch::jit::load("../../gitignore/fbgemm_linear.pt");
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_trials; ++i)
        auto output = fbgemm_linear.forward(jit_input);
    stop = std::chrono::high_resolution_clock::now();
    duration = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
    std::cout << "jit::fbgemm_linear " << duration / num_trials << std::endl;
}  