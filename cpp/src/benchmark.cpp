#include <ATen/ATen.h>
#include <iostream>
#include <torch/torch.h>
#include <chrono>
#include <torch/script.h>
#include <string>

const size_t NUM_TRIALS = 100;

template <typename DimSize, typename ChronoTimePoint>
void print_result(const std::string& layer_name, DimSize n, DimSize m, ChronoTimePoint start,
                  ChronoTimePoint stop) {
    auto duration = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
    std::cout << layer_name << ' ' << n << 'x' << m << ' '
              << duration / static_cast<double>(NUM_TRIALS) << std::endl;
}

int main() {
    // size_t n = 256, m = 512;
    std::vector<int> N = {256, 1024, 2048};
    std::vector<int> M = {512, 2048, 4096};
    std::cout << "at::get_num_interop_threads: " << at::get_num_interop_threads()
              << " at::get_num_threads " << at::get_num_threads() << std::endl;
    at::set_num_interop_threads(1);
    at::set_num_threads(1);
    std::cout << "at::get_num_interop_threads: " << at::get_num_interop_threads()
              << " at::get_num_threads " << at::get_num_threads() << std::endl;

    for (size_t i = 0; i < N.size(); ++i) {
        int n = N.at(i);
        int m = M.at(i);
        torch::Tensor weight = torch::rand({n, m}, torch::requires_grad(false));
        torch::Tensor input = torch::rand({n, m}, torch::requires_grad(false));

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_TRIALS; ++i)
            auto output = torch::linear(input, weight);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
        std::cout << "torch::linear " << n << 'x' << m << ' ' << duration / NUM_TRIALS << std::endl;

        std::vector<torch::jit::IValue> jit_input = {input};

        // std::string linear_path = "../../gitignore/jit_models/linear"
        // torch::jit::Module linear = torch::jit::load("../../gitignore/linear.pt");
        // std::vector<torch::jit::IValue> jit_input = {input};
        // auto start = std::chrono::high_resolution_clock::now();
        // for (size_t i = 0; i < NUM_TRIALS; ++i)
        //     auto output = linear.forward(jit_input);
        // auto stop = std::chrono::high_resolution_clock::now();
        // auto duration = static_cast<double>(
        //     std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
        // std::cout << "jit::linear " << duration / NUM_TRIALS << std::endl;

        std::string linear_name = "linear_" + std::to_string(n) + "x" + std::to_string(m) + ".pt";
        std::string x86_linear_path = "../../gitignore/jit_models/x86_" + linear_name;
        torch::jit::Module x86_linear = torch::jit::load(x86_linear_path);
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_TRIALS; ++i)
            auto output = x86_linear.forward(jit_input);
        stop = std::chrono::high_resolution_clock::now();

        print_result("jit::x86_linear", n, m, start, stop);

        // duration = static_cast<double>(
        //     std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
        // std::cout << "jit::x86_linear " << n << 'x' << m << duration / NUM_TRIALS << std::endl;

        std::string fbgemm_linear_path = "../../gitignore/jit_models/fbgemm_" + linear_name;
        torch::jit::Module fbgemm_linear = torch::jit::load(fbgemm_linear_path);
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_TRIALS; ++i)
            auto output = fbgemm_linear.forward(jit_input);
        stop = std::chrono::high_resolution_clock::now();

        print_result("jit::fbgemm_linear", n, m, start, stop);
        // duration = static_cast<double>(
        //     std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
        // std::cout << "jit::fbgemm_linear " << n << 'x' << m << duration / NUM_TRIALS <<
        // std::endl;
    }
}