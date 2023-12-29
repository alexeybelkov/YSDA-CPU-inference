#include <execinfo.h>
#include <ATen/ATen.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <chrono>

int main() {

    torch::NoGradGuard no_grad;
    at::set_num_interop_threads(1);
    at::set_num_threads(1);

    std::vector<int> B = {1, 2, 4, 8, 16, 32, 64, 256, 512, 1024};
    std::vector<std::array<double, 10>> times;

    for (int b : B) {
        at::Tensor input = at::rand({b, 512}, at::requires_grad(false));
        times.push_back({});
        torch::jit::Module fbgemm_linear =
            torch::jit::load("../../gitignore/jit_models/fbgemm_linear_256x512.pt");
        std::vector<torch::jit::IValue> jit_input = {input};
        auto& t = times[times.size() - 1];
        for (size_t i = 0; i < 100 + t.size(); ++i) {
            auto start = std::chrono::system_clock::now();
            auto out = fbgemm_linear.forward(jit_input);
            auto end = std::chrono::system_clock::now();
            double elapsed =
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            if (i < 9) {
                t[i] = elapsed;
            }

            else if (i >= 9) {
                t[9] += elapsed;
            }
        }
        t[9] /= 100.0;

        std::cout << b << ' ';
        for (double ms : t) {
            std::cout << ms << ' ';
        }
        std::cout << std::endl;
    }
}
