#include <execinfo.h>
#include <ATen/ATen.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
// #include <ATen/aten_op.h>
// #include <ATen/native/quantized/cpu/qlinear.cpp>

int main() {
    // using at::_ops::quantized::linear;
    // at::Tensor weight = at::rand({256, 64}, at::requires_grad(false));
    // at::Tensor bias = at::rand({256}, at::requires_grad(false));
    // at::Tensor input = at::rand({256, 64}, at::requires_grad(false));
    
    // https://github.com/pytorch/pytorch/blob/3df2c42921acb2cc4da879b271fe44dd1e93829a/aten/src/ATen/native/QuantizedLinear.cpp#L292C4-L292C66
    // auto [quantized, col_offsets, scale, zero_point] = torch::fbgemm_linear_quantize_weight(weight);
    // auto packed = fbgemm_pack_quantized_matrix(quantized);
    // at::native::fbgemm_linear_int8_weight(input, quantized, packed, col_offsets, scale, zero_point, bias);

    // x = torch.rand(256, 512)
    // linear = nn.Linear(512, 256).eval()

    at::Tensor weight = at::rand({256, 512}, at::requires_grad(false));
    at::Tensor bias = at::rand({256}, at::requires_grad(false));
    at::Tensor input = at::rand({256, 512}, at::requires_grad(false));

    torch::jit::Module linear = torch::jit::load("../../gitignore/linear.pt");
    std::vector<torch::jit::IValue> jit_input = {input};
    std::cout << "PRE LINEAR\n";
    auto out = linear.forward(jit_input);

    // https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor

    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/torch/csrc/jit/api/function_impl.cpp:63
    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/torch/csrc/jit/runtime/graph_executor.cpp:582


    // std::cout << "PRE LINEAR\n";
    
    std::cout << "POST LINEAR\n";


    /*
    #0  at::native::fbgemm_linear_int8_weight_fp32_activation (input=..., weight=..., packed=..., col_offsets=..., weight_scale=..., weight_zero_point=..., bias=...)
    at /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/QuantizedLinear.cpp:130
#1  0x00007fffe4202ce6 in at::native::fbgemm_linear_int8_weight (input=..., weight=..., packed=..., col_offsets=..., weight_scale=..., weight_zero_point=..., bias=...)
    at /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/QuantizedLinear.cpp:200
#2  0x0000555555557c7d in main () at /home/alexey/YSDA/YSDA-CPU-inference/cpp/src/quantized_linear.cpp:19
    */
}  