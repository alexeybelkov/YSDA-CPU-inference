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

    at::set_num_interop_threads(1);
    at::set_num_threads(1);

    at::Tensor weight = at::rand({256, 512}, at::requires_grad(false));
    at::Tensor bias = at::rand({256}, at::requires_grad(false));
    at::Tensor input = at::rand({256, 512}, at::requires_grad(false));

    torch::jit::Module fbgemm_linear = torch::jit::load("../../gitignore/jit_models/fbgemm_linear_256x512.pt");
    std::vector<torch::jit::IValue> jit_input = {input};
    auto out = fbgemm_linear.forward(jit_input);

    // https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor

    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/torch/csrc/jit/api/function_impl.cpp:63
    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/torch/csrc/jit/runtime/graph_executor.cpp:582
    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/torch/csrc/autograd/generated/VariableType_2.cpp:14078  quantize_per_tensor_tensor_qparams
    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch-build/aten/src/ATen/Operators_2.cpp:7962
    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch-build/aten/src/ATen/Operators_2.cpp:8155
    // at::(anonymous namespace)::(anonymous namespace)::wrapper_CPU___local_scalar_dense (self=...) at /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch-build/aten/src/ATen/RegisterCPU.cpp:12946
    //  at::make_per_tensor_affine_quantizer (scale=0.061936106532812119, zero_point=64, scalar_type=c10::ScalarType::QUInt8) at /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/quantized/Quantizer.cpp:46
    // at::native::quantize_per_tensor_tensor_qparams (self=..., scale=..., zero_point=..., dtype=c10::ScalarType::QUInt8) at /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/quantized/QTensor.cpp:72
    // at::PerTensorAffineQuantizer::quantize (this=0x555557ef8350, rtensor=...) at /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/quantized/Quantizer.cpp:164
    // at::new_qtensor (sizes=..., options=..., quantizer=...) at /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/quantized/Quantizer.cpp:113
    // at::PerTensorAffineQuantizer::quantize (this=0x555557ef8350, rtensor=...) at /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/quantized/Quantizer.cpp:179
    // at::native::quantize_tensor_per_tensor_affine (rtensor=..., qtensor=..., scale=0.061936106532812119, zero_point=64) at /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/quantized/AffineQuantizer.cpp:121
    // at::native::(anonymous namespace)::quantize_tensor_per_tensor_affine_cpu (rtensor=..., qtensor=..., scale=0.061936106532812119, zero_point=64) at /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:3335

    // qlinear
    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/quantized/cpu/qlinear.cpp:1034
    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch-build/aten/src/ATen/RegisterQuantizedCPU.cpp:227
    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/aten/src/ATen/native/quantized/cpu/qlinear.cpp:37

    // kernel execution
    // /home/alexey/YSDA/YSDA-CPU-inference/cpp/pytorch/third_party/fbgemm/src/ExecuteKernelU8S8.cc:301


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