#include <sycl/sycl.hpp>

#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
void launch_decode(const torch::Tensor &x, torch::Tensor &state,
                   const torch::Tensor &weight, const torch::Tensor &indices,
                   torch::Tensor &output) {
  const auto batch = x.size(0);
  const auto channels = x.size(1);
  const auto state_len = state.size(2);
  const auto kernel = weight.size(1);

  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  auto *state_ptr = reinterpret_cast<scalar_t *>(state.data_ptr());
  const auto *weight_ptr =
      reinterpret_cast<const scalar_t *>(weight.data_ptr());
  const auto *index_ptr = indices.data_ptr<int32_t>();
  auto *output_ptr = reinterpret_cast<scalar_t *>(output.data_ptr());

  sycl::queue &queue = c10::xpu::getCurrentXPUStream(x.get_device()).queue();
  queue.parallel_for(sycl::range<1>(batch * channels), [=](sycl::id<1> item) {
    const int64_t linear = item[0];
    const int64_t request = linear / channels;
    const int64_t channel = linear - request * channels;
    const int32_t slot = index_ptr[request];
    const int64_t state_base =
        (static_cast<int64_t>(slot) * channels + channel) * state_len;
    const int64_t weight_base = channel * kernel;

    float sum = 0.0f;
    for (int64_t tap = 0; tap < state_len; ++tap) {
      const float value = static_cast<float>(state_ptr[state_base + tap]);
      sum += value * static_cast<float>(weight_ptr[weight_base + tap]);
      if (tap + 1 < state_len) {
        state_ptr[state_base + tap] = state_ptr[state_base + tap + 1];
      }
    }
    const scalar_t current = x_ptr[linear];
    state_ptr[state_base + state_len - 1] = current;
    sum += static_cast<float>(current) *
           static_cast<float>(weight_ptr[weight_base + kernel - 1]);
    const float activated = sum / (1.0f + sycl::exp(-sum));
    output_ptr[linear] = static_cast<scalar_t>(activated);
  });
}

torch::Tensor causal_conv1d_decode(torch::Tensor x, torch::Tensor state,
                                   torch::Tensor weight,
                                   torch::Tensor indices) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(state.device() == x.device(),
              "state must be on the same XPU device as x");
  TORCH_CHECK(weight.device() == x.device(),
              "weight must be on the same XPU device as x");
  TORCH_CHECK(indices.device() == x.device(),
              "indices must be on the same XPU device as x");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must have shape [batch, channels]");
  TORCH_CHECK(state.dim() == 3,
              "state must have shape [slots, channels, kernel-1]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [channels, kernel]");
  TORCH_CHECK(indices.dim() == 1 && indices.size(0) == x.size(0),
              "indices must have shape [batch]");
  TORCH_CHECK(state.size(1) == x.size(1), "state channel count must match x");
  TORCH_CHECK(weight.size(0) == x.size(1), "weight channel count must match x");
  TORCH_CHECK(weight.size(1) == state.size(2) + 1,
              "weight kernel size must equal state length + 1");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(state.scalar_type() == x.scalar_type(),
              "state dtype must match x");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(),
              "weight dtype must match x");

  auto output = torch::empty_like(x);
  if (x.scalar_type() == torch::kFloat32) {
    launch_decode<float>(x, state, weight, indices, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_decode<sycl::ext::oneapi::bfloat16>(x, state, weight, indices,
                                               output);
  } else {
    TORCH_CHECK(false, "causal_conv1d_decode supports float32 and bfloat16");
  }
  return output;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("causal_conv1d_decode", &causal_conv1d_decode,
             "SYCL causal depthwise convolution decode");
}
