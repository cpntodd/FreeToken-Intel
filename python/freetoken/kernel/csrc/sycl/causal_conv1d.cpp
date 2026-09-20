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

template <typename scalar_t>
void launch_q8_0_matvec(const torch::Tensor &x, const torch::Tensor &qweight,
                         torch::Tensor &output) {
  constexpr int64_t kBlockSize = 32;
  constexpr int64_t kBlockBytes = 34;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  auto *out_ptr = reinterpret_cast<scalar_t *>(output.data_ptr());
  sycl::queue &queue = c10::xpu::getCurrentXPUStream(x.get_device()).queue();
  const int64_t groups = batch * out_features;

  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroupSize),
                        sycl::range<1>(kWorkgroupSize)),
      [=](sycl::nd_item<1> item) {
        const int64_t group = item.get_group(0);
        const int64_t token = group / out_features;
        const int64_t row = group - token * out_features;
        float partial = 0.0f;
        for (int64_t block = item.get_local_id(0); block < blocks_per_row;
             block += kWorkgroupSize) {
          const uint8_t *packed = weight_ptr +
              (row * blocks_per_row + block) * kBlockBytes;
          const float scale = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed));
          const auto *quants = reinterpret_cast<const int8_t *>(packed + 2);
          const int64_t input_base = token * in_features + block * kBlockSize;
          for (int64_t element = 0; element < kBlockSize; ++element) {
            partial += static_cast<float>(x_ptr[input_base + element]) *
                       static_cast<float>(quants[element]) * scale;
          }
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor q8_0_matvec(torch::Tensor x, torch::Tensor qweight) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device(),
              "qweight must be on the same XPU device as x");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous(),
              "x and qweight must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8,
              "qweight must use uint8 storage");
  TORCH_CHECK(x.size(1) % 32 == 0,
              "Q8_0 input width must be divisible by 32");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 32 * 34,
              "qweight has invalid Q8_0 row geometry");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_q8_0_matvec<float>(x, qweight, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_q8_0_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, output);
  } else {
    TORCH_CHECK(false, "q8_0_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_q4_k_matvec(const torch::Tensor &x, const torch::Tensor &qweight,
                        torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 144;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  auto *out_ptr = reinterpret_cast<scalar_t *>(output.data_ptr());
  sycl::queue &queue = c10::xpu::getCurrentXPUStream(x.get_device()).queue();
  const int64_t groups = batch * out_features;

  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroupSize),
                        sycl::range<1>(kWorkgroupSize)),
      [=](sycl::nd_item<1> item) {
        const int64_t group = item.get_group(0);
        const int64_t token = group / out_features;
        const int64_t row = group - token * out_features;
        float partial = 0.0f;
        for (int64_t index = item.get_local_id(0); index < in_features;
             index += kWorkgroupSize) {
          const int64_t block = index / kBlockSize;
          const int64_t in_block = index % kBlockSize;
          const uint8_t *packed = weight_ptr +
              (row * blocks_per_row + block) * kBlockBytes;
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed));
          const float dmin = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed + 2));
          const int64_t group64 = in_block / 64;
          const int64_t scale_index = group64 * 2 + (in_block % 64 >= 32);
          int scale;
          int minimum;
          if (scale_index < 4) {
            scale = packed[4 + scale_index] & 0x3F;
            minimum = packed[8 + scale_index] & 0x3F;
          } else {
            scale = (packed[4 + scale_index + 4] & 0x0F) |
                    ((packed[4 + scale_index - 4] >> 6) << 4);
            minimum = (packed[4 + scale_index + 4] >> 4) |
                      ((packed[4 + scale_index] >> 6) << 4);
          }
          const uint8_t quant_byte = packed[16 + group64 * 32 + (in_block % 32)];
          const int quant = in_block % 64 < 32 ? quant_byte & 0x0F : quant_byte >> 4;
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (d * static_cast<float>(scale * quant) -
                      dmin * static_cast<float>(minimum));
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor q4_k_matvec(torch::Tensor x, torch::Tensor qweight) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device(),
              "qweight must be on the same XPU device as x");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous(),
              "x and qweight must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8,
              "qweight must use uint8 storage");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "Q4_K input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 144,
              "qweight has invalid Q4_K row geometry");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_q4_k_matvec<float>(x, qweight, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_q4_k_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, output);
  } else {
    TORCH_CHECK(false, "q4_k_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_q2_k_matvec(const torch::Tensor &x, const torch::Tensor &qweight,
                        torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 84;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  auto *out_ptr = reinterpret_cast<scalar_t *>(output.data_ptr());
  sycl::queue &queue = c10::xpu::getCurrentXPUStream(x.get_device()).queue();
  const int64_t groups = batch * out_features;

  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroupSize),
                        sycl::range<1>(kWorkgroupSize)),
      [=](sycl::nd_item<1> item) {
        const int64_t group = item.get_group(0);
        const int64_t token = group / out_features;
        const int64_t row = group - token * out_features;
        float partial = 0.0f;
        for (int64_t index = item.get_local_id(0); index < in_features;
             index += kWorkgroupSize) {
          const int64_t block = index / kBlockSize;
          const int64_t in_block = index % kBlockSize;
          const uint8_t *packed = weight_ptr +
              (row * blocks_per_row + block) * kBlockBytes;
          const int64_t scale_index = in_block / 16;
          const uint8_t scale_min = packed[scale_index];
          const int scale = scale_min & 0x0F;
          const int minimum = scale_min >> 4;
          const int64_t group_in_half = scale_index % 8;
          const int64_t quant_offset = (scale_index / 8) * 32 +
              (group_in_half % 2) * 16 + in_block % 16;
          const int shift = (group_in_half / 2) * 2;
          const int quant = (packed[16 + quant_offset] >> shift) & 0x03;
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed + 80));
          const float dmin = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed + 82));
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (d * static_cast<float>(scale * quant) -
                      dmin * static_cast<float>(minimum));
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor q2_k_matvec(torch::Tensor x, torch::Tensor qweight) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device(),
              "qweight must be on the same XPU device as x");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous(),
              "x and qweight must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8,
              "qweight must use uint8 storage");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "Q2_K input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 84,
              "qweight has invalid Q2_K row geometry");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_q2_k_matvec<float>(x, qweight, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_q2_k_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, output);
  } else {
    TORCH_CHECK(false, "q2_k_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_q3_k_matvec(const torch::Tensor &x, const torch::Tensor &qweight,
                        torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 110;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  auto *out_ptr = reinterpret_cast<scalar_t *>(output.data_ptr());
  sycl::queue &queue = c10::xpu::getCurrentXPUStream(x.get_device()).queue();
  const int64_t groups = batch * out_features;

  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroupSize),
                        sycl::range<1>(kWorkgroupSize)),
      [=](sycl::nd_item<1> item) {
        const int64_t group = item.get_group(0);
        const int64_t token = group / out_features;
        const int64_t row = group - token * out_features;
        float partial = 0.0f;
        for (int64_t index = item.get_local_id(0); index < in_features;
             index += kWorkgroupSize) {
          const int64_t block = index / kBlockSize;
          const int64_t in_block = index % kBlockSize;
          const uint8_t *packed = weight_ptr +
              (row * blocks_per_row + block) * kBlockBytes;
          const int64_t scale_index = in_block / 16;
          const int64_t scale_word = scale_index / 4;
          const int64_t scale_byte = scale_index % 4;
          const int scale_low = (packed[96 + (scale_word % 2) * 4 + scale_byte] >>
                                 (scale_word >= 2 ? 4 : 0)) & 0x0F;
          const int scale_high = (packed[104 + scale_byte] >> (scale_word * 2)) & 0x03;
          const int scale = (scale_low | (scale_high << 4)) - 32;
          const int64_t group_in_half = scale_index % 8;
          const int64_t quant_offset = (scale_index / 8) * 32 +
              (group_in_half % 2) * 16 + in_block % 16;
          const int shift = (group_in_half / 2) * 2;
          const int low = (packed[32 + quant_offset] >> shift) & 0x03;
          const bool high = (packed[(group_in_half % 2) * 16 + in_block % 16] &
                             (1 << (scale_index / 2))) != 0;
          const int quant = low - (high ? 0 : 4);
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed + 108));
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     d * static_cast<float>(scale * quant);
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor q3_k_matvec(torch::Tensor x, torch::Tensor qweight) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device(),
              "qweight must be on the same XPU device as x");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous(),
              "x and qweight must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8,
              "qweight must use uint8 storage");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "Q3_K input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 110,
              "qweight has invalid Q3_K row geometry");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_q3_k_matvec<float>(x, qweight, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_q3_k_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, output);
  } else {
    TORCH_CHECK(false, "q3_k_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_q5_k_matvec(const torch::Tensor &x, const torch::Tensor &qweight,
                        torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 176;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  auto *out_ptr = reinterpret_cast<scalar_t *>(output.data_ptr());
  sycl::queue &queue = c10::xpu::getCurrentXPUStream(x.get_device()).queue();
  const int64_t groups = batch * out_features;

  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroupSize),
                        sycl::range<1>(kWorkgroupSize)),
      [=](sycl::nd_item<1> item) {
        const int64_t group = item.get_group(0);
        const int64_t token = group / out_features;
        const int64_t row = group - token * out_features;
        float partial = 0.0f;
        for (int64_t index = item.get_local_id(0); index < in_features;
             index += kWorkgroupSize) {
          const int64_t block = index / kBlockSize;
          const int64_t in_block = index % kBlockSize;
          const uint8_t *packed = weight_ptr +
              (row * blocks_per_row + block) * kBlockBytes;
          const int64_t group64 = in_block / 64;
          const int64_t scale_index = group64 * 2 + (in_block % 64 >= 32);
          int scale;
          int minimum;
          if (scale_index < 4) {
            scale = packed[4 + scale_index] & 0x3F;
            minimum = packed[8 + scale_index] & 0x3F;
          } else {
            scale = (packed[4 + scale_index + 4] & 0x0F) |
                    ((packed[4 + scale_index - 4] >> 6) << 4);
            minimum = (packed[4 + scale_index + 4] >> 4) |
                      ((packed[4 + scale_index] >> 6) << 4);
          }
          const int64_t in64 = in_block % 64;
          const uint8_t quant_byte = packed[48 + group64 * 32 + (in_block % 32)];
          int quant = in64 < 32 ? quant_byte & 0x0F : quant_byte >> 4;
          const uint8_t high_bits = packed[16 + (in_block % 32)];
          const int high_bit = in64 < 32
              ? ((high_bits >> (2 * group64)) & 1)
              : ((high_bits >> (2 * group64 + 1)) & 1);
          quant |= high_bit << 4;
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed));
          const float dmin = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed + 2));
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (d * static_cast<float>(scale * quant) -
                      dmin * static_cast<float>(minimum));
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor q5_k_matvec(torch::Tensor x, torch::Tensor qweight) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device(),
              "qweight must be on the same XPU device as x");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous(),
              "x and qweight must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8,
              "qweight must use uint8 storage");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "Q5_K input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 176,
              "qweight has invalid Q5_K row geometry");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_q5_k_matvec<float>(x, qweight, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_q5_k_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, output);
  } else {
    TORCH_CHECK(false, "q5_k_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("causal_conv1d_decode", &causal_conv1d_decode,
             "SYCL causal depthwise convolution decode");
  module.def("q8_0_matvec", &q8_0_matvec, "SYCL Q8_0 matrix-vector product");
  module.def("q4_k_matvec", &q4_k_matvec, "SYCL Q4_K matrix-vector product");
  module.def("q2_k_matvec", &q2_k_matvec, "SYCL Q2_K matrix-vector product");
  module.def("q3_k_matvec", &q3_k_matvec, "SYCL Q3_K matrix-vector product");
  module.def("q5_k_matvec", &q5_k_matvec, "SYCL Q5_K matrix-vector product");
}
