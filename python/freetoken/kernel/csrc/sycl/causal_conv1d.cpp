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

template <typename scalar_t>
void launch_iq3_xxs_matvec(const torch::Tensor &x, const torch::Tensor &qweight,
                           const torch::Tensor &table,
                           const torch::Tensor &signs, torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 98;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  const auto *table_ptr = table.data_ptr<float>();
  const auto *signs_ptr = signs.data_ptr<int64_t>();
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
          const int64_t subblock = in_block / 32;
          const int64_t position = in_block % 32;
          const int64_t pair = position / 8;
          const int64_t lane = position % 8;
          const int64_t q_offset = 2 + subblock * 8 + pair * 2 + (lane >= 4);
          const int q_index = packed[q_offset];
          const int grid_lane = lane % 4;
          const uint32_t aux = static_cast<uint32_t>(packed[66 + subblock * 4]) |
              (static_cast<uint32_t>(packed[67 + subblock * 4]) << 8) |
              (static_cast<uint32_t>(packed[68 + subblock * 4]) << 16) |
              (static_cast<uint32_t>(packed[69 + subblock * 4]) << 24);
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed));
          const float scale = d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
          const int sign_code = static_cast<int>(signs_ptr[(aux >> (pair * 7)) & 0x7F]);
          const float sign = (sign_code & (1 << lane)) == 0 ? 1.0f : -1.0f;
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (scale * table_ptr[q_index * 4 + grid_lane] * sign);
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor iq3_xxs_matvec(torch::Tensor x, torch::Tensor qweight,
                             torch::Tensor table, torch::Tensor signs) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device() && table.device() == x.device() &&
                  signs.device() == x.device(),
              "IQ3_XXS inputs must share an XPU device");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() &&
                  table.is_contiguous() && signs.is_contiguous(),
              "IQ3_XXS inputs must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8 &&
                  table.scalar_type() == torch::kFloat32 &&
                  signs.scalar_type() == torch::kInt64,
              "IQ3_XXS inputs have invalid dtypes");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "IQ3_XXS input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 98,
              "qweight has invalid IQ3_XXS row geometry");
  TORCH_CHECK(table.numel() == 256 * 4 && signs.numel() >= 128,
              "IQ3_XXS lookup tables have invalid sizes");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_iq3_xxs_matvec<float>(x, qweight, table, signs, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_iq3_xxs_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, table, signs,
                                                       output);
  } else {
    TORCH_CHECK(false, "iq3_xxs_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_iq2_s_matvec(const torch::Tensor &x, const torch::Tensor &qweight,
                         const torch::Tensor &table, torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 82;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  const auto *table_ptr = table.data_ptr<float>();
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
          const int64_t subblock = in_block / 32;
          const int64_t position = in_block % 32;
          const int64_t part = position / 8;
          const int64_t lane = position % 8;
          const int q_index = packed[2 + subblock * 4 + part] |
              ((static_cast<int>(packed[66 + subblock]) << (8 - 2 * part)) & 0x300);
          const int sign_code = packed[34 + subblock * 4 + part];
          const int scale_index = subblock * 2 + part / 2;
          const uint8_t scale_byte = packed[74 + scale_index / 2];
          const int scale_nibble = scale_index % 2 == 0 ?
              scale_byte & 0x0F : scale_byte >> 4;
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed));
          const float scale = d * (0.5f + static_cast<float>(scale_nibble)) * 0.25f;
          const float sign = (sign_code & (1 << lane)) == 0 ? 1.0f : -1.0f;
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (scale * table_ptr[q_index * 8 + lane] * sign);
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor iq2_s_matvec(torch::Tensor x, torch::Tensor qweight,
                           torch::Tensor table) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device() && table.device() == x.device(),
              "IQ2_S inputs must share an XPU device");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() && table.is_contiguous(),
              "IQ2_S inputs must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8 &&
                  table.scalar_type() == torch::kFloat32,
              "IQ2_S inputs have invalid dtypes");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "IQ2_S input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 82,
              "qweight has invalid IQ2_S row geometry");
  TORCH_CHECK(table.numel() == 1024 * 8,
              "IQ2_S lookup table has invalid size");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_iq2_s_matvec<float>(x, qweight, table, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_iq2_s_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, table, output);
  } else {
    TORCH_CHECK(false, "iq2_s_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_iq3_s_matvec(const torch::Tensor &x, const torch::Tensor &qweight,
                         const torch::Tensor &table, torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 110;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  const auto *table_ptr = table.data_ptr<float>();
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
          const int64_t subblock = in_block / 32;
          const int64_t position = in_block % 32;
          const int64_t part = position / 8;
          const int64_t within_part = position % 8;
          const bool second = within_part >= 4;
          const int64_t grid_lane = within_part % 4;
          const uint8_t high_byte = packed[66 + subblock];
          const int high_shift = second ? 7 - 2 * part : 8 - 2 * part;
          const int q_index = packed[2 + subblock * 8 + part * 2 + second] |
              ((static_cast<int>(high_byte) << high_shift) & 0x100);
          const int sign_code = packed[74 + subblock * 4 + part];
          const int scale_byte = packed[106 + subblock / 2];
          const int scale_nibble = subblock % 2 == 0 ?
              scale_byte & 0x0F : scale_byte >> 4;
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed));
          const float scale = d * (1.0f + 2.0f * static_cast<float>(scale_nibble));
          const float sign = (sign_code & (1 << within_part)) == 0 ? 1.0f : -1.0f;
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (scale * table_ptr[q_index * 4 + grid_lane] * sign);
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor iq3_s_matvec(torch::Tensor x, torch::Tensor qweight,
                           torch::Tensor table) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device() && table.device() == x.device(),
              "IQ3_S inputs must share an XPU device");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() && table.is_contiguous(),
              "IQ3_S inputs must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8 &&
                  table.scalar_type() == torch::kFloat32,
              "IQ3_S inputs have invalid dtypes");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "IQ3_S input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 110,
              "qweight has invalid IQ3_S row geometry");
  TORCH_CHECK(table.numel() == 512 * 4,
              "IQ3_S lookup table has invalid size");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_iq3_s_matvec<float>(x, qweight, table, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_iq3_s_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, table, output);
  } else {
    TORCH_CHECK(false, "iq3_s_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_iq2_xxs_matvec(const torch::Tensor &x,
                           const torch::Tensor &qweight,
                           const torch::Tensor &table,
                           const torch::Tensor &signs,
                           torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 66;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  const auto *table_ptr = table.data_ptr<float>();
  const auto *signs_ptr = signs.data_ptr<int64_t>();
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
          const int64_t subblock = in_block / 32;
          const int64_t position = in_block % 32;
          const int64_t grid_index = position / 8;
          const int64_t lane = position % 8;
          const uint32_t low_word = static_cast<uint32_t>(packed[2 + subblock * 8]) |
              (static_cast<uint32_t>(packed[3 + subblock * 8]) << 8) |
              (static_cast<uint32_t>(packed[4 + subblock * 8]) << 16) |
              (static_cast<uint32_t>(packed[5 + subblock * 8]) << 24);
          const uint32_t high_word = static_cast<uint32_t>(packed[6 + subblock * 8]) |
              (static_cast<uint32_t>(packed[7 + subblock * 8]) << 8) |
              (static_cast<uint32_t>(packed[8 + subblock * 8]) << 16) |
              (static_cast<uint32_t>(packed[9 + subblock * 8]) << 24);
          const int q_index = static_cast<int>((low_word >> (8 * grid_index)) & 0xFF);
          const int sign_index = static_cast<int>((high_word >> (7 * grid_index)) & 0x7F);
          const int sign_code = static_cast<int>(signs_ptr[sign_index]);
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed));
          const float scale = d * (0.5f + static_cast<float>(high_word >> 28)) * 0.25f;
          const float sign = (sign_code & (1 << lane)) == 0 ? 1.0f : -1.0f;
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (scale * table_ptr[q_index * 8 + lane] * sign);
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor iq2_xxs_matvec(torch::Tensor x, torch::Tensor qweight,
                             torch::Tensor table, torch::Tensor signs) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device() && table.device() == x.device() &&
                  signs.device() == x.device(),
              "IQ2_XXS inputs must share an XPU device");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() &&
                  table.is_contiguous() && signs.is_contiguous(),
              "IQ2_XXS inputs must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8 &&
                  table.scalar_type() == torch::kFloat32 &&
                  signs.scalar_type() == torch::kInt64,
              "IQ2_XXS inputs have invalid dtypes");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "IQ2_XXS input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 66,
              "qweight has invalid IQ2_XXS row geometry");
  TORCH_CHECK(table.numel() == 256 * 8 && signs.numel() >= 128,
              "IQ2_XXS lookup tables have invalid sizes");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_iq2_xxs_matvec<float>(x, qweight, table, signs, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_iq2_xxs_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, table,
                                                       signs, output);
  } else {
    TORCH_CHECK(false, "iq2_xxs_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_iq2_xs_matvec(const torch::Tensor &x,
                          const torch::Tensor &qweight,
                          const torch::Tensor &table,
                          const torch::Tensor &signs,
                          torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 74;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  const auto *table_ptr = table.data_ptr<float>();
  const auto *signs_ptr = signs.data_ptr<int64_t>();
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
          const int64_t subblock = in_block / 32;
          const int64_t position = in_block % 32;
          const int64_t code_index = position / 8;
          const int64_t lane = position % 8;
          const int64_t code_offset = 2 + subblock * 8 + code_index * 2;
          const int code = static_cast<int>(packed[code_offset]) |
                           (static_cast<int>(packed[code_offset + 1]) << 8);
          const int q_index = code & 0x1FF;
          const int sign_index = (code >> 9) & 0x7F;
          const int sign_code = static_cast<int>(signs_ptr[sign_index]);
          const int scale_index = subblock * 2 + code_index / 2;
          const uint8_t scale_byte = packed[66 + scale_index / 2];
          const int scale_nibble = scale_index % 2 == 0 ?
              scale_byte & 0x0F : scale_byte >> 4;
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed));
          const float scale = d * (0.5f + static_cast<float>(scale_nibble)) * 0.25f;
          const float sign = (sign_code & (1 << lane)) == 0 ? 1.0f : -1.0f;
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (scale * table_ptr[q_index * 8 + lane] * sign);
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor iq2_xs_matvec(torch::Tensor x, torch::Tensor qweight,
                            torch::Tensor table, torch::Tensor signs) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device() && table.device() == x.device() &&
                  signs.device() == x.device(),
              "IQ2_XS inputs must share an XPU device");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() &&
                  table.is_contiguous() && signs.is_contiguous(),
              "IQ2_XS inputs must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8 &&
                  table.scalar_type() == torch::kFloat32 &&
                  signs.scalar_type() == torch::kInt64,
              "IQ2_XS inputs have invalid dtypes");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "IQ2_XS input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 74,
              "qweight has invalid IQ2_XS row geometry");
  TORCH_CHECK(table.numel() == 512 * 8 && signs.numel() >= 128,
              "IQ2_XS lookup tables have invalid sizes");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_iq2_xs_matvec<float>(x, qweight, table, signs, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_iq2_xs_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, table,
                                                      signs, output);
  } else {
    TORCH_CHECK(false, "iq2_xs_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_iq4_xs_matvec(const torch::Tensor &x,
                          const torch::Tensor &qweight,
                          torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 136;
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
        constexpr float codebook[16] = {
            -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
            1.0f,    13.0f,   25.0f,  38.0f,  53.0f,  69.0f,  89.0f,  113.0f};
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
          const int64_t subblock = in_block / 32;
          const int64_t position = in_block % 32;
          const uint16_t high_scales = static_cast<uint16_t>(packed[2]) |
                                       (static_cast<uint16_t>(packed[3]) << 8);
          const int low = (packed[4 + subblock / 2] >> (4 * (subblock % 2))) & 0x0F;
          const int high = ((high_scales >> (2 * subblock)) & 0x03) << 4;
          const int scale = (low | high) - 32;
          const uint8_t quant_byte = packed[8 + subblock * 16 + (position % 16)];
          const int quant = position < 16 ? quant_byte & 0x0F : quant_byte >> 4;
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed));
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (d * static_cast<float>(scale) * codebook[quant]);
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor iq4_xs_matvec(torch::Tensor x, torch::Tensor qweight) {
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
              "IQ4_XS input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 136,
              "qweight has invalid IQ4_XS row geometry");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_iq4_xs_matvec<float>(x, qweight, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_iq4_xs_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, output);
  } else {
    TORCH_CHECK(false, "iq4_xs_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_iq1_s_matvec(const torch::Tensor &x,
                         const torch::Tensor &qweight,
                         const torch::Tensor &table,
                         torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 50;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  const auto *table_ptr = table.data_ptr<float>();
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
          const int64_t subblock = in_block / 32;
          const int64_t position = in_block % 32;
          const int64_t part = position / 8;
          const int64_t lane = position % 8;
          const uint16_t high = static_cast<uint16_t>(packed[34 + subblock * 2]) |
                                (static_cast<uint16_t>(packed[35 + subblock * 2]) << 8);
          const int q_index = packed[2 + subblock * 4 + part] |
              (((static_cast<int>(high) >> (3 * part)) & 0x07) << 8);
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(packed));
          const float scale = d * (2.0f * static_cast<float>((high >> 12) & 0x07) + 1.0f);
          const float delta = (high & 0x8000) == 0 ? 0.125f : -0.125f;
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (scale * (table_ptr[q_index * 8 + lane] + delta));
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

template <typename scalar_t>
void launch_iq1_s_matvec_tiled(const torch::Tensor &x,
                               const torch::Tensor &qweight,
                               const torch::Tensor &table,
                               torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 50;
  constexpr int64_t kWorkgroupSize = 256;
  constexpr int64_t kTokenTile = 4;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const int64_t token_tiles = (batch + kTokenTile - 1) / kTokenTile;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  const auto *table_ptr = table.data_ptr<float>();
  auto *out_ptr = reinterpret_cast<scalar_t *>(output.data_ptr());
  sycl::queue &queue = c10::xpu::getCurrentXPUStream(x.get_device()).queue();
  const int64_t groups = token_tiles * out_features;

  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroupSize),
                        sycl::range<1>(kWorkgroupSize)),
      [=](sycl::nd_item<1> item) {
        const int64_t group = item.get_group(0);
        const int64_t token_base = (group / out_features) * kTokenTile;
        const int64_t row = group % out_features;
        float partial[kTokenTile] = {};
        for (int64_t index = item.get_local_id(0); index < in_features;
             index += kWorkgroupSize) {
          const int64_t block = index / kBlockSize;
          const int64_t in_block = index % kBlockSize;
          const uint8_t *packed =
              weight_ptr + (row * blocks_per_row + block) * kBlockBytes;
          const int64_t subblock = in_block / 32;
          const int64_t position = in_block % 32;
          const int64_t part = position / 8;
          const int64_t lane = position % 8;
          const uint16_t high =
              static_cast<uint16_t>(packed[34 + subblock * 2]) |
              (static_cast<uint16_t>(packed[35 + subblock * 2]) << 8);
          const int q_index =
              packed[2 + subblock * 4 + part] |
              (((static_cast<int>(high) >> (3 * part)) & 0x07) << 8);
          const float d =
              static_cast<float>(*reinterpret_cast<const sycl::half *>(packed));
          const float scale =
              d * (2.0f * static_cast<float>((high >> 12) & 0x07) + 1.0f);
          const float delta = (high & 0x8000) == 0 ? 0.125f : -0.125f;
          const float weight = scale * (table_ptr[q_index * 8 + lane] + delta);
          for (int64_t tile = 0; tile < kTokenTile; ++tile) {
            const int64_t token = token_base + tile;
            if (token < batch) {
              partial[tile] +=
                  static_cast<float>(x_ptr[token * in_features + index]) *
                  weight;
            }
          }
        }
        for (int64_t tile = 0; tile < kTokenTile; ++tile) {
          const int64_t token = token_base + tile;
          const float sum = sycl::reduce_over_group(
              item.get_group(), partial[tile], sycl::plus<float>());
          if (item.get_local_id(0) == 0 && token < batch) {
            out_ptr[token * out_features + row] = static_cast<scalar_t>(sum);
          }
        }
      });
}

torch::Tensor iq1_s_matvec(torch::Tensor x, torch::Tensor qweight,
                           torch::Tensor table) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device() && table.device() == x.device(),
              "IQ1_S inputs must share an XPU device");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() &&
                  table.is_contiguous(),
              "IQ1_S inputs must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8 &&
                  table.scalar_type() == torch::kFloat32,
              "IQ1_S inputs have invalid dtypes");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "IQ1_S input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 50,
              "qweight has invalid IQ1_S row geometry");
  TORCH_CHECK(table.numel() == 2048 * 8, "IQ1_S lookup table has invalid size");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    if (x.size(0) >= 4) {
      launch_iq1_s_matvec_tiled<float>(x, qweight, table, output);
    } else {
      launch_iq1_s_matvec<float>(x, qweight, table, output);
    }
  } else if (x.scalar_type() == torch::kBFloat16) {
    if (x.size(0) >= 4) {
      launch_iq1_s_matvec_tiled<sycl::ext::oneapi::bfloat16>(x, qweight, table,
                                                             output);
    } else {
      launch_iq1_s_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, table,
                                                       output);
    }
  } else {
    TORCH_CHECK(false, "iq1_s_matvec supports float32 and bfloat16 inputs");
  }
  return output;
}

template <typename scalar_t>
void launch_iq1_m_matvec(const torch::Tensor &x,
                         const torch::Tensor &qweight,
                         const torch::Tensor &table,
                         torch::Tensor &output) {
  constexpr int64_t kBlockSize = 256;
  constexpr int64_t kBlockBytes = 56;
  constexpr int64_t kWorkgroupSize = 256;
  const int64_t batch = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t out_features = qweight.size(0);
  const int64_t blocks_per_row = in_features / kBlockSize;
  const auto *x_ptr = reinterpret_cast<const scalar_t *>(x.data_ptr());
  const auto *weight_ptr = qweight.data_ptr<uint8_t>();
  const auto *table_ptr = table.data_ptr<float>();
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
          const int64_t subblock = in_block / 32;
          const int64_t position = in_block % 32;
          const int64_t part = position / 8;
          const int64_t lane = position % 8;
          const int qh_offset = 32 + subblock * 2;
          const int high_byte = packed[qh_offset + part / 2];
          const int high_shift = (part % 2) * 4;
          const int q_index = packed[subblock * 4 + part] |
                              (((high_byte >> high_shift) & 0x07) << 8);
          const int delta_bit = part % 2 == 0 ? 0x08 : 0x80;
          const int delta_byte = packed[qh_offset + part / 2];
          const float delta = (delta_byte & delta_bit) == 0 ? 0.125f : -0.125f;
          const int scale_index = subblock * 2 + part / 2;
          const int scale_word_index = scale_index / 4;
          const int scale_word_byte = 48 + scale_word_index * 2;
          const uint16_t scale_word = static_cast<uint16_t>(packed[scale_word_byte]) |
              (static_cast<uint16_t>(packed[scale_word_byte + 1]) << 8);
          const int scale_shift = (scale_index % 4) * 3;
          const int scale_value = (scale_word >> scale_shift) & 0x07;

          const uint16_t d_bits =
              ((static_cast<uint16_t>(packed[48]) |
                (static_cast<uint16_t>(packed[49]) << 8)) >> 12) |
              ((((static_cast<uint16_t>(packed[50]) |
                  (static_cast<uint16_t>(packed[51]) << 8)) >> 8) & 0x00F0)) |
              ((((static_cast<uint16_t>(packed[52]) |
                  (static_cast<uint16_t>(packed[53]) << 8)) >> 4) & 0x0F00)) |
              ((static_cast<uint16_t>(packed[54]) |
                (static_cast<uint16_t>(packed[55]) << 8)) & 0xF000);
          const float d = static_cast<float>(
              *reinterpret_cast<const sycl::half *>(&d_bits));
          const float scale = d * (2.0f * static_cast<float>(scale_value) + 1.0f);
          const float value = table_ptr[q_index * 8 + lane] + delta;
          partial += static_cast<float>(x_ptr[token * in_features + index]) *
                     (scale * value);
        }
        const float sum = sycl::reduce_over_group(item.get_group(), partial,
                                                   sycl::plus<float>());
        if (item.get_local_id(0) == 0) {
          out_ptr[group] = static_cast<scalar_t>(sum);
        }
      });
}

torch::Tensor iq1_m_matvec(torch::Tensor x, torch::Tensor qweight,
                           torch::Tensor table) {
  TORCH_CHECK(x.device().is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(qweight.device() == x.device() && table.device() == x.device(),
              "IQ1_M inputs must share an XPU device");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2,
              "x and qweight must be rank-2 tensors");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() && table.is_contiguous(),
              "IQ1_M inputs must be contiguous");
  TORCH_CHECK(qweight.scalar_type() == torch::kUInt8 &&
                  table.scalar_type() == torch::kFloat32,
              "IQ1_M inputs have invalid dtypes");
  TORCH_CHECK(x.size(1) % 256 == 0,
              "IQ1_M input width must be divisible by 256");
  TORCH_CHECK(qweight.size(1) == x.size(1) / 256 * 56,
              "qweight has invalid IQ1_M row geometry");
  TORCH_CHECK(table.numel() == 2048 * 8,
              "IQ1_M lookup table has invalid size");
  auto output = torch::empty({x.size(0), qweight.size(0)}, x.options());
  if (x.scalar_type() == torch::kFloat32) {
    launch_iq1_m_matvec<float>(x, qweight, table, output);
  } else if (x.scalar_type() == torch::kBFloat16) {
    launch_iq1_m_matvec<sycl::ext::oneapi::bfloat16>(x, qweight, table, output);
  } else {
    TORCH_CHECK(false, "iq1_m_matvec supports float32 and bfloat16 inputs");
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
  module.def("iq3_xxs_matvec", &iq3_xxs_matvec,
             "SYCL IQ3_XXS matrix-vector product");
  module.def("iq2_s_matvec", &iq2_s_matvec, "SYCL IQ2_S matrix-vector product");
  module.def("iq3_s_matvec", &iq3_s_matvec, "SYCL IQ3_S matrix-vector product");
  module.def("iq2_xxs_matvec", &iq2_xxs_matvec,
             "SYCL IQ2_XXS matrix-vector product");
  module.def("iq2_xs_matvec", &iq2_xs_matvec, "SYCL IQ2_XS matrix-vector product");
  module.def("iq4_xs_matvec", &iq4_xs_matvec, "SYCL IQ4_XS matrix-vector product");
  module.def("iq1_s_matvec", &iq1_s_matvec, "SYCL IQ1_S matrix-vector product");
  module.def("iq1_m_matvec", &iq1_m_matvec, "SYCL IQ1_M matrix-vector product");
}
