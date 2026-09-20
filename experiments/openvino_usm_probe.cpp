#include <torch/extension.h>

#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

#include <string>
#include <vector>

pybind11::dict probe_xpu_usm(torch::Tensor input) {
  if (input.device().type() != c10::DeviceType::XPU) {
    throw std::invalid_argument("input must be an XPU tensor");
  }
  if (input.scalar_type() != at::kFloat || input.dim() != 2 ||
      input.size(0) != 2 || input.size(1) != 8 || !input.is_contiguous()) {
    throw std::invalid_argument(
        "input must be contiguous FP32 with shape [2, 8]");
  }

  ov::Core core;
  auto context =
      core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
  const auto context_type =
      context.get_params().at("CONTEXT_TYPE").as<std::string>();

  auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape{2, 8});
  auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{},
                                            std::vector<float>{2.0f});
  auto doubled = std::make_shared<ov::op::v1::Multiply>(parameter, scale);
  auto result_node = std::make_shared<ov::op::v0::Result>(doubled);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{result_node},
                                           ov::ParameterVector{parameter},
                                           "xpu_usm_interop_probe");
  auto compiled = core.compile_model(model, context);
  const ov::Any device_property = compiled.get_property("EXECUTION_DEVICES");
  const auto devices = device_property.as<std::vector<std::string>>();
  if (devices.empty()) {
    throw std::runtime_error("OpenVINO reported no execution device");
  }
  for (const auto &device : devices) {
    if (device.rfind("GPU", 0) != 0) {
      throw std::runtime_error("OpenVINO used a non-GPU device: " + device);
    }
  }

  ov::intel_gpu::ocl::USMTensor remote_input;
  try {
    remote_input = context.create_tensor(ov::element::f32, ov::Shape{2, 8},
                                         input.data_ptr());
  } catch (const std::exception &error) {
    throw std::runtime_error(
        "OpenVINO context type " + context_type +
        " rejected the PyTorch XPU pointer: " + error.what());
  }
  const bool same_pointer = remote_input.get() == input.data_ptr();
  if (!same_pointer) {
    throw std::runtime_error(
        "OpenVINO RemoteTensor did not retain the XPU pointer");
  }

  auto request = compiled.create_infer_request();
  request.set_input_tensor(remote_input);
  request.infer();

  ov::Tensor host_output(ov::element::f32, ov::Shape{2, 8});
  request.get_output_tensor().copy_to(host_output);
  const auto *data = host_output.data<const float>();
  std::vector<float> values(data, data + 16);

  pybind11::dict result;
  result["context_type"] = context_type;
  result["execution_devices"] = devices;
  result["pointer_identity"] = same_pointer;
  result["output"] = values;
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("probe_xpu_usm", &probe_xpu_usm);
}
