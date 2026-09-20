#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(call)                                                         \
  do {                                                                         \
    VkResult result = (call);                                                  \
    if (result != VK_SUCCESS)                                                  \
      throw std::runtime_error(#call " failed: " +                             \
                               std::to_string(static_cast<int>(result)));      \
  } while (false)

struct Buffer {
  VkBuffer handle{};
  VkDeviceMemory memory{};
  VkDeviceSize size{};
  uint32_t memory_type_index{};
  VkMemoryPropertyFlags memory_properties{};
  void *mapped{};
};

struct Dimensions {
  uint32_t rows;
  uint32_t input_size;
  uint32_t output_size;
};

struct CooperativeDimensions {
  uint32_t rows;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t padded_input_size;
  uint32_t padded_output_size;
};

struct CooperativeTile {
  uint32_t m;
  uint32_t n;
  uint32_t k;
};

static std::vector<uint32_t> read_spirv(const char *path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error(std::string("cannot open SPIR-V file: ") + path);
  auto size = static_cast<size_t>(stream.tellg());
  if (size == 0 || size % sizeof(uint32_t) != 0)
    throw std::runtime_error("SPIR-V file has an invalid size");
  std::vector<uint32_t> words(size / sizeof(uint32_t));
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(words.data()),
              static_cast<std::streamsize>(size));
  return words;
}

static uint32_t find_memory_type(VkPhysicalDevice physical, uint32_t bits,
                                 VkMemoryPropertyFlags required,
                                 VkMemoryPropertyFlags preferred,
                                 uint32_t *selected_properties) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical, &properties);
  uint32_t fallback = UINT32_MAX;
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    if (!(bits & (1u << i)))
      continue;
    const auto flags = properties.memoryTypes[i].propertyFlags;
    if ((flags & required) != required)
      continue;
    if (fallback == UINT32_MAX)
      fallback = i;
    if ((flags & preferred) == preferred) {
      *selected_properties = flags;
      return i;
    }
  }
  if (fallback != UINT32_MAX) {
    *selected_properties = properties.memoryTypes[fallback].propertyFlags;
    return fallback;
  }
  throw std::runtime_error("no compatible host-visible Vulkan memory type");
}

static bool has_device_extension(VkPhysicalDevice physical, const char *name) {
  uint32_t count = 0;
  VK_CHECK(
      vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr));
  std::vector<VkExtensionProperties> extensions(count);
  VK_CHECK(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count,
                                                extensions.data()));
  return std::any_of(extensions.begin(), extensions.end(),
                     [name](const auto &ext) {
                       return std::strcmp(ext.extensionName, name) == 0;
                     });
}

static std::vector<VkCooperativeMatrixPropertiesKHR>
cooperative_matrix_properties(VkInstance instance, VkPhysicalDevice physical) {
  auto query =
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vkGetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (!query ||
      !has_device_extension(physical, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
    return {};

  uint32_t count = 0;
  VK_CHECK(query(physical, &count, nullptr));
  std::vector<VkCooperativeMatrixPropertiesKHR> properties(count);
  for (auto &property : properties)
    property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  if (count != 0)
    VK_CHECK(query(physical, &count, properties.data()));
  return properties;
}

static const char *component_type_name(VkComponentTypeKHR type) {
  switch (type) {
  case VK_COMPONENT_TYPE_FLOAT16_KHR:
    return "float16";
  case VK_COMPONENT_TYPE_FLOAT32_KHR:
    return "float32";
  case VK_COMPONENT_TYPE_FLOAT64_KHR:
    return "float64";
  case VK_COMPONENT_TYPE_SINT8_KHR:
    return "int8";
  case VK_COMPONENT_TYPE_SINT16_KHR:
    return "int16";
  case VK_COMPONENT_TYPE_SINT32_KHR:
    return "int32";
  case VK_COMPONENT_TYPE_UINT8_KHR:
    return "uint8";
  case VK_COMPONENT_TYPE_UINT16_KHR:
    return "uint16";
  case VK_COMPONENT_TYPE_UINT32_KHR:
    return "uint32";
  default:
    return "other";
  }
}

static Buffer make_buffer(VkDevice device, VkPhysicalDevice physical,
                          VkDeviceSize size) {
  Buffer buffer{.size = size};
  VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  create.size = size;
  create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &create, nullptr, &buffer.handle));

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device, buffer.handle, &requirements);
  VkMemoryPropertyFlags memory_properties = 0;
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex =
      find_memory_type(physical, requirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       &memory_properties);
  buffer.memory_type_index = allocation.memoryTypeIndex;
  buffer.memory_properties = memory_properties;
  VK_CHECK(vkAllocateMemory(device, &allocation, nullptr, &buffer.memory));
  VK_CHECK(vkBindBufferMemory(device, buffer.handle, buffer.memory, 0));
  VK_CHECK(
      vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped));
  return buffer;
}

static void destroy_buffer(VkDevice device, Buffer &buffer) {
  if (buffer.mapped)
    vkUnmapMemory(device, buffer.memory);
  if (buffer.handle)
    vkDestroyBuffer(device, buffer.handle, nullptr);
  if (buffer.memory)
    vkFreeMemory(device, buffer.memory, nullptr);
}

template <typename T>
static double upload(VkDevice device, const Buffer &buffer,
                     const std::vector<T> &data) {
  auto started = std::chrono::steady_clock::now();
  std::memcpy(buffer.mapped, data.data(), data.size() * sizeof(T));
  if (!(buffer.memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    VK_CHECK(vkFlushMappedMemoryRanges(device, 1, &range));
  }
  auto ended = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(ended - started).count();
}

static void invalidate_if_needed(VkDevice device, const Buffer &buffer) {
  if (buffer.memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    return;
  VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
  range.memory = buffer.memory;
  range.offset = 0;
  range.size = VK_WHOLE_SIZE;
  VK_CHECK(vkInvalidateMappedMemoryRanges(device, 1, &range));
}

static uint16_t float_to_half(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t sign = (bits >> 16) & 0x8000;
  const uint32_t source_exponent = (bits >> 23) & 0xff;
  uint32_t mantissa = bits & 0x7fffff;
  if (source_exponent == 0xff)
    return static_cast<uint16_t>(sign | 0x7c00 | (mantissa ? 0x0200 : 0));

  int exponent = static_cast<int>(source_exponent) - 127 + 15;
  if (exponent <= 0) {
    if (exponent < -10)
      return static_cast<uint16_t>(sign);
    mantissa |= 0x800000;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u)))
      ++rounded;
    return static_cast<uint16_t>(sign | rounded);
  }
  if (exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7c00);

  uint32_t rounded = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder == 0x1000 && (rounded & 1u))) {
    ++rounded;
    if (rounded == 0x400) {
      rounded = 0;
      ++exponent;
      if (exponent >= 31)
        return static_cast<uint16_t>(sign | 0x7c00);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               rounded);
}

static float half_to_float(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 0x1f;
  uint32_t mantissa = value & 0x03ff;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0)
      bits = sign;
    else {
      int unbiased = -14;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --unbiased;
      }
      mantissa &= 0x03ff;
      bits = sign | (static_cast<uint32_t>(unbiased + 127) << 23) |
             (mantissa << 13);
    }
  } else if (exponent == 0x1f) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(bits);
}

static std::vector<uint16_t> read_half_values(const char *path,
                                              size_t expected_count) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error(std::string("cannot open FP16 tensor: ") + path);
  const auto byte_count = stream.tellg();
  const size_t expected_bytes = expected_count * sizeof(uint16_t);
  if (byte_count < 0 || static_cast<uint64_t>(byte_count) != expected_bytes)
    throw std::runtime_error(std::string("FP16 tensor has an invalid size: ") +
                             path);
  std::vector<uint16_t> values(expected_count);
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(values.data()),
              static_cast<std::streamsize>(expected_bytes));
  if (!stream)
    throw std::runtime_error(std::string("cannot read FP16 tensor: ") + path);
  return values;
}

static double write_float_values(const char *path,
                                const std::vector<float> &values) {
  auto started = std::chrono::steady_clock::now();
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    throw std::runtime_error(std::string("cannot create FP32 output: ") + path);
  stream.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!stream)
    throw std::runtime_error(std::string("cannot write FP32 output: ") + path);
  stream.close();
  if (stream.fail())
    throw std::runtime_error(std::string("cannot close FP32 output: ") + path);
  auto ended = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(ended - started).count();
}

static double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const size_t middle = samples.size() / 2;
  if (samples.size() % 2 != 0)
    return samples[middle];
  return (samples[middle - 1] + samples[middle]) / 2.0;
}

static uint32_t parse_dimension(const char *value) {
  size_t consumed = 0;
  auto parsed = std::stoul(value, &consumed);
  if (consumed != std::strlen(value) || parsed == 0 || parsed > UINT32_MAX)
    throw std::runtime_error(
        "dimensions and iteration count must be positive integers");
  return static_cast<uint32_t>(parsed);
}

int main(int argc, char **argv) try {
  const auto probe_started = std::chrono::steady_clock::now();
  const bool raw_tensor_mode = argc == 12;
  if (argc != 5 && argc != 9 && !raw_tensor_mode) {
    std::cerr << "usage: vulkan_dense_probe [naive|tiled|cooperative|best] "
                 "NAIVE_SPV TILED_SPV COOPERATIVE_SPV "
                 "[ROWS INPUT_SIZE OUTPUT_SIZE ITERATIONS "
                 "[INPUT_F16 WEIGHT_F16 OUTPUT_F32]]\n";
    return 2;
  }
  if (raw_tensor_mode && std::endian::native != std::endian::little)
    throw std::runtime_error("raw tensor mode requires a little-endian host");
  const std::string requested_kernel = argv[1];
  if (requested_kernel != "naive" && requested_kernel != "tiled" &&
      requested_kernel != "cooperative" && requested_kernel != "best")
    throw std::runtime_error(
        "kernel must be naive, tiled, cooperative, or best");
  Dimensions dims{8, 256, 512};
  uint32_t iterations = 20;
  if (argc >= 9) {
    dims.rows = parse_dimension(argv[5]);
    dims.input_size = parse_dimension(argv[6]);
    dims.output_size = parse_dimension(argv[7]);
    iterations = parse_dimension(argv[8]);
  }

  VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.pApplicationName = "FreeToken Vulkan dense probe";
  application.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo instance_create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance_create.pApplicationInfo = &application;
  VkInstance instance{};
  VK_CHECK(vkCreateInstance(&instance_create, nullptr, &instance));

  uint32_t physical_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &physical_count, nullptr));
  std::vector<VkPhysicalDevice> physical_devices(physical_count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &physical_count,
                                      physical_devices.data()));

  VkPhysicalDevice physical{};
  VkPhysicalDeviceProperties physical_properties{};
  uint32_t queue_family = UINT32_MAX;
  for (auto candidate : physical_devices) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(candidate, &properties);
    if (properties.vendorID != 0x8086 ||
        properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      continue;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(count);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, queues.data());
    for (uint32_t i = 0; i < count; ++i) {
      if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        physical = candidate;
        physical_properties = properties;
        queue_family = i;
        break;
      }
    }
    if (physical)
      break;
  }
  if (!physical)
    throw std::runtime_error("no Intel discrete Vulkan compute device found");

  auto matrix_properties = cooperative_matrix_properties(instance, physical);
  VkPhysicalDeviceMemoryProperties memory_properties{};
  vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
  bool host_visible_device_local = false;
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    const auto flags = memory_properties.memoryTypes[i].propertyFlags;
    host_visible_device_local |=
        (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) ==
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }
  VkPhysicalDeviceSubgroupProperties subgroup_properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 properties2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties2.pNext = &subgroup_properties;
  vkGetPhysicalDeviceProperties2(physical, &properties2);

  bool has_cooperative_matrix_extension =
      has_device_extension(physical, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR supported_matrix_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceVulkan11Features supported_vulkan11_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features supported_vulkan12_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  supported_matrix_features.pNext = &supported_vulkan11_features;
  supported_vulkan11_features.pNext = &supported_vulkan12_features;
  VkPhysicalDeviceFeatures2 supported_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  supported_features.pNext = &supported_matrix_features;
  if (has_cooperative_matrix_extension)
    vkGetPhysicalDeviceFeatures2(physical, &supported_features);

  CooperativeTile cooperative_tile{};
  for (const auto &property : matrix_properties) {
    if (property.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
        property.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
        property.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        property.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        property.scope == VK_SCOPE_SUBGROUP_KHR) {
      cooperative_tile = {property.MSize, property.NSize, property.KSize};
      break;
    }
  }
  const bool cooperative_matrix_runtime_supported =
      has_cooperative_matrix_extension && !matrix_properties.empty() &&
      supported_matrix_features.cooperativeMatrix &&
      supported_vulkan12_features.shaderFloat16 &&
      supported_vulkan11_features.storageBuffer16BitAccess &&
      supported_vulkan12_features.vulkanMemoryModel &&
      subgroup_properties.subgroupSize == 32 && cooperative_tile.m != 0;
  const bool cooperative_shader_available = std::string(argv[4]) != "-";
  const bool cooperative_available =
      cooperative_matrix_runtime_supported && cooperative_shader_available;
  std::string kernel = requested_kernel;
  if (kernel == "best")
    kernel = cooperative_available ? "cooperative" : "naive";
  if (kernel == "cooperative" && !cooperative_available)
    throw std::runtime_error("no compatible subgroup FP16-to-FP32 "
                             "cooperative-matrix path is available");
  const std::string weight_layout = kernel == "naive" ? "row" : "column";
  const char *shader_path = kernel == "naive"         ? argv[2]
                            : kernel == "cooperative" ? argv[4]
                                                      : argv[3];

  const auto device_init_started = std::chrono::steady_clock::now();
  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_create{
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_create.queueFamilyIndex = queue_family;
  queue_create.queueCount = 1;
  queue_create.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_create.queueCreateInfoCount = 1;
  device_create.pQueueCreateInfos = &queue_create;
  VkPhysicalDeviceVulkan11Features enabled_vulkan11_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features enabled_vulkan12_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR enabled_matrix_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  enabled_matrix_features.pNext = &enabled_vulkan11_features;
  enabled_vulkan11_features.pNext = &enabled_vulkan12_features;
  enabled_matrix_features.cooperativeMatrix = VK_TRUE;
  enabled_vulkan11_features.storageBuffer16BitAccess = VK_TRUE;
  enabled_vulkan12_features.shaderFloat16 = VK_TRUE;
  enabled_vulkan12_features.vulkanMemoryModel = VK_TRUE;
  const char *cooperative_extension = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
  if (kernel == "cooperative") {
    device_create.pNext = &enabled_matrix_features;
    device_create.enabledExtensionCount = 1;
    device_create.ppEnabledExtensionNames = &cooperative_extension;
  }
  VkDevice device{};
  VK_CHECK(vkCreateDevice(physical, &device_create, nullptr, &device));
  VkQueue queue{};
  vkGetDeviceQueue(device, queue_family, 0, &queue);
  const auto device_init_ended = std::chrono::steady_clock::now();
  const double device_init_ms =
      std::chrono::duration<double, std::milli>(device_init_ended -
                                                device_init_started)
          .count();

  std::vector<float> input(dims.rows * dims.input_size);
  std::vector<float> weight(dims.output_size * dims.input_size);
  if (raw_tensor_mode) {
    const auto raw_input = read_half_values(argv[9], input.size());
    const auto raw_weight = read_half_values(argv[10], weight.size());
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = half_to_float(raw_input[i]);
      if (!std::isfinite(input[i]))
        throw std::runtime_error("activation contains a non-finite FP16 value");
    }
    for (size_t i = 0; i < weight.size(); ++i) {
      weight[i] = half_to_float(raw_weight[i]);
      if (!std::isfinite(weight[i]))
        throw std::runtime_error("weight contains a non-finite FP16 value");
    }
  } else {
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 17.0f;
    for (size_t i = 0; i < weight.size(); ++i)
      weight[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 13.0f;
  }
  const auto round_up = [](uint32_t value, uint32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
  };
  const uint32_t padded_rows = kernel == "cooperative"
                                   ? round_up(dims.rows, cooperative_tile.m)
                                   : dims.rows;
  const uint32_t padded_input_size =
      kernel == "cooperative" ? round_up(dims.input_size, cooperative_tile.k)
                              : dims.input_size;
  const uint32_t padded_output_size =
      kernel == "cooperative" ? round_up(dims.output_size, cooperative_tile.n)
                              : dims.output_size;
  std::vector<float> shader_weight;
  std::vector<uint16_t> half_input;
  std::vector<uint16_t> half_weight;
  if (kernel == "cooperative") {
    half_input.resize(static_cast<size_t>(padded_rows) * padded_input_size);
    half_weight.resize(static_cast<size_t>(padded_input_size) *
                       padded_output_size);
    for (uint32_t row = 0; row < dims.rows; ++row)
      for (uint32_t k = 0; k < dims.input_size; ++k)
        half_input[static_cast<size_t>(row) * padded_input_size + k] =
            float_to_half(
                input[static_cast<size_t>(row) * dims.input_size + k]);
    for (uint32_t output = 0; output < dims.output_size; ++output)
      for (uint32_t k = 0; k < dims.input_size; ++k)
        half_weight[static_cast<size_t>(k) * padded_output_size + output] =
            float_to_half(
                weight[static_cast<size_t>(output) * dims.input_size + k]);
  } else if (weight_layout == "row") {
    shader_weight = weight;
  } else {
    shader_weight.resize(weight.size());
    for (uint32_t output = 0; output < dims.output_size; ++output)
      for (uint32_t k = 0; k < dims.input_size; ++k)
        shader_weight[static_cast<size_t>(k) * dims.output_size + output] =
            weight[static_cast<size_t>(output) * dims.input_size + k];
  }
  std::vector<float> expected(dims.rows * dims.output_size, 0.0f);
  const auto reference_started = std::chrono::steady_clock::now();
  for (uint32_t row = 0; row < dims.rows; ++row)
    for (uint32_t column = 0; column < dims.output_size; ++column)
      for (uint32_t k = 0; k < dims.input_size; ++k)
        expected[row * dims.output_size + column] +=
            (kernel == "cooperative"
                 ? half_to_float(
                       half_input[static_cast<size_t>(row) * padded_input_size +
                                  k])
                 : input[static_cast<size_t>(row) * dims.input_size + k]) *
            (kernel == "cooperative"
                 ? half_to_float(
                       half_weight[static_cast<size_t>(k) * padded_output_size +
                                   column])
                 : weight[static_cast<size_t>(column) * dims.input_size + k]);
  const auto reference_ended = std::chrono::steady_clock::now();
  const double cpu_reference_ms =
      std::chrono::duration<double, std::milli>(reference_ended -
                                                reference_started)
          .count();
  if (!std::all_of(expected.begin(), expected.end(),
                   [](float value) { return std::isfinite(value); }))
    throw std::runtime_error("CPU reference contains a non-finite value");

  const size_t input_bytes = kernel == "cooperative"
                                 ? half_input.size() * sizeof(uint16_t)
                                 : input.size() * sizeof(float);
  const size_t weight_bytes = kernel == "cooperative"
                                  ? half_weight.size() * sizeof(uint16_t)
                                  : shader_weight.size() * sizeof(float);
  const size_t output_bytes = kernel == "cooperative"
                                  ? static_cast<size_t>(padded_rows) *
                                        padded_output_size * sizeof(float)
                                  : expected.size() * sizeof(float);
  const auto vulkan_setup_started = std::chrono::steady_clock::now();
  Buffer input_buffer = make_buffer(device, physical, input_bytes);
  Buffer weight_buffer = make_buffer(device, physical, weight_bytes);
  Buffer output_buffer = make_buffer(device, physical, output_bytes);
  const double weight_upload_ms =
      kernel == "cooperative" ? upload(device, weight_buffer, half_weight)
                              : upload(device, weight_buffer, shader_weight);
  const auto upload_input = [&]() {
    return kernel == "cooperative" ? upload(device, input_buffer, half_input)
                                   : upload(device, input_buffer, input);
  };
  upload_input();

  VkDescriptorSetLayoutBinding bindings[3]{};
  for (uint32_t i = 0; i < 3; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo layout_create{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout_create.bindingCount = 3;
  layout_create.pBindings = bindings;
  VkDescriptorSetLayout descriptor_layout{};
  VK_CHECK(vkCreateDescriptorSetLayout(device, &layout_create, nullptr,
                                       &descriptor_layout));

  VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(CooperativeDimensions)};
  VkPipelineLayoutCreateInfo pipeline_layout_create{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_layout_create.setLayoutCount = 1;
  pipeline_layout_create.pSetLayouts = &descriptor_layout;
  pipeline_layout_create.pushConstantRangeCount = 1;
  pipeline_layout_create.pPushConstantRanges = &push_range;
  VkPipelineLayout pipeline_layout{};
  VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_create, nullptr,
                                  &pipeline_layout));

  auto spirv = read_spirv(shader_path);
  VkShaderModuleCreateInfo shader_create{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  shader_create.codeSize = spirv.size() * sizeof(uint32_t);
  shader_create.pCode = spirv.data();
  VkShaderModule shader{};
  VK_CHECK(vkCreateShaderModule(device, &shader_create, nullptr, &shader));
  VkComputePipelineCreateInfo pipeline_create{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_create.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  pipeline_create.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_create.stage.module = shader;
  pipeline_create.stage.pName = "main";
  pipeline_create.layout = pipeline_layout;
  std::array<int, 3> specialization_data{static_cast<int>(cooperative_tile.m),
                                         static_cast<int>(cooperative_tile.n),
                                         static_cast<int>(cooperative_tile.k)};
  VkSpecializationMapEntry specialization_entries[3] = {
      {0, 0, sizeof(int)},
      {1, sizeof(int), sizeof(int)},
      {2, sizeof(int) * 2, sizeof(int)},
  };
  VkSpecializationInfo specialization_info{};
  specialization_info.mapEntryCount = 3;
  specialization_info.pMapEntries = specialization_entries;
  specialization_info.dataSize = sizeof(specialization_data);
  specialization_info.pData = specialization_data.data();
  if (kernel == "cooperative")
    pipeline_create.stage.pSpecializationInfo = &specialization_info;
  VkPipeline pipeline{};
  VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_create,
                                    nullptr, &pipeline));

  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo pool_create{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_create.maxSets = 1;
  pool_create.poolSizeCount = 1;
  pool_create.pPoolSizes = &pool_size;
  VkDescriptorPool pool{};
  VK_CHECK(vkCreateDescriptorPool(device, &pool_create, nullptr, &pool));
  VkDescriptorSetAllocateInfo set_allocate{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  set_allocate.descriptorPool = pool;
  set_allocate.descriptorSetCount = 1;
  set_allocate.pSetLayouts = &descriptor_layout;
  VkDescriptorSet descriptor_set{};
  VK_CHECK(vkAllocateDescriptorSets(device, &set_allocate, &descriptor_set));
  VkDescriptorBufferInfo buffer_infos[3] = {
      {input_buffer.handle, 0, input_buffer.size},
      {weight_buffer.handle, 0, weight_buffer.size},
      {output_buffer.handle, 0, output_buffer.size},
  };
  VkWriteDescriptorSet writes[3]{};
  for (uint32_t i = 0; i < 3; ++i) {
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[i].dstSet = descriptor_set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &buffer_infos[i];
  }
  vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

  VkCommandPoolCreateInfo command_pool_create{
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  command_pool_create.queueFamilyIndex = queue_family;
  VkCommandPool command_pool{};
  VK_CHECK(vkCreateCommandPool(device, &command_pool_create, nullptr,
                               &command_pool));
  VkCommandBufferAllocateInfo command_allocate{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command_allocate.commandPool = command_pool;
  command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_allocate.commandBufferCount = 1;
  VkCommandBuffer command{};
  VK_CHECK(vkAllocateCommandBuffers(device, &command_allocate, &command));
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK_CHECK(vkBeginCommandBuffer(command, &begin));
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
  if (kernel == "cooperative") {
    const CooperativeDimensions cooperative_dims{
        dims.rows, dims.input_size, dims.output_size, padded_input_size,
        padded_output_size};
    vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(cooperative_dims), &cooperative_dims);
    vkCmdDispatch(command,
                  (dims.output_size + cooperative_tile.n - 1) /
                      cooperative_tile.n,
                  (dims.rows + cooperative_tile.m - 1) / cooperative_tile.m, 1);
  } else {
    vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(Dimensions), &dims);
    vkCmdDispatch(command, (dims.output_size + 15) / 16, (dims.rows + 15) / 16,
                  1);
  }
  VK_CHECK(vkEndCommandBuffer(command));
  const auto vulkan_setup_ended = std::chrono::steady_clock::now();
  const double vulkan_setup_ms =
      std::chrono::duration<double, std::milli>(vulkan_setup_ended -
                                                vulkan_setup_started)
          .count();

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command;
  constexpr uint32_t warmup_iterations = 3;
  for (uint32_t i = 0; i < warmup_iterations; ++i) {
    upload_input();
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
  }
  std::vector<double> input_upload_samples;
  std::vector<double> dispatch_samples;
  std::vector<double> upload_dispatch_samples;
  for (uint32_t i = 0; i < iterations; ++i) {
    auto request_started = std::chrono::steady_clock::now();
    input_upload_samples.push_back(upload_input());
    auto dispatch_started = std::chrono::steady_clock::now();
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    auto dispatch_ended = std::chrono::steady_clock::now();
    upload_dispatch_samples.push_back(std::chrono::duration<double, std::milli>(
                                          dispatch_ended - request_started)
                                          .count());
    dispatch_samples.push_back(std::chrono::duration<double, std::milli>(
                                   dispatch_ended - dispatch_started)
                                   .count());
  }

  const auto readback_started = std::chrono::steady_clock::now();
  invalidate_if_needed(device, output_buffer);
  auto *output = static_cast<const float *>(output_buffer.mapped);
  std::vector<float> result(expected.size());
  float max_error = 0.0f;
  for (uint32_t row = 0; row < dims.rows; ++row)
    for (uint32_t column = 0; column < dims.output_size; ++column) {
      const size_t expected_index =
          static_cast<size_t>(row) * dims.output_size + column;
      const size_t output_index =
          kernel == "cooperative"
              ? static_cast<size_t>(row) * padded_output_size + column
              : expected_index;
      result[expected_index] = output[output_index];
      if (!std::isfinite(result[expected_index]))
        throw std::runtime_error("Vulkan output contains a non-finite value");
      max_error = std::max(max_error,
                           std::abs(result[expected_index] - expected[expected_index]));
    }
  const auto readback_ended = std::chrono::steady_clock::now();
  const double output_readback_ms =
      std::chrono::duration<double, std::milli>(readback_ended - readback_started)
          .count();
  const double output_file_write_ms =
      raw_tensor_mode ? write_float_values(argv[11], result) : 0.0;
  const double probe_elapsed_ms =
      std::chrono::duration<double, std::milli>(readback_ended - probe_started)
          .count();

  std::cout
      << "{\n"
      << "  \"device\": \"" << physical_properties.deviceName << "\",\n"
      << "  \"vendor_id\": " << physical_properties.vendorID << ",\n"
      << "  \"device_id\": " << physical_properties.deviceID << ",\n"
      << "  \"kernel\": \""
      << (kernel == "naive"   ? "naive_fp32"
          : kernel == "tiled" ? "transposed_gemv_fp32"
                              : "cooperative_fp16_fp32_acc")
      << "\",\n"
      << "  \"raw_tensor_mode\": " << (raw_tensor_mode ? "true" : "false")
      << ",\n"
      << "  \"weight_layout\": \"" << weight_layout << "\",\n"
      << "  \"buffer_memory_type_index\": " << input_buffer.memory_type_index
      << ",\n"
      << "  \"buffer_memory_device_local\": "
      << ((input_buffer.memory_properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
              ? "true"
              : "false")
      << ",\n"
      << "  \"subgroup_size\": " << subgroup_properties.subgroupSize << ",\n"
      << "  \"host_visible_device_local_memory\": "
      << (host_visible_device_local ? "true" : "false") << ",\n"
      << "  \"cooperative_matrix_supported\": "
      << (cooperative_matrix_runtime_supported ? "true" : "false") << ",\n"
      << "  \"cooperative_shader_available\": "
      << (cooperative_shader_available ? "true" : "false") << ",\n"
      << "  \"cooperative_matrix_tile\": [" << cooperative_tile.m << ", "
      << cooperative_tile.n << ", " << cooperative_tile.k << "],\n"
      << "  \"cooperative_matrix_properties\": [";
  for (size_t i = 0; i < matrix_properties.size(); ++i) {
    const auto &property = matrix_properties[i];
    std::cout << (i == 0 ? "\n" : ",\n") << "    {\"m\": " << property.MSize
              << ", \"n\": " << property.NSize << ", \"k\": " << property.KSize
              << ", \"a\": \"" << component_type_name(property.AType)
              << "\", \"b\": \"" << component_type_name(property.BType)
              << "\", \"c\": \"" << component_type_name(property.CType)
              << "\", \"result\": \""
              << component_type_name(property.ResultType)
              << "\", \"scope\": " << property.scope << "}";
  }
  if (!matrix_properties.empty())
    std::cout << "\n  ";
  std::cout << "],\n"
            << "  \"rows\": " << dims.rows << ",\n"
            << "  \"input_size\": " << dims.input_size << ",\n"
            << "  \"output_size\": " << dims.output_size << ",\n"
            << "  \"iterations\": " << iterations << ",\n"
            << "  \"device_init_ms\": " << device_init_ms << ",\n"
            << "  \"vulkan_setup_ms\": " << vulkan_setup_ms << ",\n"
            << "  \"cpu_reference_ms\": " << cpu_reference_ms << ",\n"
            << "  \"weight_upload_ms\": " << weight_upload_ms << ",\n"
            << "  \"median_input_upload_ms\": " << median(input_upload_samples)
            << ",\n"
            << "  \"median_dispatch_ms\": " << median(dispatch_samples) << ",\n"
            << "  \"median_upload_plus_dispatch_ms\": "
            << median(upload_dispatch_samples) << ",\n"
            << "  \"output_readback_ms\": " << output_readback_ms << ",\n"
            << "  \"output_file_write_ms\": " << output_file_write_ms << ",\n"
            << "  \"probe_elapsed_before_output_file_write_ms\": "
            << probe_elapsed_ms << ",\n"
            << "  \"max_abs_error\": " << max_error << "\n"
            << "}\n";

  vkDestroyCommandPool(device, command_pool, nullptr);
  vkDestroyDescriptorPool(device, pool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyShaderModule(device, shader, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
  destroy_buffer(device, output_buffer);
  destroy_buffer(device, weight_buffer);
  destroy_buffer(device, input_buffer);
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  const float max_allowed_error = kernel == "cooperative" ? 1e-2f : 1e-3f;
  return max_error <= max_allowed_error ? 0 : 1;
} catch (const std::exception &error) {
  std::cerr << "vulkan_dense_probe: " << error.what() << '\n';
  return 1;
}
