#include <vulkan/vulkan.h>

#include <algorithm>
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
    do {                                                                       \
        VkResult result = (call);                                               \
        if (result != VK_SUCCESS)                                               \
            throw std::runtime_error(#call " failed: " +                       \
                                     std::to_string(static_cast<int>(result))); \
    } while (false)

struct Buffer {
    VkBuffer handle{};
    VkDeviceMemory memory{};
    VkDeviceSize size{};
};

struct Dimensions {
    uint32_t rows;
    uint32_t input_size;
    uint32_t output_size;
};

static std::vector<uint32_t> read_spirv(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::runtime_error(std::string("cannot open SPIR-V file: ") + path);
    auto size = static_cast<size_t>(stream.tellg());
    if (size == 0 || size % sizeof(uint32_t) != 0)
        throw std::runtime_error("SPIR-V file has an invalid size");
    std::vector<uint32_t> words(size / sizeof(uint32_t));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(size));
    return words;
}

static uint32_t find_memory_type(VkPhysicalDevice physical, uint32_t bits,
                                 VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (properties.memoryTypes[i].propertyFlags & required) == required)
            return i;
    }
    throw std::runtime_error("no compatible host-visible Vulkan memory type");
}

static Buffer make_buffer(VkDevice device, VkPhysicalDevice physical, VkDeviceSize size) {
    Buffer buffer{.size = size};
    VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    create.size = size;
    create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &create, nullptr, &buffer.handle));

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer.handle, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = find_memory_type(
        physical, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device, &allocation, nullptr, &buffer.memory));
    VK_CHECK(vkBindBufferMemory(device, buffer.handle, buffer.memory, 0));
    return buffer;
}

static void destroy_buffer(VkDevice device, Buffer& buffer) {
    if (buffer.handle)
        vkDestroyBuffer(device, buffer.handle, nullptr);
    if (buffer.memory)
        vkFreeMemory(device, buffer.memory, nullptr);
}

static void upload(VkDevice device, const Buffer& buffer, const std::vector<float>& data) {
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped));
    std::memcpy(mapped, data.data(), data.size() * sizeof(float));
    vkUnmapMemory(device, buffer.memory);
}

int main(int argc, char** argv) try {
    if (argc != 2) {
        std::cerr << "usage: vulkan_dense_probe DENSE_SPV\n";
        return 2;
    }
    constexpr Dimensions dims{8, 256, 512};

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
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &physical_count, physical_devices.data()));

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

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_create.queueFamilyIndex = queue_family;
    queue_create.queueCount = 1;
    queue_create.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_create.queueCreateInfoCount = 1;
    device_create.pQueueCreateInfos = &queue_create;
    VkDevice device{};
    VK_CHECK(vkCreateDevice(physical, &device_create, nullptr, &device));
    VkQueue queue{};
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    std::vector<float> input(dims.rows * dims.input_size);
    std::vector<float> weight(dims.output_size * dims.input_size);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 17.0f;
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 13.0f;
    std::vector<float> expected(dims.rows * dims.output_size, 0.0f);
    for (uint32_t row = 0; row < dims.rows; ++row)
        for (uint32_t column = 0; column < dims.output_size; ++column)
            for (uint32_t k = 0; k < dims.input_size; ++k)
                expected[row * dims.output_size + column] +=
                    input[row * dims.input_size + k] *
                    weight[column * dims.input_size + k];

    Buffer input_buffer = make_buffer(device, physical, input.size() * sizeof(float));
    Buffer weight_buffer = make_buffer(device, physical, weight.size() * sizeof(float));
    Buffer output_buffer = make_buffer(device, physical, expected.size() * sizeof(float));
    upload(device, input_buffer, input);
    upload(device, weight_buffer, weight);

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

    VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Dimensions)};
    VkPipelineLayoutCreateInfo pipeline_layout_create{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_create.setLayoutCount = 1;
    pipeline_layout_create.pSetLayouts = &descriptor_layout;
    pipeline_layout_create.pushConstantRangeCount = 1;
    pipeline_layout_create.pPushConstantRanges = &push_range;
    VkPipelineLayout pipeline_layout{};
    VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_create, nullptr,
                                     &pipeline_layout));

    auto spirv = read_spirv(argv[1]);
    VkShaderModuleCreateInfo shader_create{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
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
    VkPipeline pipeline{};
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_create,
                                      nullptr, &pipeline));

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo pool_create{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_create.maxSets = 1;
    pool_create.poolSizeCount = 1;
    pool_create.pPoolSizes = &pool_size;
    VkDescriptorPool pool{};
    VK_CHECK(vkCreateDescriptorPool(device, &pool_create, nullptr, &pool));
    VkDescriptorSetAllocateInfo set_allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
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

    VkCommandPoolCreateInfo command_pool_create{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_create.queueFamilyIndex = queue_family;
    VkCommandPool command_pool{};
    VK_CHECK(vkCreateCommandPool(device, &command_pool_create, nullptr, &command_pool));
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
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                            0, 1, &descriptor_set, 0, nullptr);
    vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(Dimensions), &dims);
    vkCmdDispatch(command, (dims.output_size + 15) / 16, (dims.rows + 15) / 16, 1);
    VK_CHECK(vkEndCommandBuffer(command));

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    auto started = std::chrono::steady_clock::now();
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    auto ended = std::chrono::steady_clock::now();

    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(device, output_buffer.memory, 0, output_buffer.size, 0, &mapped));
    auto* output = static_cast<float*>(mapped);
    float max_error = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i)
        max_error = std::max(max_error, std::abs(output[i] - expected[i]));
    vkUnmapMemory(device, output_buffer.memory);
    double milliseconds =
        std::chrono::duration<double, std::milli>(ended - started).count();

    std::cout << "{\n"
              << "  \"device\": \"" << physical_properties.deviceName << "\",\n"
              << "  \"vendor_id\": " << physical_properties.vendorID << ",\n"
              << "  \"device_id\": " << physical_properties.deviceID << ",\n"
              << "  \"rows\": " << dims.rows << ",\n"
              << "  \"input_size\": " << dims.input_size << ",\n"
              << "  \"output_size\": " << dims.output_size << ",\n"
              << "  \"dispatch_ms\": " << milliseconds << ",\n"
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
    return max_error <= 1e-3f ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "vulkan_dense_probe: " << error.what() << '\n';
    return 1;
}
