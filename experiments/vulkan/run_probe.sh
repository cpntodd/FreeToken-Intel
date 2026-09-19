#!/usr/bin/env bash
set -euo pipefail

probe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${probe_dir}/build"
mkdir -p "${build_dir}"

glslc -O "${probe_dir}/dense.comp" -o "${build_dir}/dense.spv"
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers \
  "${probe_dir}/vulkan_dense_probe.cpp" -lvulkan \
  -o "${build_dir}/vulkan_dense_probe"
"${build_dir}/vulkan_dense_probe" "${build_dir}/dense.spv"
