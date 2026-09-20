#!/usr/bin/env bash
set -euo pipefail

probe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${probe_dir}/build"
kernel="${1:-both}"
if [[ $# -gt 0 ]]; then
  shift
fi
mkdir -p "${build_dir}"

glslc -O "${probe_dir}/dense_naive.comp" -o "${build_dir}/dense_naive.spv"
glslc -O "${probe_dir}/dense.comp" -o "${build_dir}/dense.spv"
cooperative_spv="-"
if glslc -O --target-env=vulkan1.2 "${probe_dir}/dense_cooperative.comp" \
  -o "${build_dir}/dense_cooperative.spv"; then
  cooperative_spv="${build_dir}/dense_cooperative.spv"
fi
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers \
  "${probe_dir}/vulkan_dense_probe.cpp" -lvulkan \
  -o "${build_dir}/vulkan_dense_probe"

run_kernel() {
  "${build_dir}/vulkan_dense_probe" "${1}" \
    "${build_dir}/dense_naive.spv" "${build_dir}/dense.spv" \
    "${cooperative_spv}" "${@:2}"
}

case "${kernel}" in
  naive)
    run_kernel naive "$@"
    ;;
  tiled)
    run_kernel tiled "$@"
    ;;
  cooperative)
    run_kernel cooperative "$@"
    ;;
  best)
    run_kernel best "$@"
    ;;
  both)
    run_kernel naive "$@"
    run_kernel tiled "$@"
    run_kernel best "$@"
    ;;
  *)
    echo "usage: run_probe.sh [naive|tiled|cooperative|best|both] [ROWS INPUT_SIZE OUTPUT_SIZE ITERATIONS]" >&2
    exit 2
    ;;
esac
