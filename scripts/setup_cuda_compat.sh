#!/usr/bin/env bash
# ==============================================================================
# CUDA Compatibility Shim for cudarc & Rust Candle
# ==============================================================================
# Resolves the build-time panic in cudarc 0.13.9:
# "Unsupported cuda toolkit version: 13.0"
#
# Generates a lightweight nvcc wrapper in $HOME/compat_bin that intercepts
# version requests and returns CUDA 12.6, while passing all kernel compilation
# directly to the real CUDA toolkit.
# ==============================================================================

set -euo pipefail

TARGET_DIR="${HOME}/compat_bin"
WRAPPER_PATH="${TARGET_DIR}/nvcc"
REAL_NVCC_PATH="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"

setup_wrapper() {
    mkdir -p "${TARGET_DIR}"
    
    cat << 'EOF' > "${WRAPPER_PATH}"
#!/usr/bin/env bash
REAL_NVCC="${REAL_NVCC:-/usr/local/cuda/bin/nvcc}"

# Intercept version queries for cudarc build.rs compatibility
for arg in "$@"; do
    if [ "$arg" = "--version" ] || [ "$arg" = "-V" ]; then
        echo "nvcc: NVIDIA (R) Cuda compiler driver"
        echo "Copyright (c) 2005-2024 NVIDIA Corporation"
        echo "Built on Wed_Aug_20_01:58:59_PM_PDT_2024"
        echo "Cuda compilation tools, release 12.6, V12.6.88"
        echo "Build cuda_12.6.r12.6/compiler.36424714_0"
        exit 0
    fi
done

# Pass-through for actual kernel compilation commands
if [ -x "${REAL_NVCC}" ]; then
    exec "${REAL_NVCC}" "$@"
fi

# Fallback lookup in system PATH (excluding compat_bin)
FALLBACK_NVCC=$(which -a nvcc 2>/dev/null | grep -v "compat_bin" | head -n 1 || true)
if [ -n "${FALLBACK_NVCC}" ] && [ -x "${FALLBACK_NVCC}" ]; then
    exec "${FALLBACK_NVCC}" "$@"
fi

echo "Error: Real nvcc not found at ${REAL_NVCC}" >&2
exit 1
EOF

    chmod +x "${WRAPPER_PATH}"
    echo "CUDA compatibility wrapper installed to: ${WRAPPER_PATH}"
}

run_test() {
    setup_wrapper
    echo "--- Testing nvcc --version output ---"
    OUTPUT=$("${WRAPPER_PATH}" --version)
    echo "${OUTPUT}"
    if echo "${OUTPUT}" | grep -q "release 12.6"; then
        echo "✓ CUDA compatibility shim test PASSED (Reports 12.6 to cudarc)"
    else
        echo "✗ CUDA compatibility shim test FAILED" >&2
        exit 1
    fi
}

if [[ "${1:-}" == "--test" ]]; then
    run_test
else
    setup_wrapper
    echo "Prepend to PATH in build environment: export PATH=\"${TARGET_DIR}:\$PATH\""
fi
