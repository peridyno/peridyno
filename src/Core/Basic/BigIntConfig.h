#ifndef BIGINT_BIGINTCONFIG_H
#define BIGINT_BIGINTCONFIG_H

// Centralized compile-time configuration for dyno BigInt.
// The defaults are conservative and can be overridden by defining the
// corresponding macros before including any BigInt headers or via the build system.

// Detect whether we are being compiled by NVCC / CUDA frontends.
#if defined(__CUDACC__)
    #define DYNO_COMPILING_WITH_CUDA 1
#else
    #define DYNO_COMPILING_WITH_CUDA 0
#endif

// Toggle CUDA-specific code paths. CMake can set DYNO_ENABLE_CUDA to 1 when CUDA targets are built.
#ifndef DYNO_ENABLE_CUDA
    #define DYNO_ENABLE_CUDA DYNO_COMPILING_WITH_CUDA
#endif

// Optional PTX-optimized 128-bit implementation (requires CUDA toolchain).
#ifndef DYNO_ENABLE_PTX
    #define DYNO_ENABLE_PTX 0
#endif
#if DYNO_ENABLE_PTX && !DYNO_ENABLE_CUDA
    #undef DYNO_ENABLE_PTX
    #define DYNO_ENABLE_PTX 0
#endif
#if DYNO_ENABLE_PTX && !defined(__CUDACC__)
    #error "DYNO_ENABLE_PTX requires compilation with NVCC (__CUDACC__)."
#endif

// Host/device annotation helper.
#if DYNO_ENABLE_CUDA && defined(__CUDACC__)
    #define DYNO_HOST_DEVICE __host__ __device__
    #define DYNO_DEVICE_CODE 1
#else
    #define DYNO_HOST_DEVICE
    #define DYNO_DEVICE_CODE 0
#endif


// Enable inline x86_64 asm helpers on supported host compilers (never on device code).
#ifndef DYNO_ENABLE_HOST_ASM64
    #if !DYNO_DEVICE_CODE && (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
        #define DYNO_ENABLE_HOST_ASM64 1
    #else
        #define DYNO_ENABLE_HOST_ASM64 0
    #endif
#endif

// Portable restrict qualifier.
#ifndef DYNO_RESTRICT
    #if defined(_MSC_VER)
        #define DYNO_RESTRICT __restrict
    #else
        #define DYNO_RESTRICT __restrict__
    #endif
#endif

#endif // BIGINT_BIGINTCONFIG_H