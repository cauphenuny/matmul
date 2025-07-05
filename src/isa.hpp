#pragma once

#include <string>

namespace isa {

// 获取目标指令集信息
inline std::string get_target_info() {
    std::string info = "Architecture: ";

// CPU 架构检测
#if defined(__x86_64__) || defined(_M_X64)
    info += "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    info += "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
    info += "ARM64";
#elif defined(__arm__) || defined(_M_ARM)
    info += "ARM32";
#elif defined(__riscv)
    info += "RISC-V";
#elif defined(__mips__)
    info += "MIPS";
#else
    info += "Unknown";
#endif

    info += " | SIMD: ";
    bool has_simd = false;

// x86/x86_64 SIMD 指令集检测
#if defined(__AVX512F__)
    info += "AVX512 ";
    has_simd = true;
#elif defined(__AVX2__)
    info += "AVX2 ";
    has_simd = true;
#elif defined(__AVX__)
    info += "AVX ";
    has_simd = true;
#elif defined(__SSE4_2__)
    info += "SSE4.2 ";
    has_simd = true;
#elif defined(__SSE4_1__)
    info += "SSE4.1 ";
    has_simd = true;
#elif defined(__SSSE3__)
    info += "SSSE3 ";
    has_simd = true;
#elif defined(__SSE3__)
    info += "SSE3 ";
    has_simd = true;
#elif defined(__SSE2__)
    info += "SSE2 ";
    has_simd = true;
#elif defined(__SSE__)
    info += "SSE ";
    has_simd = true;
#endif

// ARM SIMD 指令集检测
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    info += "NEON ";
    has_simd = true;
#endif

// 其他特性检测
#if defined(__FMA__)
    info += "FMA ";
    has_simd = true;
#endif

#if defined(__BMI2__)
    info += "BMI2 ";
    has_simd = true;
#endif

#if defined(__F16C__)
    info += "F16C ";
    has_simd = true;
#endif

    if (!has_simd) {
        info += "None";
    } else if (info.back() == ' ') {
        info.pop_back();  // 移除最后的空格
    }

    return info;
}

// 获取编译器信息
inline std::string get_compiler_info() {
    std::string info = "Compiler: ";

#if defined(__clang__)
    info += "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    info += "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#elif defined(_MSC_VER)
    info += "MSVC " + std::to_string(_MSC_VER);
#elif defined(__INTEL_COMPILER)
    info += "Intel ICC " + std::to_string(__INTEL_COMPILER);
#else
    info += "Unknown";
#endif

    return info;
}

// 获取构建配置信息
inline std::string get_build_info() {
    std::string info = "Build: ";

#ifdef NDEBUG
    info += "Release";
#else
    info += "Debug";
#endif

#ifdef _OPENMP
    info += " | OpenMP: Yes";
#else
    info += " | OpenMP: No";
#endif

    return info;
}

// 获取完整的系统信息
inline std::string get_full_info() {
    return get_target_info() + "\n" + get_compiler_info() + "\n" + get_build_info();
}

// 检查特定指令集是否可用
namespace check {

inline constexpr bool has_sse2() {
#ifdef __SSE2__
    return true;
#else
    return false;
#endif
}

inline constexpr bool has_avx() {
#ifdef __AVX__
    return true;
#else
    return false;
#endif
}

inline constexpr bool has_avx2() {
#ifdef __AVX2__
    return true;
#else
    return false;
#endif
}

inline constexpr bool has_avx512() {
#ifdef __AVX512F__
    return true;
#else
    return false;
#endif
}

inline constexpr bool has_neon() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

inline constexpr bool has_fma() {
#ifdef __FMA__
    return true;
#else
    return false;
#endif
}

}  // namespace check

}  // namespace isa
