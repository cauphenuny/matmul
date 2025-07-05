#!/usr/bin/env python3
"""
指令集检测测试脚本
"""

import libmatmul as mm


def test_isa_detection():
    print("=== 系统信息检测 ===")
    print(mm.get_full_info())
    print()

    print("=== 详细信息 ===")
    print(f"目标架构: {mm.get_target_isa()}")
    print(f"编译器: {mm.get_compiler_info()}")
    print(f"构建配置: {mm.get_build_info()}")
    print()

    print("=== 指令集支持检测 ===")
    print(f"SSE2 支持: {mm.has_sse2()}")
    print(f"AVX 支持: {mm.has_avx()}")
    print(f"AVX2 支持: {mm.has_avx2()}")
    print(f"AVX512 支持: {mm.has_avx512()}")
    print(f"NEON 支持: {mm.has_neon()}")
    print(f"FMA 支持: {mm.has_fma()}")


if __name__ == "__main__":
    test_isa_detection()
