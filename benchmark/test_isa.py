#!/usr/bin/env python3

import libmatmul as mm
import platform


def test_isa_detection():
    """测试指令集检测功能"""
    print("=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {platform.python_version()}")

    print("\n=== Compiled Target ISA ===")
    isa_info = mm.get_target_isa()
    print(isa_info)

    # 解析指令集信息
    print("\n=== ISA Analysis ===")
    if "ARM64" in isa_info:
        print("✓ Target: ARM64 (Apple Silicon)")
        if "NEON" in isa_info:
            print("✓ SIMD: NEON available")
        else:
            print("✗ SIMD: NEON not detected")
    elif "x86_64" in isa_info:
        print("✓ Target: x86_64 (Intel/AMD)")
        if "AVX512" in isa_info:
            print("✓ SIMD: AVX512 available")
        elif "AVX2" in isa_info:
            print("✓ SIMD: AVX2 available")
        elif "AVX" in isa_info:
            print("✓ SIMD: AVX available")
        elif "SSE" in isa_info:
            print("✓ SIMD: SSE available")
        else:
            print("✗ SIMD: No SIMD detected")
    else:
        print("? Target: Unknown architecture")


if __name__ == "__main__":
    test_isa_detection()
