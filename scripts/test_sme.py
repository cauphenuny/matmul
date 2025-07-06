#!/usr/bin/env python3
"""
测试ARM SME矩阵乘法实现
"""

import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmark'))

try:
    import libmatmul
except ImportError as e:
    print(f"无法导入libmatmul: {e}")
    print("请确保已经编译了库文件")
    sys.exit(1)

def test_matrix_multiplication():
    """测试矩阵乘法实现"""
    print("=== ARM SME矩阵乘法测试 ===")
    
    # 获取系统信息
    print(f"目标架构: {libmatmul.get_target_isa()}")
    print(f"编译器信息: {libmatmul.get_compiler_info()}")
    print(f"构建信息: {libmatmul.get_build_info()}")
    
    # 检查SME支持
    print(f"\n指令集支持:")
    print(f"  - NEON: {libmatmul.has_neon()}")
    if hasattr(libmatmul, 'has_sme'):
        print(f"  - SME: {libmatmul.has_sme()}")
    else:
        print(f"  - SME: 未知")
    print(f"  - FMA: {libmatmul.has_fma()}")
    
    # 测试矩阵大小
    test_sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128)]
    
    for N, M, P in test_sizes:
        print(f"\n--- 测试矩阵大小: {N}x{M} * {M}x{P} = {N}x{P} ---")
        
        # 生成测试数据
        a = np.random.randint(1, 10, (N, M), dtype=np.int32)
        b = np.random.randint(1, 10, (M, P), dtype=np.int32)
        
        # 计算参考结果
        expected = a @ b
        
        # 测试不同的实现
        implementations = [
            ("trivial", libmatmul.trivial),
            ("transpose_iter", libmatmul.transpose_iter),
            ("multithread", libmatmul.multithread),
            ("chunk", libmatmul.chunk),
            ("transpose", libmatmul.transpose),
            ("simd", libmatmul.simd),
            ("simd_optimized", libmatmul.simd_optimized),
        ]
        
        # 如果支持SME，添加SME实现
        if "SME" in libmatmul.get_target_isa():
            implementations.append(("simd_arm_sme", libmatmul.simd_arm_sme))
            if hasattr(libmatmul, 'simd_arm_sme_advanced'):
                implementations.append(("simd_arm_sme_advanced", libmatmul.simd_arm_sme_advanced))
        
        for name, impl in implementations:
            try:
                result = impl(a, b)
                
                # 检查结果是否正确
                if np.array_equal(result, expected):
                    print(f"  ✓ {name}: 正确")
                else:
                    print(f"  ✗ {name}: 错误")
                    print(f"    期望: {expected[0, 0]}, 得到: {result[0, 0]}")
                    
            except Exception as e:
                print(f"  ✗ {name}: 异常 - {e}")

def benchmark_sme():
    """基准测试SME实现"""
    print("\n=== ARM SME基准测试 ===")
    
    # 测试大矩阵
    N, M, P = 512, 512, 512
    print(f"测试矩阵大小: {N}x{M} * {M}x{P} = {N}x{P}")
    
    # 生成测试数据
    a = np.random.randint(1, 10, (N, M), dtype=np.int32)
    b = np.random.randint(1, 10, (M, P), dtype=np.int32)
    
    # 计算参考结果
    print("计算参考结果...")
    expected = a @ b
    
    # 测试SME实现
    if "SME" in libmatmul.get_target_isa():
        implementations = [
            ("simd_arm_sme", libmatmul.simd_arm_sme),
            ("simd_arm_sme_advanced", libmatmul.simd_arm_sme_advanced),
        ]
        
        for name, impl in implementations:
            try:
                print(f"测试 {name}...")
                result = impl(a, b)
                
                if np.array_equal(result, expected):
                    print(f"  ✓ {name}: 结果正确")
                else:
                    print(f"  ✗ {name}: 结果错误")
                    
            except Exception as e:
                print(f"  ✗ {name}: 异常 - {e}")
    else:
        print("当前系统不支持SME")

def main():
    """主函数"""
    try:
        test_matrix_multiplication()
        benchmark_sme()
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
