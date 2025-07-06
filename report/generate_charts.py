#!/usr/bin/env python3
"""
矩阵乘法性能分析图表生成脚本
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 性能数据 (从benchmark结果中提取)
performance_data = {
    'numpy': [0.0722, 0.6631, 9.8778, 80.7523, 950.1124, 11928.2853, 259570.5816],
    'trivial': [0.1160, 1.2753, 13.7274, 120.6185, 859.7644, 9677.0381, 211787.5126],
    'transpose loop iter': [0.0502, 0.4082, 2.5376, 8.8812, 66.0672, 527.7147, 4244.6427],
    'multi-thread': [0.0640, 0.3820, 1.5725, 11.5955, 84.4738, 662.8226, 6127.7839],
    'chunk': [0.1018, 0.3982, 2.8003, 11.9732, 100.5427, 873.4043, 8381.0507],
    'chunk, multi-thread': [0.5588, 1.7377, 3.1473, 13.0390, 105.2605, 830.2613, 8694.7662],
    'transpose matrix B': [0.0305, 0.2732, 2.0661, 8.4455, 58.6889, 471.1689, 4177.3346],
    'SIMD (auto)': [0.0786, 0.3009, 2.5551, 8.5047, 57.9486, 466.4718, 4165.2117],
    'SIMD (manual)': [0.0564, 0.7533, 2.6572, 16.3190, 132.4153, 1302.4115, 11850.1109],
    'SIMD (optimized)': [0.0541, 0.6838, 1.0338, 7.6023, 59.8543, 481.0391, 4162.3652],
    'SIMD, multi-thread': [0.1015, 0.8904, 1.2360, 2.8579, 15.5927, 120.3897, 1514.3115]
}

matrix_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

# 颜色映射
colors = {
    'numpy': '#1f77b4',
    'trivial': '#ff7f0e',
    'transpose loop iter': '#2ca02c',
    'multi-thread': '#d62728',
    'chunk': '#9467bd',
    'chunk, multi-thread': '#8c564b',
    'transpose matrix B': '#e377c2',
    'SIMD (auto)': '#7f7f7f',
    'SIMD (manual)': '#bcbd22',
    'SIMD (optimized)': '#17becf',
    'SIMD, multi-thread': '#ff9896'
}

def create_performance_comparison_chart():
    """创建性能对比图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图1: 不同矩阵大小的性能曲线
    for method, times in performance_data.items():
        ax1.plot(matrix_sizes, times, marker='o', label=method, 
                color=colors.get(method, '#000000'), linewidth=2, markersize=4)
    
    ax1.set_xlabel('矩阵大小 (N)', fontsize=12)
    ax1.set_ylabel('执行时间 (ms)', fontsize=12)
    ax1.set_title('不同矩阵大小下的性能对比', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # 图2: N=4096时的柱状图
    n4096_times = [performance_data[method][6] for method in performance_data.keys()]
    methods = list(performance_data.keys())
    
    bars = ax2.bar(range(len(methods)), n4096_times, 
                   color=[colors.get(method, '#000000') for method in methods], alpha=0.7)
    ax2.set_xlabel('实现方法', fontsize=12)
    ax2.set_ylabel('执行时间 (ms)', fontsize=12)
    ax2.set_title('N=4096时的性能对比', fontsize=14)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for bar, time_ms in zip(bars, n4096_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{time_ms:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('report/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_speedup_analysis():
    """创建加速比分析图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 计算加速比 (相对于numpy)
    speedup_data = {}
    numpy_times = performance_data['numpy']
    
    for method, times in performance_data.items():
        if method != 'numpy':
            speedups = [numpy_times[i] / times[i] for i in range(len(times))]
            speedup_data[method] = speedups
    
    # 图1: 加速比随矩阵大小的变化
    for method, speedups in speedup_data.items():
        ax1.plot(matrix_sizes, speedups, marker='o', label=method,
                color=colors.get(method, '#000000'), linewidth=2, markersize=4)
    
    ax1.set_xlabel('矩阵大小 (N)', fontsize=12)
    ax1.set_ylabel('加速比 (相对于numpy)', fontsize=12)
    ax1.set_title('加速比随矩阵大小的变化', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # 图2: N=4096时的加速比柱状图
    n4096_speedups = [speedup_data[method][6] for method in speedup_data.keys()]
    methods = list(speedup_data.keys())
    
    # 按加速比排序
    sorted_data = sorted(zip(methods, n4096_speedups), key=lambda x: x[1], reverse=True)
    sorted_methods, sorted_speedups = zip(*sorted_data)
    
    bars = ax2.bar(range(len(sorted_methods)), sorted_speedups,
                   color=[colors.get(method, '#000000') for method in sorted_methods], alpha=0.7)
    ax2.set_xlabel('实现方法', fontsize=12)
    ax2.set_ylabel('加速比', fontsize=12)
    ax2.set_title('N=4096时的加速比排名', fontsize=14)
    ax2.set_xticks(range(len(sorted_methods)))
    ax2.set_xticklabels(sorted_methods, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for bar, speedup in zip(bars, sorted_speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('report/speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_optimization_technique_analysis():
    """创建优化技术分析图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 按优化技术分组
    technique_groups = {
        '基础实现': ['numpy', 'trivial'],
        '循环优化': ['transpose loop iter', 'transpose matrix B'],
        '并行化': ['multi-thread', 'chunk, multi-thread'],
        '缓存优化': ['chunk'],
        'SIMD优化': ['SIMD (auto)', 'SIMD (manual)', 'SIMD (optimized)', 'SIMD, multi-thread']
    }
    
    # 图1: 不同优化技术的性能对比 (N=4096)
    n4096_times = performance_data['numpy'][6]  # numpy基准时间
    technique_performance = {}
    
    for technique, methods in technique_groups.items():
        if technique == '基础实现':
            technique_performance[technique] = n4096_times
        else:
            min_time = min(performance_data[method][6] for method in methods)
            technique_performance[technique] = min_time
    
    techniques = list(technique_performance.keys())
    times = list(technique_performance.values())
    
    bars = ax1.bar(techniques, times, alpha=0.7)
    ax1.set_ylabel('执行时间 (ms)', fontsize=12)
    ax1.set_title('不同优化技术的最佳性能 (N=4096)', fontsize=14)
    ax1.set_xticklabels(techniques, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for bar, time_ms in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{time_ms:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 图2: 优化技术加速比
    speedups = [n4096_times / time for time in times]
    bars = ax2.bar(techniques, speedups, alpha=0.7)
    ax2.set_ylabel('加速比', fontsize=12)
    ax2.set_title('不同优化技术的加速比 (N=4096)', fontsize=14)
    ax2.set_xticklabels(techniques, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=9)
    
    # 图3: 矩阵大小对性能的影响
    selected_methods = ['numpy', 'transpose matrix B', 'SIMD (auto)', 'SIMD, multi-thread']
    for method in selected_methods:
        ax3.plot(matrix_sizes, performance_data[method], marker='o', label=method,
                color=colors.get(method, '#000000'), linewidth=2, markersize=4)
    
    ax3.set_xlabel('矩阵大小 (N)', fontsize=12)
    ax3.set_ylabel('执行时间 (ms)', fontsize=12)
    ax3.set_title('关键方法的性能随矩阵大小变化', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    
    # 图4: 性能瓶颈分析
    # 计算不同矩阵大小下的最佳方法
    best_methods = []
    best_times = []
    
    for i, size in enumerate(matrix_sizes):
        min_time = float('inf')
        best_method = ''
        for method, times in performance_data.items():
            if times[i] < min_time:
                min_time = times[i]
                best_method = method
        best_methods.append(best_method)
        best_times.append(min_time)
    
    # 统计每种方法成为最佳的次数
    method_counts = {}
    for method in best_methods:
        method_counts[method] = method_counts.get(method, 0) + 1
    
    methods = list(method_counts.keys())
    counts = list(method_counts.values())
    
    bars = ax4.bar(methods, counts, alpha=0.7)
    ax4.set_ylabel('成为最佳方法的次数', fontsize=12)
    ax4.set_title('不同矩阵大小下的最佳方法统计', fontsize=14)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height,
                str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('report/optimization_technique_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table():
    """创建性能总结表格"""
    # 计算N=4096时的性能数据
    n4096_data = []
    numpy_baseline = performance_data['numpy'][6]
    
    for method, times in performance_data.items():
        time_ms = times[6]  # N=4096的时间
        speedup = numpy_baseline / time_ms
        n4096_data.append({
            '方法': method,
            '执行时间(ms)': f'{time_ms:.2f}',
            '加速比': f'{speedup:.2f}x',
            '相对性能': f'{speedup/numpy_baseline*100:.1f}%'
        })
    
    # 按加速比排序
    n4096_data.sort(key=lambda x: float(x['加速比'].replace('x', '')), reverse=True)
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(n4096_data)
    df.to_csv('report/performance_summary.csv', index=False, encoding='utf-8-sig')
    
    # 打印表格
    print("矩阵乘法性能总结 (N=4096)")
    print("=" * 80)
    print(f"{'方法':<20} {'执行时间(ms)':<12} {'加速比':<10} {'相对性能':<12}")
    print("-" * 80)
    for row in n4096_data:
        print(f"{row['方法']:<20} {row['执行时间(ms)']:<12} {row['加速比']:<10} {row['相对性能']:<12}")
    
    return df

def main():
    """主函数"""
    print("生成矩阵乘法性能分析图表...")
    
    # 创建各种图表
    create_performance_comparison_chart()
    create_speedup_analysis()
    create_optimization_technique_analysis()
    
    # 创建总结表格
    summary_df = create_summary_table()
    
    print("\n图表生成完成！")
    print("生成的文件：")
    print("- report/performance_comparison.png")
    print("- report/speedup_analysis.png")
    print("- report/optimization_technique_analysis.png")
    print("- report/performance_summary.csv")

if __name__ == "__main__":
    main() 