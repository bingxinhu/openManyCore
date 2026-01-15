#!/usr/bin/env python3
"""
增强内存架构演示脚本
"""

import logging
import numpy as np
from manycore_config import ManyCoreYAMLConfig
from manycore_memory_arch import EnhancedMemoryArchitecture

def demo_enhanced_memory_architecture():
    """演示增强内存架构的使用"""
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    config_path = "enhanced_memory_config.yaml"
    config = ManyCoreYAMLConfig(config_path)
    
    # 创建内存架构
    memory_arch = EnhancedMemoryArchitecture(config)
    
    # 示例数据
    input_data = np.random.rand(1, 1, 1, 100).astype(np.float32)
    weights = {
        "conv1_weight": np.random.randn(10, 1, 3, 3).astype(np.float32),
        "fc1_weight": np.random.randn(10, 10).astype(np.float32)
    }
    
    print("=== 增强内存架构演示 ===")
    
    # 存储数据到各核心
    print("\n1. 存储数据到各核心的MEM0和MEM1...")
    for core_id in range(min(5, config.get_total_cores())):  # 只演示前5个核心
        addr = memory_arch.store_data_mem0(core_id, input_data, "input")
        weight_locations = memory_arch.store_weights_mem1(core_id, weights)
        print(f"核心{core_id}: 输入数据@0x{addr:x}, 权重数量: {len(weight_locations)}")
    
    # 演示MEM3缓存
    print("\n2. 演示MEM3核间传输缓存...")
    sample_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    if memory_arch.cache_intercore_data(0, 1, sample_data):
        cached_data = memory_arch.get_cached_intercore_data(0, 1)
        print(f"从核心0到核心1的缓存数据: {cached_data}")
    
    # 获取内存使用情况
    print("\n3. 内存使用情况:")
    for core_id in range(min(3, config.get_total_cores())):
        mem_usage = memory_arch.get_memory_usage(core_id)
        print(f"核心{core_id}:")
        print(f"  MEM0: {mem_usage['mem0']['used']}/{mem_usage['mem0']['total']} bytes ({mem_usage['mem0']['usage_percent']:.1f}%)")
        print(f"  MEM1: {mem_usage['mem1']['used']}/{mem_usage['mem1']['total']} bytes ({mem_usage['mem1']['usage_percent']:.1f}%)")
    
    # 演示原语查找
    print("\n4. 原语查找演示:")
    primitive_opcode = memory_arch.get_primitive_opcode(0, "conv2d")
    print(f"原语'conv2d'的操作码: 0x{primitive_opcode:02x}")
    
    # 演示路由表查询
    print("\n5. 路由表查询演示:")
    routing_info = memory_arch.get_routing_info(0, 0, 1)
    if routing_info:
        next_hop, cost = routing_info
        print(f"从核心0到核心1: 下一跳={next_hop}, 代价={cost}")
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    demo_enhanced_memory_architecture()
