#!/usr/bin/env python3
"""
修复验证测试脚本
测试RelaxPrimsMapper类的修复效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generator.mapping_utils.relax_prims_mapper import RelaxPrimsMapper

def test_relax_prims_mapper_fix():
    """测试RelaxPrimsMapper修复效果"""
    print("=" * 60)
    print("测试RelaxPrimsMapper修复")
    print("=" * 60)
    
    try:
        # 创建映射器实例
        mapper = RelaxPrimsMapper()
        print("✓ RelaxPrimsMapper实例化成功")
        
        # 检查是否所有映射方法都存在
        required_methods = [
            '_map_dense', '_map_default', '_calculate_next_addr',
            '_map_sigmoid', '_map_tanh', '_map_matmul'
        ]
        
        for method_name in required_methods:
            if hasattr(mapper, method_name):
                print(f"✓ {method_name} 方法存在")
            else:
                print(f"✗ {method_name} 方法不存在")
                return False
        
        # 测试基本功能
        print("\n测试基本功能...")
        
        # 测试默认映射
        default_config = mapper._map_default(
            'unknown', {}, (1, 32, 32), (1, 32, 32), 0x0000
        )
        if default_config and default_config['type'] == 'p06':
            print("✓ 默认映射功能正常")
        else:
            print("✗ 默认映射功能异常")
            return False
        
        # 测试地址计算
        next_addr = mapper._calculate_next_addr(0x0000, (1, 32, 32))
        if next_addr > 0x0000:
            print("✓ 地址计算功能正常")
        else:
            print("✗ 地址计算功能异常")
            return False
        
        # 测试dense映射
        dense_config = mapper._map_dense(
            'dense', {}, (1, 128), (1, 64), 0x0000
        )
        if dense_config and dense_config['type'] == 'p04':
            print("✓ Dense映射功能正常")
        else:
            print("✗ Dense映射功能异常")
            return False
        
        print("\n" + "=" * 60)
        print("✓ RelaxPrimsMapper修复测试通过")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"✗ RelaxPrimsMapper测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_relax_prims_mapper_fix()
    if success:
        print("\n修复验证完成！现在可以运行演示脚本了。")
        print("运行命令: python demo_relax_prims_mapping.py")
    else:
        print("\n修复验证失败，请检查代码。")
        sys.exit(1)