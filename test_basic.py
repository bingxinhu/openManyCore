#!/usr/bin/env python3
"""
基本功能测试脚本
"""

import os
import sys

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'generator/mapping_utils'))

def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)
    
    try:
        from generator.mapping_utils.relax_to_chip_config import RelaxToChipConfig
        print("✓ RelaxToChipConfig 导入成功")
    except Exception as e:
        print(f"✗ RelaxToChipConfig 导入失败: {e}")
        return False
    
    try:
        from src.model_processor import ModelProcessor
        print("✓ ModelProcessor 导入成功")
    except Exception as e:
        print(f"✗ ModelProcessor 导入失败: {e}")
        return False
    
    try:
        from generator.mapping_utils.relax_prims_mapper import RelaxPrimsMapper
        print("✓ RelaxPrimsMapper 导入成功")
    except Exception as e:
        print(f"✗ RelaxPrimsMapper 导入失败: {e}")
    
    try:
        from generator.mapping_utils.map_config_gen import MapConfigGen
        print("✓ MapConfigGen 导入成功")
    except Exception as e:
        print(f"✗ MapConfigGen 导入失败: {e}")
    
    return True

def test_onnx_model():
    """测试ONNX模型是否存在"""
    print("\n" + "=" * 60)
    print("测试ONNX模型")
    print("=" * 60)
    
    onnx_files = [
        "./lenet.onnx",
        "./onnx_model/lenet_simple.onnx",
        "./onnx_model/lenet_fallback.onnx"
    ]
    
    found = False
    for onnx_file in onnx_files:
        if os.path.exists(onnx_file):
            print(f"✓ 找到ONNX模型: {onnx_file}")
            found = True
            break
        else:
            print(f"✗ 未找到: {onnx_file}")
    
    if not found:
        print("请先运行 python pthTonnx.py 生成ONNX模型")
        return False
    
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("测试基本功能")
    print("=" * 60)
    
    try:
        from generator.mapping_utils.relax_prims_mapper import RelaxPrimsMapper
        
        # 创建映射器实例
        mapper = RelaxPrimsMapper()
        print("✓ RelaxPrimsMapper 实例化成功")
        
        # 测试映射表
        print(f"映射表包含 {len(mapper.relax_to_prims_map)} 个映射")
        print("部分映射示例:")
        for i, (key, value) in enumerate(list(mapper.relax_to_prims_map.items())[:5]):
            print(f"  {key} -> {value}")
        
        return True
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试...")
    
    # 测试导入
    if not test_imports():
        print("\n导入测试失败，请检查环境配置")
        return
    
    # 测试ONNX模型
    test_onnx_model()
    
    # 测试基本功能
    test_basic_functionality()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print("\n如果所有测试都通过，可以运行演示脚本:")
    print("python demo_relax_prims_mapping.py")

if __name__ == "__main__":
    main()