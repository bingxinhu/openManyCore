"""
演示脚本：将TVM Relax IR与prims.py映射并生成芯片核心配置
"""

import os
import sys
import logging
import numpy as np

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'generator/mapping_utils'))

from generator.mapping_utils.relax_to_chip_config import RelaxToChipConfig
from src.model_processor import ModelProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DemoConfig:
    """演示配置类"""
    def __init__(self):
        self.onnx_path = "./onnx_model/lenet.onnx"
        self.input_shape = (1, 1, 28, 28)  # batch, channels, height, width
    
    def get_onnx_path(self):
        return self.onnx_path
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_core_ids_by_role(self, role):
        # 返回计算核心ID列表
        return list(range(16))  # 假设有16个计算核心

def demo_relax_to_prims_mapping():
    """演示Relax IR到prims.py的映射流程"""
    print("=" * 60)
    print("Relax IR到prims.py映射演示")
    print("=" * 60)
    
    # 初始化配置
    config = DemoConfig()
    
    # 步骤1: 加载ONNX模型并转换为Relax IR
    print("\n步骤1: 加载模型并转换为Relax IR...")
    processor = ModelProcessor(config)
    
    try:
        mod, input_name, onnx_model = processor.load_and_convert()
        print("✓ Relax IR转换成功")
        print(f"Relax IR模块结构: {len(mod.functions)} 个函数")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("使用演示模式...")
        return demo_fallback()
    
    # 步骤2: 分析模型层结构
    print("\n步骤2: 分析模型层结构...")
    try:
        layers = processor.analyze_layers(mod)
        print(f"✓ 层分析完成: {len(layers)} 个层")
        for i, (name, layer_type) in enumerate(layers):
            print(f"  层 {i}: {name} -> {layer_type}")
    except Exception as e:
        print(f"✗ 层分析失败: {e}")
        layers = [("conv1", "conv2d"), ("relu1", "relu"), ("pool1", "max_pool2d")]
    
    # 步骤3: 映射到prims.py并生成芯片配置
    print("\n步骤3: 映射到prims.py并生成芯片配置...")
    compiler = RelaxToChipConfig()
    
    try:
        # 方法1: 使用Relax IR进行完整编译
        chip_config = compiler.compile_from_relax(mod, config.input_shape, config.onnx_path)
        print("✓ 完整编译成功")
    except Exception as e:
        print(f"✗ 完整编译失败: {e}")
        # 方法2: 使用简化编译
        try:
            chip_config = compiler.compile_from_onnx(config.onnx_path, config.input_shape)
            print("✓ 简化编译成功")
        except Exception as e2:
            print(f"✗ 简化编译失败: {e2}")
            print("使用演示模式...")
            return demo_fallback()
    
    # 步骤4: 显示配置结果
    print("\n步骤4: 配置结果...")
    print(f"✓ 芯片配置生成完成")
    
    # 修复：正确访问字典键而不是对象属性
    if 'phases' in chip_config:
        total_cores = 0
        for chip_pos in chip_config['phases']:
            for phase_id in chip_config['phases'][chip_pos]:
                total_cores += len(chip_config['phases'][chip_pos][phase_id]['cores'])
        
        print(f"  芯片数量: {len(chip_config['phases'])}")
        print(f"  阶段组数量: {sum(len(phases) for phases in chip_config['phases'].values())}")
        print(f"  核心数量: {total_cores}")
        
        # 显示部分配置信息
        print("\n部分配置示例:")
        chip_positions = list(chip_config['phases'].keys())
        if chip_positions:
            chip_pos = chip_positions[0]
            phases = list(chip_config['phases'][chip_pos].keys())
            if phases:
                phase_id = phases[0]
                cores = list(chip_config['phases'][chip_pos][phase_id]['cores'].keys())
                if cores:
                    core_pos = cores[0]
                    core_config = chip_config['phases'][chip_pos][phase_id]['cores'][core_pos]
                    print(f"  芯片 {chip_pos}, 阶段 {phase_id}, 核心 {core_pos}:")
                    if 'prims' in core_config and core_config['prims']:
                        prim_config = core_config['prims'][0]
                        print(f"    原语类型: {prim_config.get('type', '未知')}")
                        print(f"    参数: {prim_config.get('params', {})}")
    else:
        print("  配置结构不包含phases信息")
    print("\n部分配置示例:")
    for i, (key, value) in enumerate(list(chip_config.map_config.items())[:3]):
        print(f"  {key}: {value}")
    
    print("\n✓ 演示完成!")
    return chip_config

def demo_fallback():
    """演示模式回退函数"""
    print("\n使用演示模式回退...")
    print("✓ 演示模式完成")
    return {"map_config": {"phase_0": {"core_0": "p06", "core_1": "p81"}}}

if __name__ == "__main__":
    # 运行演示
    demo_relax_to_prims_mapping()