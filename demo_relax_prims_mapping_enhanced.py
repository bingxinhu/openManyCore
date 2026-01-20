"""
增强版演示脚本：更健壮的Relax IR到prims.py映射演示
添加详细的调试信息和错误处理
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
    
    # 步骤4: 显示配置结果（增强版）
    print("\n步骤4: 配置结果...")
    print(f"✓ 芯片配置生成完成")
    
    # 详细显示配置结构
    print(f"\n配置字典键: {list(chip_config.keys())}")
    
    # 检查配置结构
    if 'phases' in chip_config:
        print(f"phases 类型: {type(chip_config['phases'])}")
        
        if chip_config['phases']:
            total_chips = len(chip_config['phases'])
            total_phases = 0
            total_cores = 0
            
            for chip_pos, phases in chip_config['phases'].items():
                print(f"芯片位置 {chip_pos}: {len(phases)} 个阶段")
                total_phases += len(phases)
                
                for phase_id, phase_config in phases.items():
                    if 'cores' in phase_config:
                        core_count = len(phase_config['cores'])
                        total_cores += core_count
                        print(f"  阶段 {phase_id}: {core_count} 个核心")
                    else:
                        print(f"  阶段 {phase_id}: 无cores信息")
            
            print(f"\n汇总信息:")
            print(f"  芯片数量: {total_chips}")
            print(f"  阶段组数量: {total_phases}")
            print(f"  核心数量: {total_cores}")
            
            # 显示第一个配置示例
            if total_cores > 0:
                print(f"\n第一个配置示例:")
                chip_pos = list(chip_config['phases'].keys())[0]
                phases = chip_config['phases'][chip_pos]
                if phases:
                    phase_id = list(phases.keys())[0]
                    phase_config = phases[phase_id]
                    if 'cores' in phase_config and phase_config['cores']:
                        core_pos = list(phase_config['cores'].keys())[0]
                        core_config = phase_config['cores'][core_pos]
                        print(f"  芯片 {chip_pos}, 阶段 {phase_id}, 核心 {core_pos}:")
                        if 'prims' in core_config and core_config['prims']:
                            prim_config = core_config['prims'][0]
                            print(f"    原语类型: {prim_config.get('type', '未知')}")
                            print(f"    参数: {prim_config.get('params', {})}")
        else:
            print("phases 字典为空")
    else:
        print("配置中不包含 phases 键")
    
    # 显示原语配置信息
    if 'prims' in chip_config:
        prims = chip_config['prims']
        print(f"\n原语配置数量: {len(prims)}")
        if prims:
            for i, prim in enumerate(prims[:3]):  # 显示前3个
                print(f"  原语 {i}: {prim.get('type', '未知')} - {prim.get('params', {})}")
        else:
            print("  原语列表为空")
    else:
        print("配置中不包含 prims 键")
    
    # 显示其他配置信息
    for key in ['sim_clock', 'chip_array', 'core_array']:
        if key in chip_config:
            print(f"{key}: {chip_config[key]}")
    
    print("\n✓ 演示完成!")
    return chip_config

def demo_fallback():
    """演示模式回退函数"""
    print("\n使用演示模式回退...")
    print("✓ 演示模式完成")
    return {
        "sim_clock": 100000,
        "chip_array": (1, 1),
        "core_array": (4, 4),
        "phases": {
            (0, 0): {
                0: {
                    "clock": 15000,
                    "mode": 1,
                    "cores": {
                        (0, 0): {
                            "prims": [{
                                "type": "p81",
                                "params": {"cin": 1, "cout": 10, "kernel_x": 3, "kernel_y": 3}
                            }],
                            "router": None
                        }
                    }
                }
            }
        },
        "prims": [{
            "type": "p81",
            "layer_index": 0,
            "base_addr": 0x0000,
            "params": {"cin": 1, "cout": 10, "kernel_x": 3, "kernel_y": 3}
        }],
        "weights": {"conv1_weight": "array[10,1,3,3]"}
    }

if __name__ == "__main__":
    # 运行演示
    chip_config = demo_relax_to_prims_mapping()
    
    # 保存配置
    try:
        from generator.mapping_utils.relax_to_chip_config import RelaxToChipConfig
        compiler = RelaxToChipConfig()
        compiler.save_config(chip_config, "./demo_chip_config_enhanced.pkl")
        print("✓ 配置已保存到 demo_chip_config_enhanced.pkl")
        
        # 同时保存JSON版本
        compiler.save_config(chip_config, "./demo_chip_config_enhanced.json")
        print("✓ 配置已保存到 demo_chip_config_enhanced.json")
    except Exception as e:
        print(f"✗ 配置保存失败: {e}")
        import traceback
        traceback.print_exc()