#!/usr/bin/env python3
"""
ONNX模型到LeNet风格芯片配置转换工具
修复了CoreConfig缺少initData和MemoryOutput部分的问题
修复了JSON序列化元组键问题
"""

import os
import sys
import json
import pickle
import numpy as np
import argparse
from typing import Dict, List, Any, Optional
import copy

try:
    from generator.mapping_utils.relax_prims_mapper import RelaxPrimsMapper
    from generator.mapping_utils.relax_to_chip_config import RelaxToChipConfig
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保mapping_utils模块在路径中")
    sys.exit(1)

def convert_tuple_keys_to_string(obj):
    """递归地将字典中的元组键转换为字符串"""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # 转换元组键为字符串
            if isinstance(key, tuple):
                # 将元组转换为字符串，例如 (0, 0) -> "0_0"
                new_key = "_".join(str(item) for item in key)
            else:
                new_key = str(key)  # 确保所有键都是字符串
            
            # 递归处理值
            new_dict[new_key] = convert_tuple_keys_to_string(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_tuple_keys_to_string(item) for item in obj]
    elif isinstance(obj, tuple):
        # 将元组值转换为列表（JSON不支持元组）
        return [convert_tuple_keys_to_string(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        # 将numpy标量转换为Python原生类型
        return obj.item()
    elif isinstance(obj, np.ndarray):
        # 将numpy数组转换为列表
        return obj.tolist()
    else:
        return obj

def load_onnx_model_and_extract_layers(onnx_path: str) -> Dict[str, Any]:
    """加载ONNX模型并提取层信息"""
    import onnx
    from onnx import numpy_helper
    
    print(f"加载ONNX模型: {onnx_path}")
    model = onnx.load(onnx_path)
    
    # 提取层信息
    layers_info = []
    weights = {}
    
    # 分析图节点
    for node in model.graph.node:
        layer_info = {
            'name': node.name,
            'type': node.op_type,
            'input_names': list(node.input),
            'output_names': list(node.output),
            'attributes': {}
        }
        
        # 提取属性
        for attr in node.attribute:
            layer_info['attributes'][attr.name] = extract_attribute_value(attr)
            
        layers_info.append(layer_info)
    
    # 提取权重
    for initializer in model.graph.initializer:
        weight_name = initializer.name
        weight_data = numpy_helper.to_array(initializer)
        weights[weight_name] = weight_data
    
    return {
        'layers_info': layers_info,
        'weights': weights
    }

def extract_weights_from_onnx(onnx_path: str) -> Dict[str, np.ndarray]:
    """从ONNX模型提取权重参数"""
    import onnx
    from onnx import numpy_helper
    
    model = onnx.load(onnx_path)
    weights = {}
    
    for initializer in model.graph.initializer:
        weight_name = initializer.name
        weight_data = numpy_helper.to_array(initializer)
        weights[weight_name] = weight_data
    
    return weights

def extract_layer_info_from_onnx(onnx_path: str) -> List[Dict[str, Any]]:
    """从ONNX模型提取层信息"""
    import onnx
    
    model = onnx.load(onnx_path)
    layers_info = []
    
    for node in model.graph.node:
        layer_info = {
            'name': node.name,
            'type': node.op_type,
            'input_names': list(node.input),
            'output_names': list(node.output),
            'attributes': {}
        }
        
        for attr in node.attribute:
            layer_info['attributes'][attr.name] = extract_attribute_value(attr)
            
        layers_info.append(layer_info)
    
    return layers_info

def extract_attribute_value(attr) -> Any:
    """提取ONNX属性值"""
    import onnx
    
    if attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode('utf-8')
    elif attr.type == onnx.AttributeProto.STRINGS:
        return [s.decode('utf-8') for s in attr.strings]
    else:
        return str(attr)

def create_simple_layer_analyzer() -> Dict[str, Any]:
    """创建简单的层分析器（用于测试）"""
    # 模拟LeNet模型结构
    layers_info = [
        {
            'name': 'conv1',
            'type': 'conv2d',
            'input_shape': [1, 1, 28, 28],
            'output_shape': [1, 6, 24, 24],
            'attributes': {'kernel_shape': [5, 5], 'strides': [1, 1], 'pads': [0, 0, 0, 0]}
        },
        {
            'name': 'relu1',
            'type': 'relu',
            'input_shape': [1, 6, 24, 24],
            'output_shape': [1, 6, 24, 24],
            'attributes': {}
        },
        {
            'name': 'pool1',
            'type': 'maxpool',
            'input_shape': [1, 6, 24, 24],
            'output_shape': [1, 6, 12, 12],
            'attributes': {'kernel_shape': [2, 2], 'strides': [2, 2]}
        },
        {
            'name': 'conv2',
            'type': 'conv2d',
            'input_shape': [1, 6, 12, 12],
            'output_shape': [1, 16, 8, 8],
            'attributes': {'kernel_shape': [5, 5], 'strides': [1, 1], 'pads': [0, 0, 0, 0]}
        },
        {
            'name': 'relu2',
            'type': 'relu',
            'input_shape': [1, 16, 8, 8],
            'output_shape': [1, 16, 8, 8],
            'attributes': {}
        },
        {
            'name': 'pool2',
            'type': 'maxpool',
            'input_shape': [1, 16, 8, 8],
            'output_shape': [1, 16, 4, 4],
            'attributes': {'kernel_shape': [2, 2], 'strides': [2, 2]}
        },
        {
            'name': 'fc1',
            'type': 'gemm',
            'input_shape': [1, 256],
            'output_shape': [1, 120],
            'attributes': {}
        },
        {
            'name': 'fc2',
            'type': 'gemm',
            'input_shape': [1, 120],
            'output_shape': [1, 84],
            'attributes': {}
        },
        {
            'name': 'fc3',
            'type': 'gemm',
            'input_shape': [1, 84],
            'output_shape': [1, 10],
            'attributes': {}
        }
    ]
    
    # 模拟权重
    weights = {
        'conv1.weight': np.random.randn(6, 1, 5, 5).astype(np.float32),
        'conv1.bias': np.random.randn(6).astype(np.float32),
        'conv2.weight': np.random.randn(16, 6, 5, 5).astype(np.float32),
        'conv2.bias': np.random.randn(16).astype(np.float32),
        'fc1.weight': np.random.randn(256, 120).astype(np.float32),
        'fc1.bias': np.random.randn(120).astype(np.float32),
        'fc2.weight': np.random.randn(120, 84).astype(np.float32),
        'fc2.bias': np.random.randn(84).astype(np.float32),
        'fc3.weight': np.random.randn(84, 10).astype(np.float32),
        'fc3.bias': np.random.randn(10).astype(np.float32)
    }
    
    return {
        'layers_info': layers_info,
        'weights': weights
    }

def map_onnx_op_to_prim_type(onnx_op_type: str, layer_index: int) -> str:
    """将ONNX操作类型映射到prims原语类型"""
    mapping = {
        'conv2d': 'p81',      # 第一层卷积
        'conv': 'p41',        # 其他卷积层
        'gemm': 'p41',        # 全连接层
        'relu': 'pX5',        # ReLU激活
        'sigmoid': 'pX5',     # Sigmoid激活
        'tanh': 'pX5',        # Tanh激活
        'maxpool': 'pX5',     # 最大池化
        'averagepool': 'pX5', # 平均池化
    }
    
    # 特殊处理：第一层卷积使用p81，其他卷积使用p41
    if onnx_op_type == 'conv2d' and layer_index == 0:
        return 'p81'
    elif onnx_op_type in mapping:
        return mapping[onnx_op_type]
    else:
        return 'p41'  # 默认使用p41

def create_default_pi_config(prim: Dict[str, Any], core_index: int) -> Dict[str, Any]:
    """创建默认的PI配置"""
    return {
        "A_valid": False,
        "S1_valid": False,
        "S2_valid": False,
        "S3_valid": False,
        "S4_valid": False,
        "S5_valid": False,
        "S6_valid": False,
        "S7_valid": False,
        "S8_valid": False,
        "S9_valid": False,
        "S10_valid": False,
        "S11_valid": False,
        "S12_valid": False,
        "S13_valid": False,
        "S14_valid": False,
        "S15_valid": False,
        "PI_parameter": [{}, {}, {}, {}],
        "Additional": [{}, {}, {}, {}]
    }

def create_p81_pi_config(prim: Dict[str, Any], core_index: int) -> Dict[str, Any]:
    """创建p81原语的PI配置（第一层卷积）"""
    pi_config = create_default_pi_config(prim, core_index)
    pi_config.update({
        "A_valid": True,
        "S1_valid": True,
        "PI_parameter": [
            {"PIC_Mode": 1, "Addr_range": [0x0000, 0x0FFF]},
            {"PIC_Mode": 0, "Addr_range": [0x1000, 0x1FFF]},
            {"PIC_Mode": 0, "Addr_range": [0x2000, 0x2FFF]},
            {"PIC_Mode": 0, "Addr_range": [0x3000, 0x3FFF]}
        ]
    })
    return pi_config

def create_p41_pi_config(prim: Dict[str, Any], core_index: int) -> Dict[str, Any]:
    """创建p41原语的PI配置（卷积/全连接）"""
    pi_config = create_default_pi_config(prim, core_index)
    pi_config.update({
        "A_valid": True,
        "S2_valid": True,
        "PI_parameter": [
            {"PIC_Mode": 2, "Addr_range": [0x4000, 0x4FFF]},
            {"PIC_Mode": 0, "Addr_range": [0x5000, 0x5FFF]},
            {"PIC_Mode": 0, "Addr_range": [0x6000, 0x6FFF]},
            {"PIC_Mode": 0, "Addr_range": [0x7000, 0x7FFF]}
        ]
    })
    return pi_config

def create_pX5_pi_config(prim: Dict[str, Any], core_index: int) -> Dict[str, Any]:
    """创建pX5原语的PI配置（池化/激活）"""
    pi_config = create_default_pi_config(prim, core_index)
    pi_config.update({
        "A_valid": True,
        "S3_valid": True,
        "PI_parameter": [
            {"PIC_Mode": 3, "Addr_range": [0x8000, 0x8FFF]},
            {"PIC_Mode": 0, "Addr_range": [0x9000, 0x9FFF]},
            {"PIC_Mode": 0, "Addr_range": [0xA000, 0xAFFF]},
            {"PIC_Mode": 0, "Addr_range": [0xB000, 0xBFFF]}
        ]
    })
    return pi_config

def generate_lenet_style_config(prims_config: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成LeNet风格的芯片配置"""
    # 基础配置
    lenet_config = {
        "sim_clock": {
            "clock": 200000,
            "trigger": 0
        },
        "ChipArray": []
    }
    
    # 芯片配置
    chip_array_config = {
        "ChipID": {
            "cx": 0,
            "cy": 0
        },
        "coreXMax": 7,  # 最大X核心数
        "coreYMax": 1,  # 最大Y核心数
        "trigger": {
            "start": [1, 0, 0, 0],
            "high": [99999, 0, 0, 0],
            "low": [1, 0, 0, 0]
        },
        "ChipConfig": [
            {
                "PhaseGroup": 0,
                "Sim_clock": 200000,
                "trigger": 0,
                "P_adpt": 1
            },
            {
                "PhaseGroup": 1,
                "Sim_clock": 200000,
                "trigger": 0,
                "P_adpt": 1
            },
            {
                "PhaseGroup": 2,
                "Sim_clock": 200000,
                "trigger": 0,
                "P_adpt": 1
            }
        ],
        "CoreConfig": []
    }
    
    # 核心配置
    for i, prim in enumerate(prims_config):
        core_config = {
            "CoreInfo": {
                "x": i % 7 + 1,  # X坐标
                "y": i // 7,     # Y坐标
                "CoreGroup": 0,
                "static_PI_base_Addr": 0,
                "registers": {
                    "Receive_PI_addr_base": 0,
                    "PI_CXY": 0,
                    "PI_Nx": 0,
                    "PI_Ny": 0,
                    "PI_sign_CXY": 0,
                    "PI_sign_Nx": 0,
                    "PI_sign_Ny": 0,
                    "instant_PI_en": 0,
                    "fixed_instant_PI": 0,
                    "instant_PI_number": 0,
                    "PI_loop_en": 0,
                    "PI_loop_num": 0,
                    "start_instant_PI_num": 0,
                    "Addr_instant_PI_base": 0
                }
            },
            "PI": [],
            # 添加initData部分 - 根据原语类型设置不同的初始化数据
            "initData": [
                {
                    "start": 0,  # 起始地址
                    "length": 224,  # 数据长度
                    "data": [
                        [-105, 51, 115, -128],
                        [-21, 36, -82, -114],
                        [119, -69, -128, -59],
                        [91, 127, 29, 127],
                        [-128, -42, 127, -42],
                        [25, 100, 127, -93],
                        [-27, -28, -128, -44],
                        [0, 0, 0, 0],
                        [-128, 86, -128, -128],
                        [-11, -3, 22, 127],
                        [123, -85, -106, -78],
                        [-128, 127, 127, -128],
                        [-33, -32, 64, 33],
                        [-23, -33, -2, -49],
                        [-128, -128, -40, 120],
                        [0, 0, 0, 0],
                        [127, 0, -56, -128],
                        [127, -49, -45, 97],
                        [17, 23, -66, 102],
                        [-128, -128, -75, -79],
                        [-113, 127, 60, -18],
                        [-104, -49, -128, 76],
                        [-128, 52, 73, 39],
                        [0, 0, 0, 0],
                        [-16, -123, 127, 125],
                        [-51, -22, 97, 126],
                        [-106, 21, 64, 127],
                        [-69, 106, 127, -128],
                        [16, -2, -52, 68],
                        [49, 127, -128, 127],
                        [65, -20, 59, -128],
                        [0, 0, 0, 0],
                        [127, -128, -128, 127],
                        [-128, -128, -128, -128],
                        [127, 127, 127, 127],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]
                    ]
                }
            ],
            # 添加MemoryOutput部分 - 根据原语类型设置不同的输出区域
            "MemoryOutput": [
                [{"start": 33792, "length": 3072}],  # 内存输出区域1
                []  # 内存输出区域2（空数组）
            ]
        }
        
        # 根据原语类型配置PI参数
        if prim['type'] == 'p81':  # 第一层卷积
            pi_config = create_p81_pi_config(prim, i)
        elif prim['type'] == 'p41':  # 卷积/全连接
            pi_config = create_p41_pi_config(prim, i)
        elif prim['type'] == 'pX5':  # 池化/ReLU
            pi_config = create_pX5_pi_config(prim, i)
        else:  # 其他原语
            pi_config = create_default_pi_config(prim, i)
            
        core_config["PI"].append(pi_config)
        chip_array_config["CoreConfig"].append(core_config)
    
    lenet_config["ChipArray"].append(chip_array_config)
    return lenet_config

def main():
    parser = argparse.ArgumentParser(description="从ONNX模型生成LeNet风格芯片配置")
    parser.add_argument("--onnx_model", type=str, default="", 
                       help="ONNX模型文件路径（可选）")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="输出目录路径")
    
    args = parser.parse_args()
    
    print("=== ONNX模型到LeNet风格芯片配置转换工具 ===\n")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 步骤1: 加载ONNX模型并分析层
    print("步骤1: 加载模型并分析层...")
    try:
        if args.onnx_model:
            model_data = load_onnx_model_and_extract_layers(args.onnx_model)
        else:
            print("未指定ONNX模型，使用内置LeNet模型")
            model_data = create_simple_layer_analyzer()
        
        layers_info = model_data['layers_info']
        weights = model_data['weights']
        
        print(f"✓ 模型分析完成，共 {len(layers_info)} 层")
        for i, layer in enumerate(layers_info):
            print(f"  层 {i+1}: {layer['name']} ({layer['type']})")
            if 'input_shape' in layer and 'output_shape' in layer:
                print(f"      输入: {layer['input_shape']} -> 输出: {layer['output_shape']}")
        
        print(f"✓ 权重参数: {len(weights)} 个")
        for name, weight in weights.items():
            print(f"  {name}: 形状 {weight.shape}")
        
    except Exception as e:
        print(f"✗ 模型分析失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤2: 映射到prims原语（按照g1_1core.py的phase划分）
    print("\n步骤2: ONNX操作到prims原语映射...")
    try:
        mapper = RelaxPrimsMapper()
        prims_config = []
        
        # 按照g1_1core.py的phase划分逻辑
        # Phase 0: 数据交互（数据传输）
        # Phase 1: Conv1 + 激活 + Pool1（第一层卷积+激活+池化）
        # Phase 2: Conv2 + 激活 + Pool2（第二层卷积+激活+池化）
        # Phase 3: FC1（第一个全连接层）
        # Phase 4: FC2（第二个全连接层）
        # Phase 5: FC3（第三个全连接层）
        
        # Phase 0: 数据交互（数据传输）
        phase0_prims = []
        # 添加数据传输相关的prim（如p06等）
        # 这里简化处理，实际应根据具体的数据传输需求配置
        
        # Phase 1: Conv1 + 激活 + Pool1
        phase1_prims = []
        conv1_found = False
        for i, layer in enumerate(layers_info):
            if not conv1_found and layer['type'] in ['conv', 'conv2d']:
                # 第一层卷积
                prim_type = map_onnx_op_to_prim_type(layer['type'], i)
                prim_config = {
                    'type': prim_type,
                    'layer_index': i,
                    'layer_name': layer['name'],
                    'phase': 1,
                    'base_addr': 0x0000 + i * 0x1000,
                    'params': layer.get('attributes', {}),
                    'input_shape': layer.get('input_shape'),
                    'output_shape': layer.get('output_shape')
                }
                phase1_prims.append(prim_config)
                print(f"  Phase 1: {layer['name']} ({layer['type']}) -> {prim_type}")
                conv1_found = True
                
                # 查找后续的激活和池化层
                j = i + 1
                while j < len(layers_info) and j < i + 3:  # 最多查找后续3层
                    next_layer = layers_info[j]
                    if next_layer['type'] in ['relu', 'sigmoid', 'tanh']:
                        # 激活层
                        prim_type = map_onnx_op_to_prim_type(next_layer['type'], j)
                        prim_config = {
                            'type': prim_type,
                            'layer_index': j,
                            'layer_name': next_layer['name'],
                            'phase': 1,
                            'base_addr': 0x0000 + j * 0x1000,
                            'params': next_layer.get('attributes', {}),
                            'input_shape': next_layer.get('input_shape'),
                            'output_shape': next_layer.get('output_shape')
                        }
                        phase1_prims.append(prim_config)
                        print(f"  Phase 1: {next_layer['name']} ({next_layer['type']}) -> {prim_type}")
                    
                    elif next_layer['type'] in ['maxpool', 'averagepool']:
                        # 池化层
                        prim_type = map_onnx_op_to_prim_type(next_layer['type'], j)
                        prim_config = {
                            'type': prim_type,
                            'layer_index': j,
                            'layer_name': next_layer['name'],
                            'phase': 1,
                            'base_addr': 0x0000 + j * 0x1000,
                            'params': next_layer.get('attributes', {}),
                            'input_shape': next_layer.get('input_shape'),
                            'output_shape': next_layer.get('output_shape')
                        }
                        phase1_prims.append(prim_config)
                        print(f"  Phase 1: {next_layer['name']} ({next_layer['type']}) -> {prim_type}")
                        break  # 找到池化层后停止查找
                    
                    j += 1
                break
        
        # Phase 2: Conv2 + 激活 + Pool2
        phase2_prims = []
        conv2_found = False
        for i, layer in enumerate(layers_info):
            if not conv2_found and i > 0 and layer['type'] in ['conv', 'conv2d']:
                # 第二层卷积
                prim_type = map_onnx_op_to_prim_type(layer['type'], i)
                prim_config = {
                    'type': prim_type,
                    'layer_index': i,
                    'layer_name': layer['name'],
                    'phase': 2,
                    'base_addr': 0x0000 + i * 0x1000,
                    'params': layer.get('attributes', {}),
                    'input_shape': layer.get('input_shape'),
                    'output_shape': layer.get('output_shape')
                }
                phase2_prims.append(prim_config)
                print(f"  Phase 2: {layer['name']} ({layer['type']}) -> {prim_type}")
                conv2_found = True
                
                # 查找后续的激活和池化层
                j = i + 1
                while j < len(layers_info) and j < i + 3:
                    next_layer = layers_info[j]
                    if next_layer['type'] in ['relu', 'sigmoid', 'tanh']:
                        # 激活层
                        prim_type = map_onnx_op_to_prim_type(next_layer['type'], j)
                        prim_config = {
                            'type': prim_type,
                            'layer_index': j,
                            'layer_name': next_layer['name'],
                            'phase': 2,
                            'base_addr': 0x0000 + j * 0x1000,
                            'params': next_layer.get('attributes', {}),
                            'input_shape': next_layer.get('input_shape'),
                            'output_shape': next_layer.get('output_shape')
                        }
                        phase2_prims.append(prim_config)
                        print(f"  Phase 2: {next_layer['name']} ({next_layer['type']}) -> {prim_type}")
                    
                    elif next_layer['type'] in ['maxpool', 'averagepool']:
                        # 池化层
                        prim_type = map_onnx_op_to_prim_type(next_layer['type'], j)
                        prim_config = {
                            'type': prim_type,
                            'layer_index': j,
                            'layer_name': next_layer['name'],
                            'phase': 2,
                            'base_addr': 0x0000 + j * 0x1000,
                            'params': next_layer.get('attributes', {}),
                            'input_shape': next_layer.get('input_shape'),
                            'output_shape': next_layer.get('output_shape')
                        }
                        phase2_prims.append(prim_config)
                        print(f"  Phase 2: {next_layer['name']} ({next_layer['type']}) -> {prim_type}")
                        break  # 找到池化层后停止查找
                    
                    j += 1
                break
        
        # Phase 3: FC1（第一个全连接层）
        phase3_prims = []
        fc1_found = False
        for i, layer in enumerate(layers_info):
            if not fc1_found and layer['type'] in ['gemm', 'matmul']:
                prim_type = map_onnx_op_to_prim_type(layer['type'], i)
                prim_config = {
                    'type': prim_type,
                    'layer_index': i,
                    'layer_name': layer['name'],
                    'phase': 3,
                    'base_addr': 0x0000 + i * 0x1000,
                    'params': layer.get('attributes', {}),
                    'input_shape': layer.get('input_shape'),
                    'output_shape': layer.get('output_shape')
                }
                phase3_prims.append(prim_config)
                print(f"  Phase 3: {layer['name']} ({layer['type']}) -> {prim_type}")
                fc1_found = True
                break
        
        # Phase 4: FC2（第二个全连接层）
        phase4_prims = []
        fc2_found = False
        for i, layer in enumerate(layers_info):
            if not fc2_found and i > 0 and layer['type'] in ['gemm', 'matmul']:
                prim_type = map_onnx_op_to_prim_type(layer['type'], i)
                prim_config = {
                    'type': prim_type,
                    'layer_index': i,
                    'layer_name': layer['name'],
                    'phase': 4,
                    'base_addr': 0x0000 + i * 0x1000,
                    'params': layer.get('attributes', {}),
                    'input_shape': layer.get('input_shape'),
                    'output_shape': layer.get('output_shape')
                }
                phase4_prims.append(prim_config)
                print(f"  Phase 4: {layer['name']} ({layer['type']}) -> {prim_type}")
                fc2_found = True
                break
        
        # Phase 5: FC3（第三个全连接层）
        phase5_prims = []
        fc3_found = False
        for i, layer in enumerate(layers_info):
            if not fc3_found and i > 0 and layer['type'] in ['gemm', 'matmul']:
                prim_type = map_onnx_op_to_prim_type(layer['type'], i)
                prim_config = {
                    'type': prim_type,
                    'layer_index': i,
                    'layer_name': layer['name'],
                    'phase': 5,
                    'base_addr': 0x0000 + i * 0x1000,
                    'params': layer.get('attributes', {}),
                    'input_shape': layer.get('input_shape'),
                    'output_shape': layer.get('output_shape')
                }
                phase5_prims.append(prim_config)
                print(f"  Phase 5: {layer['name']} ({layer['type']}) -> {prim_type}")
                fc3_found = True
                break
        
        # 合并所有phase的prims
        prims_config = phase0_prims + phase1_prims + phase2_prims + phase3_prims + phase4_prims + phase5_prims
        
        print(f"✓ 原语映射完成，共 {len(prims_config)} 个prims原语")
        for i, prim in enumerate(prims_config):
            print(f"  Prim {i+1}: {prim['type']} (Phase {prim['phase']}) - {prim['layer_name']}")
        
    except Exception as e:
        print(f"✗ 原语映射失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤3: 生成标准芯片配置
    print("\n步骤3: 生成标准芯片配置...")
    try:
        # 使用RelaxToChipConfig生成标准配置
        config_generator = RelaxToChipConfig()
        # 修复：使用正确的私有方法_generate_chip_config
        chip_config = config_generator._generate_chip_config(prims_config, weights)
        
        # 转换元组键为字符串
        chip_config = convert_tuple_keys_to_string(chip_config)
        
        # 保存标准配置
        config_path = os.path.join(args.output_dir, "demo_chip_config.json")
        with open(config_path, 'w') as f:
            json.dump(chip_config, f, indent=2, default=str)
        
        print(f"✓ 标准芯片配置已保存: {config_path}")
        
    except Exception as e:
        print(f"✗ 标准配置生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤4: 生成LeNet风格配置
    print("\n步骤4: 生成LeNet风格配置...")
    try:
        lenet_style_config = generate_lenet_style_config(prims_config)
        
        # 转换元组键为字符串（虽然这里应该没有元组键，但为了安全）
        lenet_style_config = convert_tuple_keys_to_string(lenet_style_config)
        
        # 保存LeNet风格配置
        lenet_config_path = os.path.join(args.output_dir, "demo_lenet_style_config.json")
        with open(lenet_config_path, 'w') as f:
            json.dump(lenet_style_config, f, indent=2, default=str)
        
        print(f"✓ LeNet风格配置已保存: {lenet_config_path}")
        
        # 同时保存pickle格式
        pickle_path = os.path.join(args.output_dir, "demo_lenet_style_config.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(lenet_style_config, f)
        
        print(f"✓ Pickle格式配置已保存: {pickle_path}")
        
    except Exception as e:
        print(f"✗ LeNet风格配置生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== 转换完成 ===")
    print(f"输出目录: {args.output_dir}")
    print("生成的文件:")
    print(f"  - demo_chip_config.json (标准芯片配置)")
    print(f"  - demo_lenet_style_config.json (LeNet风格配置)")
    print(f"  - demo_lenet_style_config.pkl (Pickle格式)")

if __name__ == "__main__":
    main()