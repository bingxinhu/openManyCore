#!/usr/bin/env python3
"""
修复版演示脚本 v2 - 修复模块导入和JSON序列化问题
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generator.mapping_utils.relax_prims_mapper import RelaxPrimsMapper
from generator.mapping_utils.relax_to_chip_config import RelaxToChipConfig

def create_simple_layer_analyzer():
    """创建简单的层分析器（替代ModelLayerAnalyzer）"""
    class SimpleLayerAnalyzer:
        def analyze_model(self, model_path=None):
            """分析模型层结构"""
            # 返回预定义的层信息
            layers_info = [
                {"name": "conv1", "type": "conv2d", "input_shape": [1, 1, 28, 28], "output_shape": [1, 10, 26, 26]},
                {"name": "relu1", "type": "relu", "input_shape": [1, 10, 26, 26], "output_shape": [1, 10, 26, 26]},
                {"name": "pool1", "type": "max_pool2d", "input_shape": [1, 10, 26, 26], "output_shape": [1, 10, 13, 13]},
                {"name": "conv2", "type": "conv2d", "input_shape": [1, 10, 13, 13], "output_shape": [1, 20, 11, 11]},
                {"name": "relu2", "type": "relu", "input_shape": [1, 20, 11, 11], "output_shape": [1, 20, 11, 11]},
                {"name": "pool2", "type": "max_pool2d", "input_shape": [1, 20, 11, 11], "output_shape": [1, 20, 5, 5]},
                {"name": "flatten", "type": "reshape", "input_shape": [1, 20, 5, 5], "output_shape": [1, 500]},
                {"name": "fc1", "type": "dense", "input_shape": [1, 500], "output_shape": [1, 50]},
                {"name": "relu3", "type": "relu", "input_shape": [1, 50], "output_shape": [1, 50]},
                {"name": "fc2", "type": "dense", "input_shape": [1, 50], "output_shape": [1, 10]}
            ]
            return layers_info
    
    return SimpleLayerAnalyzer()

def main():
    print("=== 修复版演示脚本 v2 - Relax Prims映射和芯片配置生成 ===\n")
    
    # 步骤1: 加载模型并分析层
    print("步骤1: 加载模型并分析层...")
    try:
        analyzer = create_simple_layer_analyzer()
        layers_info = analyzer.analyze_model()
        
        print(f"✓ 模型层分析完成，共 {len(layers_info)} 层")
        for i, layer in enumerate(layers_info):
            print(f"  层 {i+1}: {layer['name']} ({layer['type']})")
        
    except Exception as e:
        print(f"✗ 模型分析失败: {e}")
        return
    
    # 步骤2: Relax IR到prims原语映射（按照g1_1core.py的phase划分）
    print("\n步骤2: Relax IR到prims原语映射...")
    try:
        mapper = RelaxPrimsMapper()
        prims_config = []
        
        # 按照g1_1core.py的phase划分逻辑
        # Phase 0: 数据交互（数据传输）
        # Phase 1: Conv1 + Maxpool1（第一层卷积+池化）
        # Phase 2: Conv2 + Maxpool2（第二层卷积+池化）
        # Phase 3: FC1（第一个全连接层）
        # Phase 4: FC2（第二个全连接层）
        # Phase 5: FC3（第三个全连接层）
        
        # Phase 0: 数据交互（数据传输）
        phase0_prims = []
        # 添加数据传输相关的prim（如p06等）
        # 这里简化处理，实际应根据具体的数据传输需求配置
        
        # Phase 1: Conv1 + Maxpool1
        phase1_prims = []
        for i, layer in enumerate(layers_info):
            if i == 0:  # 第一层卷积
                if layer['type'] == 'conv2d':
                    prim_type = "p81"  # 第一层卷积用p81
                    prim_config = {
                        'type': prim_type,
                        'layer_index': i,
                        'phase': 1,
                        'base_addr': 0x0000 + i * 0x1000,
                        'params': layer.get('params', {})
                    }
                    phase1_prims.append(prim_config)
                    print(f"  Phase 1: {layer['name']} ({layer['type']}) -> {prim_type}")
                    
                # 对应的池化层
                if i + 1 < len(layers_info) and layers_info[i+1]['type'] == 'max_pool2d':
                    pool_layer = layers_info[i+1]
                    prim_type = "pX5"  # 池化用pX5
                    prim_config = {
                        'type': prim_type,
                        'layer_index': i+1,
                        'phase': 1,
                        'base_addr': 0x0000 + (i+1) * 0x1000,
                        'params': pool_layer.get('params', {})
                    }
                    phase1_prims.append(prim_config)
                    print(f"  Phase 1: {pool_layer['name']} ({pool_layer['type']}) -> {prim_type}")
        
        # Phase 2: Conv2 + Maxpool2
        phase2_prims = []
        for i, layer in enumerate(layers_info):
            if i == 2:  # 第二层卷积（假设第3层是第二层卷积）
                if layer['type'] == 'conv2d':
                    prim_type = "p41"  # 后续卷积用p41
                    prim_config = {
                        'type': prim_type,
                        'layer_index': i,
                        'phase': 2,
                        'base_addr': 0x0000 + i * 0x1000,
                        'params': layer.get('params', {})
                    }
                    phase2_prims.append(prim_config)
                    print(f"  Phase 2: {layer['name']} ({layer['type']}) -> {prim_type}")
                    
                # 对应的池化层
                if i + 1 < len(layers_info) and layers_info[i+1]['type'] == 'max_pool2d':
                    pool_layer = layers_info[i+1]
                    prim_type = "pX5"  # 池化用pX5
                    prim_config = {
                        'type': prim_type,
                        'layer_index': i+1,
                        'phase': 2,
                        'base_addr': 0x0000 + (i+1) * 0x1000,
                        'params': pool_layer.get('params', {})
                    }
                    phase2_prims.append(prim_config)
                    print(f"  Phase 2: {pool_layer['name']} ({pool_layer['type']}) -> {prim_type}")
        
        # Phase 3-5: 全连接层
        phase3_prims = []
        phase4_prims = []
        phase5_prims = []
        
        # 修复：使用enumerate遍历原始layers_info来获取正确的索引
        fc_count = 0
        for i, layer in enumerate(layers_info):
            if layer['type'] == 'dense':
                phase_num = 3 + fc_count  # Phase 3, 4, 5对应FC1, FC2, FC3
                prim_type = "p41"  # 全连接用p41（1x1卷积实现）
                prim_config = {
                    'type': prim_type,
                    'layer_index': i,  # 使用原始索引i
                    'phase': phase_num,
                    'base_addr': 0x0000 + i * 0x1000,
                    'params': layer.get('params', {})
                }
                
                if phase_num == 3:
                    phase3_prims.append(prim_config)
                    print(f"  Phase 3: {layer['name']} ({layer['type']}) -> {prim_type}")
                elif phase_num == 4:
                    phase4_prims.append(prim_config)
                    print(f"  Phase 4: {layer['name']} ({layer['type']}) -> {prim_type}")
                elif phase_num == 5:
                    phase5_prims.append(prim_config)
                    print(f"  Phase 5: {layer['name']} ({layer['type']}) -> {prim_type}")
                
                fc_count += 1
        
        # 合并所有phase的prim配置
        prims_config.extend(phase0_prims)
        prims_config.extend(phase1_prims)
        prims_config.extend(phase2_prims)
        prims_config.extend(phase3_prims)
        prims_config.extend(phase4_prims)
        prims_config.extend(phase5_prims)
        
        print(f"✓ Prims映射完成，共 {len(prims_config)} 个原语配置，分布在6个phase中")
        
    except Exception as e:
        print(f"✗ Prims映射失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤3: 生成芯片配置
    print("\n步骤3: 生成芯片配置...")
    try:
        compiler = RelaxToChipConfig()
        
        # 创建模拟权重数据
        weights = {}
        for layer in layers_info:
            if layer['type'] in ['conv2d', 'dense']:
                weight_name = f"{layer['name']}_weight"
                # 模拟权重形状
                if layer['type'] == 'conv2d':
                    weights[weight_name] = np.random.randn(10, 1, 3, 3).astype(np.float32)
                else:  # dense
                    weights[weight_name] = np.random.randn(50, 500).astype(np.float32)
        
        # 修复：使用正确的_generate_chip_config方法而不是compile_from_relax
        chip_config = compiler._generate_chip_config(prims_config, weights)
        
        # 生成符合LeNet_002Config.json格式的配置
        lenet_style_config = generate_lenet_style_config(chip_config, prims_config)
        
        print("✓ 芯片配置生成完成")
        print(f"  芯片阵列: {chip_config.get('chip_array', 'N/A')}")
        print(f"  核心阵列: {chip_config.get('core_array', 'N/A')}")
        print(f"  相位数: {len(chip_config.get('phases', {}))}")
        print(f"  原语数: {len(chip_config.get('prims', []))}")
        
    except Exception as e:
        print(f"✗ 芯片配置生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤4: 显示配置详情
    print("\n步骤4: 配置详情...")
    try:
        # 显示配置结构
        print("配置结构:")
        for key in ['sim_clock', 'chip_array', 'core_array', 'phases', 'prims', 'weights']:
            value = chip_config.get(key, 'N/A')
            if key == 'phases':
                print(f"  {key}: {len(value)} phases")
                for chip_pos, phases in value.items():
                    print(f"    芯片 {chip_pos}: {len(phases)} 个相位")
            elif key == 'prims':
                print(f"  {key}: {len(value)} 个原语")
            elif key == 'weights':
                print(f"  {key}: {len(value)} 个权重张量")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"✗ 配置详情显示失败: {e}")
    
    # 步骤5: 保存配置
    print("\n步骤5: 保存配置...")
    try:
        # 保存为JSON和pickle格式
        json_path = "./demo_chip_config_final_v2.json"
        lenet_style_json_path = "./LeNet_Relax_Config.json"
        pkl_path = "./demo_chip_config_final_v2.pkl"
        
        # 使用修复后的save_config方法
        compiler.save_config(chip_config, pkl_path)
        
        # 额外保存为可读的JSON格式
        with open(json_path, 'w', encoding='utf-8') as f:
            # 使用修复后的序列化方法
            json_config = compiler._config_to_json_serializable(chip_config)
            json.dump(json_config, f, indent=2, ensure_ascii=False)
        
        # 保存LeNet风格的JSON配置
        with open(lenet_style_json_path, 'w', encoding='utf-8') as f:
            json.dump(lenet_style_config, f, indent=4, ensure_ascii=False)
        
        print(f"✓ 配置已保存:")
        print(f"  标准JSON格式: {json_path}")
        print(f"  LeNet风格JSON格式: {lenet_style_json_path}")
        print(f"  Pickle格式: {pkl_path}")
        
        # 显示文件大小
        if os.path.exists(json_path):
            size = os.path.getsize(json_path)
            print(f"  标准JSON文件大小: {size} 字节")
        
        if os.path.exists(lenet_style_json_path):
            size = os.path.getsize(lenet_style_json_path)
            print(f"  LeNet风格JSON文件大小: {size} 字节")
        
        if os.path.exists(pkl_path):
            size = os.path.getsize(pkl_path)
            print(f"  Pickle文件大小: {size} 字节")
            
    except Exception as e:
        print(f"✗ 配置保存失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== 演示完成 ===")

# 新增函数：生成LeNet风格的配置
def generate_lenet_style_config(chip_config, prims_config):
    """生成符合LeNet_002Config.json格式的配置"""
    
    # 基础配置
    lenet_config = {
        "sim_clock": 100000,  # 仿真时钟频率
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
            "PI": []
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

def create_p81_pi_config(prim, index):
    """创建p81原语的PI配置"""
    return {
        "A_valid": False,
        "S1_valid": True,
        "R_valid": True,
        "S2_valid": False,
        "PI_parameter": [
            {},
            {
                "PIC": 6,
                "PIC_Mode": 0,
                "Reset_Addr_in": 1,
                "Reset_Addr_out": 1,
                "Reset_Addr_ciso": 1,
                "Row_ck_on": 0,
                "Addr_Start_in": prim.get('base_addr', 0),
                "Addr_end_in": prim.get('base_addr', 0) + 767,
                "Addr_Start_out": prim.get('base_addr', 0) + 8448,
                "Addr_end_out": prim.get('base_addr', 0) + 9215,
                "Addr_Start_ciso": prim.get('base_addr', 0) + 4096,
                "Addr_end_ciso": prim.get('base_addr', 0) + 4107,
                "in_row_max": 0,
                "Km_num_in": 63,
                "Km_num_ciso": 0,
                "Km_num_out": 63,
                "num_in": 11,
                "num_ciso": 11,
                "num_out": 11,
                "type_in": 1,
                "type_out": 1,
                "in_cut_start": 0,
                "mem_sel": 0,
                "in_ciso_pipe_sel": 0
            },
            {
                "PIC": 9,
                "Rhead_mode": 1,
                "CXY": 0,
                "Send_en": 1,
                "Receive_en": 0,
                "Dout_Mem_sel": 1,
                "Addr_Dout_base": 1024,
                "Addr_Dout_length": 0,
                "Addr_Rhead_base": 192,
                "Addr_Rhead_length": 0,
                "Addr_Din_base": 448,
                "Addr_Din_length": 1,
                "Send_number": 111,
                "Receive_number": 0,
                "Nx": 0,
                "Ny": 0,
                "Send_PI_en": 0,
                "Back_Sign_en": 0,
                "Send_PI_num": 0,
                "Receive_sign_num": 0,
                "Send_PI_addr_base": 0,
                "Relay_number": 0,
                "Q": 0,
                "T_mode": 1,
                "Receive_sign_en": 0,
                "Soma_in_en": 1
            },
            {}
        ],
        "Additional": [
            {},
            {
                "Read_in_length": 3072,
                "Read_ciso_length": 48,
                "Write_out_length": 3072
            },
            {},
            {}
        ],
        "Addr": [
            {},
            {
                "Addr_Start_in": prim.get('base_addr', 0),
                "Addr_end_in": prim.get('base_addr', 0) + 3068,
                "Addr_Start_out": prim.get('base_addr', 0) + 33792,
                "Addr_end_out": prim.get('base_addr', 0) + 36860,
                "Addr_Start_ciso": prim.get('base_addr', 0) + 16384,
                "Addr_end_ciso": prim.get('base_addr', 0) + 16428
            },
            {
                "Addr_Dout_base": prim.get('base_addr', 0) + 36864,
                "Addr_Dout_end": prim.get('base_addr', 0) + 36864,
                "Addr_Rhead_base": prim.get('base_addr', 0) + 33536,
                "Addr_Rhead_end": prim.get('base_addr', 0) + 33536,
                "Addr_Din_base": prim.get('base_addr', 0) + 33664,
                "Addr_Din_end": prim.get('base_addr', 0) + 33666
            },
            {}
        ],
        "Data": [{}, {}, {}, {}]
    }

def create_p41_pi_config(prim, index):
    """创建p41原语的PI配置"""
    return {
        "A_valid": False,
        "S1_valid": True,
        "R_valid": True,
        "S2_valid": False,
        "PI_parameter": [
            {},
            {
                "PIC": 6,
                "PIC_Mode": 0,
                "Reset_Addr_in": 1,
                "Reset_Addr_out": 1,
                "Reset_Addr_ciso": 1,
                "Row_ck_on": 0,
                "Addr_Start_in": prim.get('base_addr', 0),
                "Addr_end_in": prim.get('base_addr', 0) + 55,
                "Addr_Start_out": prim.get('base_addr', 0) + 9216,
                "Addr_end_out": prim.get('base_addr', 0) + 9216,
                "Addr_Start_ciso": 0,
                "Addr_end_ciso": 27,
                "in_row_max": 0,
                "Km_num_in": 1,
                "Km_num_ciso": 0,
                "Km_num_out": 1,
                "num_in": 27,
                "num_ciso": 27,
                "num_out": 27,
                "type_in": 1,
                "type_out": 1,
                "in_cut_start": 0,
                "mem_sel": 1,
                "in_ciso_pipe_sel": 0
            },
            {
                "PIC": 9,
                "Rhead_mode": 1,
                "CXY": 0,
                "Send_en": 1,
                "Receive_en": 0,
                "Dout_Mem_sel": 1,
                "Addr_Dout_base": 1024,
                "Addr_Dout_length": 0,
                "Addr_Rhead_base": 192,
                "Addr_Rhead_length": 0,
                "Addr_Din_base": 448,
                "Addr_Din_length": 1,
                "Send_number": 111,
                "Receive_number": 0,
                "Nx": 0,
                "Ny": 0,
                "Send_PI_en": 0,
                "Back_Sign_en": 0,
                "Send_PI_num": 0,
                "Receive_sign_num": 0,
                "Send_PI_addr_base": 0,
                "Relay_number": 0,
                "Q": 0,
                "T_mode": 1,
                "Receive_sign_en": 0,
                "Soma_in_en": 1
            },
            {}
        ],
        "Additional": [
            {},
            {
                "Read_in_length": 224,
                "Read_ciso_length": 112,
                "Write_out_length": 224
            },
            {},
            {}
        ],
        "Addr": [
            {},
            {
                "Addr_Start_in": prim.get('base_addr', 0),
                "Addr_end_in": prim.get('base_addr', 0) + 220,
                "Addr_Start_out": prim.get('base_addr', 0) + 36864,
                "Addr_end_out": prim.get('base_addr', 0) + 37084,
                "Addr_Start_ciso": 0,
                "Addr_end_ciso": 108
            },
            {
                "Addr_Dout_base": prim.get('base_addr', 0) + 36864,
                "Addr_Dout_end": prim.get('base_addr', 0) + 36864,
                "Addr_Rhead_base": prim.get('base_addr', 0) + 33536,
                "Addr_Rhead_end": prim.get('base_addr', 0) + 33536,
                "Addr_Din_base": prim.get('base_addr', 0) + 33664,
                "Addr_Din_end": prim.get('base_addr', 0) + 33666
            },
            {}
        ],
        "Data": [{}, {}, {}, {}]
    }

def create_pX5_pi_config(prim, index):
    """创建pX5原语的PI配置"""
    return {
        "A_valid": False,
        "S1_valid": True,
        "R_valid": True,
        "S2_valid": False,
        "PI_parameter": [
            {},
            {
                "PIC": 6,
                "PIC_Mode": 1,  # 池化模式
                "Reset_Addr_in": 1,
                "Reset_Addr_out": 1,
                "Reset_Addr_ciso": 1,
                "Row_ck_on": 0,
                "Addr_Start_in": prim.get('base_addr', 0),
                "Addr_end_in": prim.get('base_addr', 0) + 127,
                "Addr_Start_out": prim.get('base_addr', 0) + 4096,
                "Addr_end_out": prim.get('base_addr', 0) + 4223,
                "Addr_Start_ciso": 0,
                "Addr_end_ciso": 63,
                "in_row_max": 0,
                "Km_num_in": 31,
                "Km_num_ciso": 0,
                "Km_num_out": 31,
                "num_in": 15,
                "num_ciso": 15,
                "num_out": 15,
                "type_in": 1,
                "type_out": 1,
                "in_cut_start": 0,
                "mem_sel": 0,
                "in_ciso_pipe_sel": 0
            },
            {},
            {}
        ],
        "Additional": [
            {},
            {
                "Read_in_length": 512,
                "Read_ciso_length": 64,
                "Write_out_length": 512
            },
            {},
            {}
        ],
        "Addr": [
            {},
            {
                "Addr_Start_in": prim.get('base_addr', 0),
                "Addr_end_in": prim.get('base_addr', 0) + 508,
                "Addr_Start_out": prim.get('base_addr', 0) + 16384,
                "Addr_end_out": prim.get('base_addr', 0) + 16892,
                "Addr_Start_ciso": 0,
                "Addr_end_ciso": 252
            },
            {},
            {}
        ],
        "Data": [{}, {}, {}, {}]
    }

def create_default_pi_config(prim, index):
    """创建默认PI配置"""
    return {
        "A_valid": False,
        "S1_valid": False,
        "R_valid": False,
        "S2_valid": False,
        "PI_parameter": [{}, {}, {}, {}],
        "Additional": [{}, {}, {}, {}],
        "Addr": [{}, {}, {}, {}],
        "Data": [{}, {}, {}, {}]
    }

if __name__ == "__main__":
    main()