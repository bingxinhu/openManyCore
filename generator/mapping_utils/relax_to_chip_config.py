"""
Relax IR到芯片核心配置的完整流程
集成Relax IR映射、prims生成和芯片配置生成
"""

import tvm
import onnx
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from .relax_prims_mapper import RelaxPrimsMapper, SimpleRelaxPrimsMapper
from .map_config_gen import MapConfigGen
from .prims import p09  # 路由器原语

class RelaxToChipConfig:
    """Relax IR到芯片核心配置的完整转换器"""
    
    def __init__(self, chip_config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.mapper = RelaxPrimsMapper()
        self.simple_mapper = SimpleRelaxPrimsMapper()
        self.config_gen = MapConfigGen()
        
        # 默认芯片配置
        self.chip_config = chip_config or {
            'chip_array': (1, 1),  # 芯片阵列大小 (x, y)
            'core_array': (16, 10),  # 核心阵列大小 (x, y)
            'memory_config': {
                'input_base': 0x0000,
                'output_base': 0x1000,
                'weight_base': 0x2000
            }
        }
    
    def compile_from_relax(self, mod: tvm.IRModule, input_shape: Tuple[int, ...], 
                          model_path: str = None) -> Dict[str, Any]:
        """
        从Relax IR编译生成芯片配置
        
        Args:
            mod: TVM Relax IR模块
            input_shape: 输入形状
            model_path: 可选，ONNX模型路径（用于提取权重）
            
        Returns:
            完整的芯片配置字典
        """
        self.logger.info("开始Relax IR到芯片配置的编译流程...")
        
        # 步骤1: 映射Relax IR到prims原语
        self.logger.info("步骤1: 映射Relax IR到prims原语...")
        prims_config = self.mapper.map_relax_to_prims(mod, input_shape)
        
        # 步骤2: 提取权重信息（如果提供了模型路径）
        weights = {}
        if model_path:
            self.logger.info("步骤2: 从ONNX模型提取权重...")
            weights = self._extract_weights_from_onnx(model_path)
        
        # 步骤3: 生成芯片核心配置
        self.logger.info("步骤3: 生成芯片核心配置...")
        chip_config = self._generate_chip_config(prims_config, weights)
        
        # 步骤4: 添加路由信息
        self.logger.info("步骤4: 添加路由配置...")
        self._add_router_config(chip_config)
        
        self.logger.info("芯片配置生成完成!")
        return chip_config
    
    def compile_from_onnx(self, model_path: str, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        从ONNX模型直接编译生成芯片配置
        
        Args:
            model_path: ONNX模型路径
            input_shape: 输入形状
            
        Returns:
            完整的芯片配置字典
        """
        self.logger.info(f"从ONNX模型编译: {model_path}")
        
        # 加载ONNX模型
        onnx_model = onnx.load(model_path)
        
        # 转换为Relax IR
        try:
            from tvm.relax.frontend.onnx import from_onnx
            mod = from_onnx(onnx_model, shape_dict={'input': input_shape})
            if not isinstance(mod, tvm.IRModule):
                mod = tvm.IRModule.from_expr(mod)
        except Exception as e:
            self.logger.error(f"ONNX到Relax IR转换失败: {e}")
            # 使用简化方法
            return self._compile_simple_from_onnx(onnx_model, input_shape)
        
        return self.compile_from_relax(mod, input_shape, model_path)
    
    def _compile_simple_from_onnx(self, onnx_model: onnx.ModelProto, 
                                 input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """简化版ONNX到芯片配置编译"""
        self.logger.info("使用简化版编译流程...")
        
        # 直接从ONNX提取层信息
        layer_info = self._extract_layers_from_onnx(onnx_model)
        
        # 简化映射
        prims_config = self.simple_mapper.map_simple(layer_info)
        
        # 提取权重
        weights = self._extract_weights_from_onnx_model(onnx_model)
        
        # 生成配置
        chip_config = self._generate_simple_chip_config(prims_config, weights)
        
        return chip_config
    
    def _extract_layers_from_onnx(self, onnx_model: onnx.ModelProto) -> List[Dict[str, Any]]:
        """从ONNX模型提取层信息"""
        layers = []
        
        for node in onnx_model.graph.node:
            layer_info = {
                'name': node.name,
                'type': node.op_type.lower(),
                'inputs': [str(input_name) for input_name in node.input],
                'outputs': [str(output_name) for output_name in node.output],
                'attrs': {}
            }
            
            # 提取属性
            for attr in node.attribute:
                layer_info['attrs'][attr.name] = self._extract_onnx_attr_value(attr)
            
            layers.append(layer_info)
        
        return layers
    
    def _extract_onnx_attr_value(self, attr):
        """提取ONNX属性值"""
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.TENSOR:
            return onnx.numpy_helper.to_array(attr.t)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        else:
            return None
    
    def _extract_weights_from_onnx(self, model_path: str) -> Dict[str, np.ndarray]:
        """从ONNX文件提取权重"""
        onnx_model = onnx.load(model_path)
        return self._extract_weights_from_onnx_model(onnx_model)
    
    def _extract_weights_from_onnx_model(self, onnx_model: onnx.ModelProto) -> Dict[str, np.ndarray]:
        """从ONNX模型对象提取权重"""
        weights = {}
        for init in onnx_model.graph.initializer:
            weights[init.name] = onnx.numpy_helper.to_array(init)
        return weights
    
    def _generate_chip_config(self, prims_config: List[Dict], weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """生成芯片核心配置"""
        chip_x, chip_y = self.chip_config['chip_array']
        core_x, core_y = self.chip_config['core_array']
        
        # 初始化配置结构
        config = {
            'sim_clock': 100000,  # 仿真时钟
            'chip_array': (chip_x, chip_y),
            'core_array': (core_x, core_y),
            'phases': {},
            'prims': prims_config,
            'weights': weights
        }
        
        # 为每个芯片和核心分配配置
        for chip_idx in range(chip_x * chip_y):
            chip_pos = (chip_idx // chip_y, chip_idx % chip_y)
            config['phases'][chip_pos] = {}
            
            # 分配核心给不同的层
            num_cores = core_x * core_y
            num_layers = len(prims_config)
            cores_per_layer = max(1, num_cores // num_layers)
            
            for layer_idx, prim_config in enumerate(prims_config):
                phase_id = layer_idx
                config['phases'][chip_pos][phase_id] = {
                    'clock': 15000,  # 相位时钟
                    'mode': 1,  # 自适应模式
                    'cores': {}
                }
                
                # 分配核心
                for core_offset in range(cores_per_layer):
                    if layer_idx * cores_per_layer + core_offset >= num_cores:
                        break
                        
                    core_idx = layer_idx * cores_per_layer + core_offset
                    core_x_pos = core_idx // core_y
                    core_y_pos = core_idx % core_y
                    core_pos = (core_x_pos, core_y_pos)
                    
                    config['phases'][chip_pos][phase_id]['cores'][core_pos] = {
                        'prims': [prim_config],
                        'router': None  # 稍后添加路由
                    }
        
        return config
    
    def _generate_simple_chip_config(self, prims_config: List[Dict], 
                                   weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """生成简化版芯片配置"""
        config = {
            'sim_clock': 100000,
            'chip_array': (1, 1),
            'core_array': (4, 4),  # 简化版使用4x4核心
            'phases': {
                (0, 0): {
                    0: {
                        'clock': 15000,
                        'mode': 1,
                        'cores': {}
                    }
                }
            },
            'prims': prims_config,
            'weights': weights
        }
        
        # 为每个原语分配一个核心
        for i, prim_config in enumerate(prims_config):
            if i >= 16:  # 最多16个核心
                break
                
            core_x = i // 4
            core_y = i % 4
            core_pos = (core_x, core_y)
            
            config['phases'][(0, 0)][0]['cores'][core_pos] = {
                'prims': [prim_config],
                'router': None
            }
        
        return config
    
    def _add_router_config(self, chip_config: Dict[str, Any]):
        """添加路由配置"""
        chip_x, chip_y = chip_config['chip_array']
        core_x, core_y = chip_config['core_array']
        
        # 为每个芯片和相位添加路由配置
        for chip_pos in chip_config['phases']:
            for phase_id in chip_config['phases'][chip_pos]:
                phase_config = chip_config['phases'][chip_pos][phase_id]
                
                for core_pos in phase_config['cores']:
                    core_config = phase_config['cores'][core_pos]
                    
                    # 添加简单的路由器配置
                    router_config = p09(
                        rhead_mode=1,
                        send_en=1,
                        receive_en=1,
                        send_num=1,
                        addr_din_length=0,
                        addr_rhead_base=0x300,
                        receive_num=1,
                        addr_rhead_length=0,
                        addr_dout_base=0x1000,
                        addr_dout_length=1,
                        addr_din_base=0x400
                    )
                    
                    core_config['router'] = router_config
        
        # 使用MapConfigGen添加路由信息
        try:
            MapConfigGen.add_router_config(
                chip_config,
                core_x_num=core_x,
                core_y_num=core_y
            )
        except Exception as e:
            self.logger.warning(f"路由配置添加失败: {e}")
    
    def save_config(self, config: Dict[str, Any], output_path: str):
        """保存配置到文件"""
        import json
        import pickle
        
        # 保存为JSON（可读格式）
        json_path = output_path.replace('.pkl', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            # 处理不可序列化的对象
            json_config = self._config_to_json_serializable(config)
            json.dump(json_config, f, indent=2, ensure_ascii=False)
        
        # 保存为pickle（完整对象）
        pkl_path = output_path if output_path.endswith('.pkl') else output_path + '.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(config, f)
        
        self.logger.info(f"配置已保存: {json_path}, {pkl_path}")
    
    def _config_to_json_serializable(self, config: Dict) -> Dict:
        """将配置转换为JSON可序列化格式"""
        json_config = {}
        
        for key, value in config.items():
            if key == 'prims':
                # 处理prims配置
                json_config[key] = []
                for prim in value:
                    if isinstance(prim, dict):
                        json_prim = {}
                        for k, v in prim.items():
                            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                                json_prim[k] = v
                            else:
                                json_prim[k] = str(v)
                        json_config[key].append(json_prim)
            elif key == 'weights':
                # 权重数据太大，只保存元数据
                json_config[key] = {name: f"array{array.shape}" for name, array in value.items()}
            elif key == 'phases':
                # 处理phases配置，将元组键转换为字符串
                json_config[key] = {}
                for chip_pos, phases in value.items():
                    # 将元组键转换为字符串，如 "(0, 0)"
                    chip_key = str(chip_pos) if isinstance(chip_pos, tuple) else chip_pos
                    json_config[key][chip_key] = {}
                    
                    for phase_id, phase_config in phases.items():
                        json_config[key][chip_key][phase_id] = {}
                        
                        for phase_key, phase_value in phase_config.items():
                            if phase_key == 'cores':
                                # 处理cores中的元组键
                                json_config[key][chip_key][phase_id][phase_key] = {}
                                for core_pos, core_config in phase_value.items():
                                    core_key = str(core_pos) if isinstance(core_pos, tuple) else core_pos
                                    json_config[key][chip_key][phase_id][phase_key][core_key] = core_config
                            else:
                                json_config[key][chip_key][phase_id][phase_key] = phase_value
            else:
                json_config[key] = value
        
        return json_config

# 使用示例
def create_demo_config():
    """创建演示配置"""
    compiler = RelaxToChipConfig()
    
    # 创建简单的演示配置
    demo_config = {
        'sim_clock': 100000,
        'chip_array': (1, 1),
        'core_array': (4, 4),
        'phases': {
            (0, 0): {
                0: {
                    'clock': 15000,
                    'mode': 1,
                    'cores': {
                        (0, 0): {
                            'prims': [{
                                'type': 'p81',
                                'params': {'cin': 1, 'cout': 10, 'kernel_x': 3, 'kernel_y': 3}
                            }],
                            'router': None
                        }
                    }
                }
            }
        },
        'prims': [{
            'type': 'p81',
            'layer_index': 0,
            'base_addr': 0x0000,
            'params': {'cin': 1, 'cout': 10, 'kernel_x': 3, 'kernel_y': 3}
        }],
        'weights': {'conv1_weight': 'array[10,1,3,3]'}
    }
    
    return demo_config