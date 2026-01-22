# tvmauto_mapper_v20.py - 修复JSON序列化问题的版本
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax import transform
from tvm.runtime import Device, cpu
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """硬件配置信息"""
    # 内存配置
    mem0_size: int = 64 * 1024  # 64KB
    mem1_size: int = 64 * 1024  # 64KB
    mem2_size: int = 16 * 1024  # 16KB
    mem3_size: int = 16  # 16B
    
    # 核心配置
    core_grid: Tuple[int, int] = (8, 1)  # 8x1核心阵列
    core_mem_banks: int = 4  # 每个核心4个内存bank
    
    # 计算单元
    mac_units_per_core: int = 64  # 每核心MAC单元数
    vector_width: int = 16  # 向量处理宽度
    
    # 路由
    router_channels: int = 8
    max_packet_size: int = 1024
    
    @property
    def total_cores(self) -> int:
        return self.core_grid[0] * self.core_grid[1]


class JSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理元组等非标准JSON类型"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        else:
            return super().default(obj)
    
    def encode(self, obj):
        # 递归转换字典键
        def convert_keys(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    # 转换键为字符串
                    if isinstance(k, tuple):
                        new_key = str(k)
                    elif isinstance(k, (int, float, bool, type(None))):
                        new_key = str(k)
                    else:
                        new_key = k
                    new_dict[new_key] = convert_keys(v)
                return new_dict
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_keys(item) for item in obj]
            else:
                return obj
        
        # 先转换键，再编码
        converted_obj = convert_keys(obj)
        return super().encode(converted_obj)


class ONNXModelAnalyzer:
    """ONNX模型分析器 - 不依赖TVM的纯Python分析"""
    
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.model = onnx.load(onnx_path)
        self.graph = self.model.graph
        
        # 提取模型信息
        self.input_info = self._extract_input_info()
        self.output_info = self._extract_output_info()
        self.operators = self._extract_operators()
        self.tensors = self._extract_tensors()
        
    def _extract_input_info(self) -> List[Dict]:
        """提取输入信息"""
        inputs = []
        for input_proto in self.graph.input:
            # 跳过已经在initializer中的输入（这些是权重）
            if input_proto.name not in [init.name for init in self.graph.initializer]:
                shape = []
                for dim in input_proto.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(1)  # 未知维度设为1
                
                inputs.append({
                    'name': input_proto.name,
                    'shape': shape,
                    'dtype': self._onnx_dtype_to_str(input_proto.type.tensor_type.elem_type)
                })
        return inputs
    
    def _extract_output_info(self) -> List[Dict]:
        """提取输出信息"""
        outputs = []
        for output_proto in self.graph.output:
            shape = []
            for dim in output_proto.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)
            
            outputs.append({
                'name': output_proto.name,
                'shape': shape,
                'dtype': self._onnx_dtype_to_str(output_proto.type.tensor_type.elem_type)
            })
        return outputs
    
    def _extract_operators(self) -> List[Dict]:
        """提取算子信息"""
        operators = []
        for node in self.graph.node:
            op_info = {
                'name': node.name,
                'type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {attr.name: self._parse_attribute(attr) for attr in node.attribute}
            }
            operators.append(op_info)
        return operators
    
    def _extract_tensors(self) -> Dict[str, Dict]:
        """提取所有张量信息"""
        tensors = {}
        
        # 初始权重和偏置
        for init in self.graph.initializer:
            tensor_data = onnx.numpy_helper.to_array(init)
            tensors[init.name] = {
                'name': init.name,
                'shape': list(tensor_data.shape),
                'dtype': str(tensor_data.dtype),
                'size_in_bytes': tensor_data.nbytes,
                'is_parameter': True,
                'data': tensor_data,
                'data_type': 'weight' if len(tensor_data.shape) > 1 else 'bias'
            }
        
        # 输入输出张量
        for info_list in [self.input_info, self.output_info]:
            for info in info_list:
                if info['name'] not in tensors:
                    tensors[info['name']] = {
                        'name': info['name'],
                        'shape': info['shape'],
                        'dtype': info['dtype'],
                        'size_in_bytes': self._calculate_size(info['shape'], info['dtype']),
                        'is_parameter': False,
                        'data': None,
                        'data_type': 'input' if info_list is self.input_info else 'output'
                    }
        
        # 中间张量（通过算子输出推断）
        for op in self.operators:
            for output_name in op['outputs']:
                if output_name not in tensors:
                    # 推断形状和类型
                    inferred_shape = self._infer_tensor_shape(output_name, op, tensors)
                    if inferred_shape:
                        tensors[output_name] = {
                            'name': output_name,
                            'shape': inferred_shape,
                            'dtype': 'float32',  # 默认float32
                            'size_in_bytes': self._calculate_size(inferred_shape, 'float32'),
                            'is_parameter': False,
                            'is_intermediate': True,
                            'data': None,
                            'data_type': 'intermediate'
                        }
        
        return tensors
    
    def _onnx_dtype_to_str(self, dtype_id: int) -> str:
        """ONNX数据类型ID转字符串"""
        dtype_map = {
            1: 'float32',
            2: 'uint8',
            3: 'int8',
            6: 'int32',
            7: 'int64',
            9: 'bool',
            10: 'float16',
            11: 'double'
        }
        return dtype_map.get(dtype_id, 'float32')
    
    def _parse_attribute(self, attr) -> Any:
        """解析ONNX属性"""
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        else:
            try:
                return str(attr)
            except:
                return None
    
    def _calculate_size(self, shape: List[int], dtype: str) -> int:
        """计算张量字节大小"""
        if not shape:
            return 0
            
        dtype_size = {
            'float32': 4, 'int32': 4, 'uint32': 4,
            'int8': 1, 'uint8': 1, 'float16': 2,
            'float64': 8, 'int64': 8
        }
        
        # 计算总元素数
        total_elements = 1
        for dim in shape:
            if dim > 0:
                total_elements *= dim
        
        return total_elements * dtype_size.get(dtype, 4)
    
    def _infer_tensor_shape(self, tensor_name: str, op_info: Dict, tensors: Dict) -> Optional[List[int]]:
        """推断张量形状"""
        op_type = op_info['type']
        
        try:
            if op_type == 'Conv':
                # 卷积输出形状推断
                input_name = op_info['inputs'][0]
                weight_name = op_info['inputs'][1]
                
                if input_name not in tensors or weight_name not in tensors:
                    return None
                
                input_shape = tensors[input_name]['shape']
                weight_shape = tensors[weight_name]['shape']
                
                # 获取卷积属性
                strides = op_info['attributes'].get('strides', [1, 1])
                pads = op_info['attributes'].get('pads', [0, 0, 0, 0])
                dilations = op_info['attributes'].get('dilations', [1, 1])
                group = op_info['attributes'].get('group', 1)
                
                # 计算输出高度和宽度
                h_in = input_shape[2] if len(input_shape) > 2 else 1
                w_in = input_shape[3] if len(input_shape) > 3 else 1
                
                h_kernel = weight_shape[2] if len(weight_shape) > 2 else 1
                w_kernel = weight_shape[3] if len(weight_shape) > 3 else 1
                
                h_out = (h_in + pads[0] + pads[2] - 
                        dilations[0] * (h_kernel - 1) - 1) // strides[0] + 1
                w_out = (w_in + pads[1] + pads[3] - 
                        dilations[1] * (w_kernel - 1) - 1) // strides[1] + 1
                
                return [input_shape[0], weight_shape[0], h_out, w_out]
            
            elif op_type == 'MaxPool':
                # 池化输出形状推断
                input_name = op_info['inputs'][0]
                if input_name not in tensors:
                    return None
                
                input_shape = tensors[input_name]['shape']
                kernel_shape = op_info['attributes'].get('kernel_shape', [2, 2])
                strides = op_info['attributes'].get('strides', kernel_shape)
                
                if len(input_shape) < 3:
                    return None
                
                h_in = input_shape[2]
                w_in = input_shape[3] if len(input_shape) > 3 else 1
                
                h_out = (h_in - kernel_shape[0]) // strides[0] + 1
                w_out = (w_in - kernel_shape[1]) // strides[1] + 1
                
                return [input_shape[0], input_shape[1], h_out, w_out]
            
            elif op_type == 'Relu' or op_type == 'Sigmoid' or op_type == 'Tanh':
                # 逐元素操作，形状不变
                input_name = op_info['inputs'][0]
                if input_name in tensors:
                    return tensors[input_name]['shape'].copy()
            
            elif op_type == 'Add' or op_type == 'Mul':
                # 逐元素操作，形状不变（广播情况暂时忽略）
                input_name = op_info['inputs'][0]
                if input_name in tensors:
                    return tensors[input_name]['shape'].copy()
            
            elif op_type == 'Reshape':
                # 重塑操作
                shape_attr = op_info['attributes'].get('shape', [])
                if shape_attr:
                    return list(shape_attr)
            
            elif op_type == 'Gemm':
                # 全连接层
                input_name = op_info['inputs'][0]
                weight_name = op_info['inputs'][1]
                
                if input_name not in tensors or weight_name not in tensors:
                    return None
                
                input_shape = tensors[input_name]['shape']
                weight_shape = tensors[weight_name]['shape']
                
                # Gemm: Y = alpha * A * B + beta * C
                # 这里简化处理
                return [input_shape[0], weight_shape[0]]
            
            elif op_type == 'BatchNormalization':
                # BN层，形状不变
                input_name = op_info['inputs'][0]
                if input_name in tensors:
                    return tensors[input_name]['shape'].copy()
            
            elif op_type == 'Concat':
                # 拼接操作
                input_shapes = []
                for input_name in op_info['inputs']:
                    if input_name in tensors:
                        input_shapes.append(tensors[input_name]['shape'])
                
                if not input_shapes:
                    return None
                
                # 简单处理：假设在指定轴拼接
                axis = op_info['attributes'].get('axis', 0)
                result_shape = input_shapes[0].copy()
                
                for shape in input_shapes[1:]:
                    if len(shape) > axis:
                        result_shape[axis] += shape[axis]
                
                return result_shape
            
        except Exception as e:
            logger.warning(f"推断形状时出错: {e}")
        
        return None
    
    def analyze(self) -> Dict:
        """分析模型并返回完整信息"""
        return {
            'model_info': {
                'ir_version': self.model.ir_version,
                'opset_import': [str(opset) for opset in self.model.opset_import],
                'producer_name': self.model.producer_name
            },
            'computation_graph': {
                'nodes': self.operators,
                'tensors': self.tensors,
                'inputs': self.input_info,
                'outputs': self.output_info
            },
            'statistics': {
                'total_operators': len(self.operators),
                'total_tensors': len(self.tensors),
                'total_parameters': sum(1 for t in self.tensors.values() if t['is_parameter']),
                'total_parameters_size': sum(t['size_in_bytes'] for t in self.tensors.values() if t['is_parameter']),
                'total_activations_size': sum(t['size_in_bytes'] for t in self.tensors.values() if not t['is_parameter'])
            }
        }


class AutoMemoryAllocator:
    """自动内存分配器"""
    
    def __init__(self, hw_config: HardwareConfig):
        self.hw_config = hw_config
        self.mem_blocks = {
            'mem0': {'base': 0x0000, 'size': hw_config.mem0_size, 'current': 0x0000, 'type': 'data'},
            'mem1': {'base': 0x4000, 'size': hw_config.mem1_size, 'current': 0x4000, 'type': 'weight'},
            'mem2': {'base': 0x8000, 'size': hw_config.mem2_size, 'current': 0x8000, 'type': 'io'},
            'mem3': {'base': 0x9000, 'size': hw_config.mem3_size, 'current': 0x9000, 'type': 'control'}
        }
        self.allocations = {}  # tensor_name -> {type, addr, size, mem_bank}
        
    def allocate(self, tensor_info: Dict, mem_type: str = 'auto') -> Dict:
        """分配内存"""
        tensor_name = tensor_info['name']
        
        # 如果已经分配过，直接返回
        if tensor_name in self.allocations:
            return self.allocations[tensor_name]
        
        tensor_size = tensor_info.get('size_in_bytes', 0)
        tensor_type = tensor_info.get('dtype', 'float32')
        
        # 确定内存类型
        if mem_type == 'auto':
            mem_type = self._infer_mem_type(tensor_info)
        
        mem_block = self.mem_blocks[mem_type]
        
        # 对齐要求
        alignment = self._get_alignment(mem_type, tensor_info)
        aligned_size = self._align_size(tensor_size, alignment)
        
        # 检查是否有足够空间
        if mem_block['current'] + aligned_size > mem_block['base'] + mem_block['size']:
            # 尝试其他内存块
            for alt_mem in ['mem0', 'mem1', 'mem2']:
                if alt_mem != mem_type:
                    alt_block = self.mem_blocks[alt_mem]
                    if alt_block['current'] + aligned_size <= alt_block['base'] + alt_block['size']:
                        mem_type = alt_mem
                        mem_block = alt_block
                        break
            else:
                raise MemoryError(f"内存不足，无法分配 {tensor_name} (大小: {aligned_size} 字节)")
        
        # 分配地址（考虑对齐）
        addr = self._align_address(mem_block['current'], alignment)
        mem_block['current'] = addr + aligned_size
        
        # 记录分配信息
        allocation = {
            'mem_type': mem_type,
            'addr': addr,
            'size': aligned_size,
            'byte_addr': addr,
            'word_addr': addr >> 2,  # 转换为字地址（32位）
            'dtype': tensor_type,
            'bank': self._assign_bank(tensor_info, mem_type),
            'tensor_name': tensor_name,
            'tensor_shape': tensor_info.get('shape', []),
            'data_type': tensor_info.get('data_type', 'unknown')
        }
        
        self.allocations[tensor_name] = allocation
        logger.info(f"分配 {tensor_name} ({tensor_info.get('data_type', 'unknown')}): "
                   f"{mem_type} 0x{addr:04X} - 0x{addr + aligned_size:04X} "
                   f"(大小: {aligned_size} 字节)")
        
        return allocation
    
    def _infer_mem_type(self, tensor_info: Dict) -> str:
        """根据张量类型推断内存类型"""
        data_type = tensor_info.get('data_type', 'unknown')
        
        if data_type == 'weight' or tensor_info.get('is_parameter', False):
            return 'mem1'
        elif data_type == 'input':
            return 'mem0'
        elif data_type == 'output':
            return 'mem0'
        elif data_type == 'intermediate':
            return 'mem0'
        elif tensor_info.get('is_io', False):
            return 'mem2'
        else:
            return 'mem0'
    
    def _get_alignment(self, mem_type: str, tensor_info: Dict) -> int:
        """获取对齐要求"""
        data_type = tensor_info.get('data_type', 'unknown')
        
        if mem_type == 'mem1':  # 权重内存
            return 32
        elif mem_type == 'mem0':  # 数据内存
            if data_type == 'weight':
                return 32
            else:
                return 16
        else:  # 其他内存
            return 4
    
    def _align_size(self, size: int, alignment: int) -> int:
        """对齐内存大小"""
        if alignment <= 1:
            return size
        return ((size + alignment - 1) // alignment) * alignment
    
    def _align_address(self, addr: int, alignment: int) -> int:
        """对齐地址"""
        if alignment <= 1:
            return addr
        return ((addr + alignment - 1) // alignment) * alignment
    
    def _assign_bank(self, tensor_info: Dict, mem_type: str) -> int:
        """分配内存bank"""
        # 简单策略：根据张量名称哈希分配
        tensor_name = tensor_info['name']
        tensor_size = tensor_info.get('size_in_bytes', 0)
        
        if tensor_size > 4096:  # 大张量独占bank 0
            return 0
        else:
            # 使用名称哈希分配
            return hash(tensor_name) % self.hw_config.core_mem_banks
    
    def get_allocation_report(self) -> Dict:
        """生成内存分配报告"""
        report = {
            'mem_usage': {},
            'allocations': self.allocations,
            'summary': {
                'total_allocated': 0,
                'total_tensors': len(self.allocations)
            }
        }
        
        total_allocated = 0
        
        for mem_type, block in self.mem_blocks.items():
            used = block['current'] - block['base']
            free = block['size'] - used
            usage_percent = used / block['size'] * 100 if block['size'] > 0 else 0
            
            report['mem_usage'][mem_type] = {
                'total': block['size'],
                'used': used,
                'free': free,
                'usage_percent': usage_percent,
                'base_address': f"0x{block['base']:04X}",
                'next_free': f"0x{block['current']:04X}"
            }
            
            total_allocated += used
        
        report['summary']['total_allocated'] = total_allocated
        
        # 按类型统计
        type_stats = {}
        for alloc in self.allocations.values():
            dtype = alloc['data_type']
            type_stats[dtype] = type_stats.get(dtype, 0) + alloc['size']
        
        report['summary']['type_stats'] = type_stats
        
        return report
    
    def get_address_map(self) -> Dict[str, Dict]:
        """获取地址映射表"""
        address_map = {}
        for tensor_name, alloc in self.allocations.items():
            address_map[tensor_name] = {
                'byte_addr': f"0x{alloc['byte_addr']:04X}",
                'word_addr': f"0x{alloc['word_addr']:04X}",
                'mem_type': alloc['mem_type'],
                'size': alloc['size']
            }
        return address_map


class HardwareMapperV2:
    """硬件映射器V2 - 更智能的映射策略"""
    
    def __init__(self, hw_config: HardwareConfig):
        self.hw_config = hw_config
        self.memory_allocator = AutoMemoryAllocator(hw_config)
        
        # 算子到硬件原语的映射表
        self.op_to_primitive = {
            'Conv': self._map_convolution,
            'Conv2D': self._map_convolution,
            'MaxPool': self._map_maxpool,
            'AveragePool': self._map_avgpool,
            'Relu': self._map_relu,
            'Sigmoid': self._map_sigmoid,
            'Tanh': self._map_tanh,
            'Add': self._map_add,
            'Mul': self._map_mul,
            'Clip': self._map_clip,
            'Reshape': self._map_reshape,
            'Flatten': self._map_flatten,
            'Gemm': self._map_gemm,
            'MatMul': self._map_gemm,
            'BatchNormalization': self._map_batchnorm,
            'Concat': self._map_concat,
            'GlobalAveragePool': self._map_global_avgpool
        }
        
        # 支持的量化配置
        self.quantization_scheme = {
            'weight_bits': 8,
            'activation_bits': 8,
            'bias_bits': 32,
            'scale_bits': 16
        }
    
    def map_model(self, model_analysis: Dict, quantization: bool = False) -> Dict:
        """映射整个模型到硬件"""
        logger.info("开始硬件映射")
        
        computation_graph = model_analysis['computation_graph']
        statistics = model_analysis['statistics']
        
        # 1. 内存分配
        memory_map = self._allocate_memory(computation_graph['tensors'])
        
        # 2. 为每个算子生成硬件原语配置
        primitives_config = self._generate_primitives(
            computation_graph['nodes'], 
            memory_map, 
            computation_graph['tensors'],
            quantization
        )
        
        # 3. 生成阶段调度
        phase_schedule = self._schedule_phases(primitives_config)
        
        # 4. 分配计算核心
        core_assignment = self._assign_cores(
            computation_graph['nodes'],
            primitives_config,
            statistics
        )
        
        # 5. 生成路由配置
        router_config = self._generate_router_config(core_assignment)
        
        # 6. 生成最终配置
        final_config = {
            'model_info': model_analysis['model_info'],
            'hardware_config': self._serializable_hw_config(),
            'memory_map': memory_map,
            'primitives': primitives_config,
            'phase_schedule': phase_schedule,
            'core_assignment': self._serializable_core_assignment(core_assignment),
            'router_config': router_config,
            'statistics': {
                'model': statistics,
                'hardware': self._calculate_hardware_stats(primitives_config, phase_schedule)
            }
        }
        
        logger.info("硬件映射完成")
        return final_config
    
    def _serializable_hw_config(self) -> Dict:
        """转换为可序列化的硬件配置"""
        config_dict = {
            'mem0_size': self.hw_config.mem0_size,
            'mem1_size': self.hw_config.mem1_size,
            'mem2_size': self.hw_config.mem2_size,
            'mem3_size': self.hw_config.mem3_size,
            'core_grid': list(self.hw_config.core_grid),
            'core_mem_banks': self.hw_config.core_mem_banks,
            'mac_units_per_core': self.hw_config.mac_units_per_core,
            'vector_width': self.hw_config.vector_width,
            'router_channels': self.hw_config.router_channels,
            'max_packet_size': self.hw_config.max_packet_size,
            'total_cores': self.hw_config.total_cores
        }
        return config_dict
    
    def _serializable_core_assignment(self, core_assignment: Dict) -> Dict:
        """转换为可序列化的核心分配"""
        serializable = {}
        for core_id, core_info in core_assignment.items():
            # 将元组键转换为字符串
            core_id_str = str(core_id)
            serializable[core_id_str] = core_info
        return serializable
    
    def _allocate_memory(self, tensors: Dict[str, Dict]) -> Dict[str, Dict]:
        """为所有张量分配内存"""
        memory_map = {}
        
        # 首先分配权重和偏置
        for tensor_name, tensor_info in tensors.items():
            if tensor_info.get('is_parameter', False):
                try:
                    alloc_info = {
                        'name': tensor_name,
                        'size_in_bytes': tensor_info['size_in_bytes'],
                        'dtype': tensor_info['dtype'],
                        'is_parameter': True,
                        'data_type': tensor_info.get('data_type', 'weight'),
                        'shape': tensor_info['shape']
                    }
                    allocation = self.memory_allocator.allocate(alloc_info, 'mem1')
                    memory_map[tensor_name] = allocation
                except Exception as e:
                    logger.warning(f"分配权重 {tensor_name} 失败: {e}")
        
        # 然后分配输入输出
        for tensor_name, tensor_info in tensors.items():
            if not tensor_info.get('is_parameter', False):
                try:
                    alloc_info = {
                        'name': tensor_name,
                        'size_in_bytes': tensor_info['size_in_bytes'],
                        'dtype': tensor_info['dtype'],
                        'is_parameter': False,
                        'data_type': tensor_info.get('data_type', 'unknown'),
                        'shape': tensor_info['shape'],
                        'is_io': tensor_info.get('data_type') in ['input', 'output']
                    }
                    
                    # 输入输出可能分配到mem0或mem2
                    mem_type = 'auto'
                    if tensor_info.get('data_type') == 'input':
                        mem_type = 'mem0'
                    elif tensor_info.get('data_type') == 'output':
                        mem_type = 'mem0'
                    
                    allocation = self.memory_allocator.allocate(alloc_info, mem_type)
                    memory_map[tensor_name] = allocation
                except Exception as e:
                    logger.warning(f"分配张量 {tensor_name} 失败: {e}")
        
        return memory_map
    
    def _generate_primitives(self, operators: List[Dict], 
                            memory_map: Dict, 
                            tensors: Dict,
                            quantization: bool = False) -> List[Dict]:
        """为每个算子生成硬件原语配置"""
        primitives = []
        
        for op_idx, op in enumerate(operators):
            op_type = op['type']
            mapper_func = self.op_to_primitive.get(op_type)
            
            if mapper_func:
                try:
                    # 获取输入输出张量的内存地址
                    input_addrs = []
                    for inp in op['inputs']:
                        if inp in memory_map:
                            input_addrs.append(memory_map[inp]['word_addr'])
                        else:
                            input_addrs.append(0)  # 默认地址
                    
                    output_addrs = []
                    for out in op['outputs']:
                        if out in memory_map:
                            output_addrs.append(memory_map[out]['word_addr'])
                        else:
                            output_addrs.append(0)
                    
                    # 获取张量形状
                    input_shapes = []
                    for inp in op['inputs']:
                        if inp in tensors:
                            input_shapes.append(tensors[inp]['shape'])
                        else:
                            input_shapes.append([])
                    
                    output_shapes = []
                    for out in op['outputs']:
                        if out in tensors:
                            output_shapes.append(tensors[out]['shape'])
                        else:
                            output_shapes.append([])
                    
                    # 调用映射函数
                    primitive_config = mapper_func(
                        op, 
                        input_addrs, 
                        output_addrs, 
                        input_shapes, 
                        output_shapes, 
                        op['attributes'],
                        quantization
                    )
                    
                    primitive_config['op_id'] = op_idx
                    primitive_config['op_type'] = op_type
                    primitive_config['op_name'] = op.get('name', f'op_{op_idx}')
                    
                    primitives.append(primitive_config)
                    
                except Exception as e:
                    logger.error(f"映射算子 {op_type} (ID: {op_idx}) 失败: {e}")
                    # 添加一个空的primitive作为占位
                    primitives.append({
                        'op_id': op_idx,
                        'op_type': op_type,
                        'op_name': op.get('name', f'op_{op_idx}'),
                        'prim_type': 'unknown',
                        'params': {},
                        'error': str(e)
                    })
            else:
                logger.warning(f"未找到算子 {op_type} 的硬件映射，跳过")
        
        return primitives
    
    def _map_convolution(self, op: Dict, input_addrs: List[int], 
                        output_addrs: List[int], input_shapes: List[List[int]], 
                        output_shapes: List[List[int]], attrs: Dict,
                        quantization: bool = False) -> Dict:
        """映射卷积算子"""
        # 提取卷积参数
        strides = attrs.get('strides', [1, 1])
        pads = attrs.get('pads', [0, 0, 0, 0])
        dilations = attrs.get('dilations', [1, 1])
        group = attrs.get('group', 1)
        
        # 输入输出形状 [N, C, H, W]
        in_shape = input_shapes[0] if input_shapes else [1, 1, 1, 1]
        weight_shape = input_shapes[1] if len(input_shapes) > 1 else [1, 1, 1, 1]
        out_shape = output_shapes[0] if output_shapes else [1, 1, 1, 1]
        
        # 确定使用哪个卷积原语
        kernel_h = weight_shape[2] if len(weight_shape) > 2 else 1
        kernel_w = weight_shape[3] if len(weight_shape) > 3 else 1
        
        if kernel_h == 1 and kernel_w == 1:
            # 1x1卷积
            primitive_type = 'p41'
        else:
            # 普通卷积
            primitive_type = 'p81'
        
        # 偏置地址（如果有）
        bias_addr = 0
        if len(input_addrs) > 2:
            bias_addr = input_addrs[2]
        
        primitive_config = {
            'prim_type': primitive_type,
            'params': {
                'px': in_shape[2] if len(in_shape) > 2 else 1,
                'py': in_shape[3] if len(in_shape) > 3 else 1,
                'cin': in_shape[1] if len(in_shape) > 1 else 1,
                'cout': weight_shape[0] if weight_shape else 1,
                'kx': kernel_w,
                'ky': kernel_h,
                'sx': strides[1] if len(strides) > 1 else 1,
                'sy': strides[0] if len(strides) > 0 else 1,
                'addr_ina': input_addrs[0] if input_addrs else 0,
                'addr_inb': input_addrs[1] if len(input_addrs) > 1 else 0,
                'addr_bias': bias_addr,
                'addr_out': output_addrs[0] if output_addrs else 0,
                'pad_top': pads[0] if len(pads) > 0 else 0,
                'pad_down': pads[2] if len(pads) > 2 else 0,
                'pad_left': pads[1] if len(pads) > 1 else 0,
                'pad_right': pads[3] if len(pads) > 3 else 0,
                'dilations': dilations,
                'group': group,
                'ina_type': 0 if quantization else 1,  # 量化时使用int8
                'inb_type': 0 if quantization else 1,
                'load_bias': 1 if bias_addr > 0 else 0
            }
        }
        
        return primitive_config
    
    def _map_maxpool(self, op: Dict, input_addrs: List[int], 
                    output_addrs: List[int], input_shapes: List[List[int]], 
                    output_shapes: List[List[int]], attrs: Dict,
                    quantization: bool = False) -> Dict:
        """映射最大池化算子"""
        kernel_shape = attrs.get('kernel_shape', [2, 2])
        strides = attrs.get('strides', kernel_shape)
        
        in_shape = input_shapes[0] if input_shapes else [1, 1, 1, 1]
        out_shape = output_shapes[0] if output_shapes else [1, 1, 1, 1]
        
        primitive_config = {
            'prim_type': 'pX5',
            'params': {
                'mode': 'max',
                'addr_in': input_addrs[0] if input_addrs else 0,
                'addr_out': output_addrs[0] if output_addrs else 0,
                'cin': in_shape[1] if len(in_shape) > 1 else 1,
                'cout': out_shape[1] if len(out_shape) > 1 else 1,
                'px': in_shape[2] if len(in_shape) > 2 else 1,
                'py': in_shape[3] if len(in_shape) > 3 else 1,
                'kx': kernel_shape[0] if len(kernel_shape) > 0 else 2,
                'ky': kernel_shape[1] if len(kernel_shape) > 1 else 2,
                'sx': strides[0] if len(strides) > 0 else 1,
                'sy': strides[1] if len(strides) > 1 else 1,
                'cmp_c': 0x0,
                'type_in': 0 if quantization else 1,
                'type_out': 0 if quantization else 1,
                'in_cut_start': 0,
                'row_ck_on': 1,
                'in_row_max': 2
            }
        }
        
        return primitive_config
    
    def _map_gemm(self, op: Dict, input_addrs: List[int], 
                 output_addrs: List[int], input_shapes: List[List[int]], 
                 output_shapes: List[List[int]], attrs: Dict,
                 quantization: bool = False) -> Dict:
        """映射全连接层"""
        # 全连接层作为1x1卷积处理
        in_shape = input_shapes[0] if input_shapes else [1, 1]
        weight_shape = input_shapes[1] if len(input_shapes) > 1 else [1, 1]
        
        # 重塑为4D形状 [N, 1, Cin, Cout]
        batch = in_shape[0] if len(in_shape) > 0 else 1
        cin = in_shape[1] if len(in_shape) > 1 else 1
        cout = weight_shape[0] if len(weight_shape) > 0 else 1
        
        # 偏置地址
        bias_addr = input_addrs[2] if len(input_addrs) > 2 else 0
        
        primitive_config = {
            'prim_type': 'p41',
            'params': {
                'px': 1,
                'py': 1,
                'cin': cin,
                'cout': cout,
                'kx': 1,
                'ky': 1,
                'sx': 1,
                'sy': 1,
                'addr_ina': input_addrs[0] if input_addrs else 0,
                'addr_inb': input_addrs[1] if len(input_addrs) > 1 else 0,
                'addr_bias': bias_addr,
                'addr_out': output_addrs[0] if output_addrs else 0,
                'pad_top': 0,
                'pad_down': 0,
                'pad_left': 0,
                'pad_right': 0,
                'ina_type': 0 if quantization else 1,
                'inb_type': 0 if quantization else 1,
                'load_bias': 1 if bias_addr > 0 else 0
            }
        }
        
        return primitive_config
    
    def _map_relu(self, op: Dict, input_addrs: List[int], 
                 output_addrs: List[int], input_shapes: List[List[int]], 
                 output_shapes: List[List[int]], attrs: Dict,
                 quantization: bool = False) -> Dict:
        """映射ReLU激活"""
        in_shape = input_shapes[0] if input_shapes else [1, 1, 1, 1]
        
        primitive_config = {
            'prim_type': 'pX5',
            'params': {
                'mode': 'relu',
                'addr_in': input_addrs[0] if input_addrs else 0,
                'addr_out': output_addrs[0] if output_addrs else 0,
                'cin': in_shape[1] if len(in_shape) > 1 else 1,
                'cout': in_shape[1] if len(in_shape) > 1 else 1,
                'px': in_shape[2] if len(in_shape) > 2 else 1,
                'py': in_shape[3] if len(in_shape) > 3 else 1,
                'kx': 1,
                'ky': 1,
                'sx': 1,
                'sy': 1,
                'cmp_c': 0x0,  # 对于ReLU，阈值是0
                'type_in': 0 if quantization else 1,
                'type_out': 0 if quantization else 1,
                'in_cut_start': 0,
                'row_ck_on': 1,
                'in_row_max': 1
            }
        }
        
        return primitive_config
    
    def _map_add(self, op: Dict, input_addrs: List[int], 
                output_addrs: List[int], input_shapes: List[List[int]], 
                output_shapes: List[List[int]], attrs: Dict,
                quantization: bool = False) -> Dict:
        """映射加法操作"""
        return self._map_elementwise(op, input_addrs, output_addrs, 
                                    input_shapes, output_shapes, 'add', quantization)
    
    def _map_mul(self, op: Dict, input_addrs: List[int], 
                output_addrs: List[int], input_shapes: List[List[int]], 
                output_shapes: List[List[int]], attrs: Dict,
                quantization: bool = False) -> Dict:
        """映射乘法操作"""
        return self._map_elementwise(op, input_addrs, output_addrs, 
                                    input_shapes, output_shapes, 'mul', quantization)
    
    def _map_clip(self, op: Dict, input_addrs: List[int], 
                 output_addrs: List[int], input_shapes: List[List[int]], 
                 output_shapes: List[List[int]], attrs: Dict,
                 quantization: bool = False) -> Dict:
        """映射Clip操作"""
        min_val = attrs.get('min', -3.4028235e+38)  # float32最小值
        max_val = attrs.get('max', 3.4028235e+38)   # float32最大值
        
        # 转换为定点数表示
        if quantization:
            min_int = int(min_val * 128) if min_val > -128 else -128
            max_int = int(max_val * 128) if max_val < 127 else 127
        else:
            min_int = int(min_val)
            max_int = int(max_val)
        
        in_shape = input_shapes[0] if input_shapes else [1, 1, 1, 1]
        
        primitive_config = {
            'prim_type': 'pX5',
            'params': {
                'mode': 'clip',
                'addr_in': input_addrs[0] if input_addrs else 0,
                'addr_out': output_addrs[0] if output_addrs else 0,
                'cin': in_shape[1] if len(in_shape) > 1 else 1,
                'cout': in_shape[1] if len(in_shape) > 1 else 1,
                'px': in_shape[2] if len(in_shape) > 2 else 1,
                'py': in_shape[3] if len(in_shape) > 3 else 1,
                'kx': 1,
                'ky': 1,
                'sx': 1,
                'sy': 1,
                'cmp_c': (max_int & 0xFF) << 24 | (min_int & 0xFFFFFF),
                'type_in': 0 if quantization else 1,
                'type_out': 0 if quantization else 1,
                'in_cut_start': 0,
                'row_ck_on': 1,
                'in_row_max': 1
            }
        }
        
        return primitive_config
    
    def _map_reshape(self, op: Dict, input_addrs: List[int], 
                    output_addrs: List[int], input_shapes: List[List[int]], 
                    output_shapes: List[List[int]], attrs: Dict,
                    quantization: bool = False) -> Dict:
        """映射重塑操作"""
        # 重塑操作通常使用数据搬运原语
        in_shape = input_shapes[0] if input_shapes else []
        out_shape = output_shapes[0] if output_shapes else []
        
        # 计算数据大小
        total_elements = 1
        for dim in in_shape:
            total_elements *= dim
        
        primitive_config = {
            'prim_type': 'p06',
            'params': {
                'addr_in': input_addrs[0] if input_addrs else 0,
                'addr_out': output_addrs[0] if output_addrs else 0,
                'addr_ciso': 0,
                'length_in': in_shape[-1] if in_shape else 1,
                'num_in': total_elements // (in_shape[-1] if in_shape else 1),
                'length_ciso': 0,
                'num_ciso': 0,
                'length_out': out_shape[-1] if out_shape else 1,
                'num_out': total_elements // (out_shape[-1] if out_shape else 1),
                'type_in': 0 if quantization else 1,
                'type_out': 0 if quantization else 1,
                'data_in': None
            }
        }
        
        return primitive_config
    
    def _map_elementwise(self, op: Dict, input_addrs: List[int], 
                        output_addrs: List[int], input_shapes: List[List[int]], 
                        output_shapes: List[List[int]], mode: str,
                        quantization: bool = False) -> Dict:
        """映射逐元素操作"""
        in_shape = input_shapes[0] if input_shapes else [1, 1, 1, 1]
        
        primitive_config = {
            'prim_type': 'pX5',
            'params': {
                'mode': mode,
                'addr_in': input_addrs[0] if input_addrs else 0,
                'addr_out': output_addrs[0] if output_addrs else 0,
                'cin': in_shape[1] if len(in_shape) > 1 else 1,
                'cout': in_shape[1] if len(in_shape) > 1 else 1,
                'px': in_shape[2] if len(in_shape) > 2 else 1,
                'py': in_shape[3] if len(in_shape) > 3 else 1,
                'kx': 1,
                'ky': 1,
                'sx': 1,
                'sy': 1,
                'cmp_c': 0x0,
                'type_in': 0 if quantization else 1,
                'type_out': 0 if quantization else 1,
                'in_cut_start': 0,
                'row_ck_on': 1,
                'in_row_max': 1
            }
        }
        
        return primitive_config
    
    # 其他映射函数的简化实现
    def _map_avgpool(self, *args, **kwargs):
        """映射平均池化"""
        config = self._map_maxpool(*args, **kwargs)
        config['params']['mode'] = 'avg'
        return config
    
    def _map_sigmoid(self, *args, **kwargs):
        """映射Sigmoid激活"""
        config = self._map_elementwise(*args, **kwargs)
        config['params']['mode'] = 'sigmoid'
        return config
    
    def _map_tanh(self, *args, **kwargs):
        """映射Tanh激活"""
        config = self._map_elementwise(*args, **kwargs)
        config['params']['mode'] = 'tanh'
        return config
    
    def _map_flatten(self, *args, **kwargs):
        """映射Flatten操作"""
        return self._map_reshape(*args, **kwargs)
    
    def _map_batchnorm(self, *args, **kwargs):
        """映射BatchNorm操作"""
        # 简化处理：作为逐元素操作
        return self._map_elementwise(*args, **kwargs)
    
    def _map_concat(self, *args, **kwargs):
        """映射Concat操作"""
        # 简化处理：使用数据搬运
        return self._map_reshape(*args, **kwargs)
    
    def _map_global_avgpool(self, *args, **kwargs):
        """映射全局平均池化"""
        config = self._map_avgpool(*args, **kwargs)
        # 设置kernel为输入尺寸
        if 'params' in config and 'input_shapes' in kwargs:
            input_shapes = kwargs['input_shapes']
            if input_shapes and input_shapes[0]:
                in_shape = input_shapes[0]
                if len(in_shape) >= 3:
                    config['params']['kx'] = in_shape[2]
                    config['params']['ky'] = in_shape[2] if len(in_shape) < 4 else in_shape[3]
        return config
    
    def _schedule_phases(self, primitives: List[Dict]) -> List[Dict]:
        """生成阶段调度"""
        phases = []
        
        # 简化的调度：每个算子一个阶段
        for i, prim in enumerate(primitives):
            # 估计时钟周期
            cycles = self._estimate_clock_cycles(prim)
            
            phase = {
                'phase_id': i,
                'primitives': [prim],
                'clock_cycles': cycles,
                'dependencies': self._find_dependencies(i, primitives),
                'estimated_latency_ms': cycles / 200000 * 1000  # 假设200MHz时钟
            }
            phases.append(phase)
        
        return phases
    
    def _estimate_clock_cycles(self, primitive: Dict) -> int:
        """估计时钟周期"""
        prim_type = primitive.get('prim_type', 'unknown')
        params = primitive.get('params', {})
        
        if prim_type in ['p81', 'p41']:
            # 卷积：O(H*W*Cin*Cout*Kx*Ky)
            h = params.get('px', 1)
            w = params.get('py', 1)
            cin = params.get('cin', 1)
            cout = params.get('cout', 1)
            kx = params.get('kx', 1)
            ky = params.get('ky', 1)
            
            # 考虑硬件并行度
            parallel_ops = min(cin, self.hw_config.mac_units_per_core)
            cycles = (h * w * cout * kx * ky * cin) // max(parallel_ops, 1)
            
            return max(cycles, 100)
        
        elif prim_type == 'pX5':
            # 池化/激活：O(H*W*C)
            h = params.get('px', 1)
            w = params.get('py', 1)
            c = params.get('cin', 1)
            
            cycles = h * w * c // self.hw_config.vector_width
            
            return max(cycles, 10)
        
        elif prim_type == 'p06':
            # 数据搬运：O(size)
            length = params.get('length_in', 1)
            num = params.get('num_in', 1)
            
            cycles = length * num // 8  # 假设8字节/周期
            
            return max(cycles, 10)
        
        else:
            return 100  # 默认值
    
    def _find_dependencies(self, op_idx: int, all_primitives: List[Dict]) -> List[int]:
        """查找依赖关系"""
        dependencies = []
        
        # 简化的依赖分析：顺序执行
        for i in range(op_idx):
            dependencies.append(i)
        
        return dependencies
    
    def _assign_cores(self, operators: List[Dict], primitives: List[Dict], 
                     statistics: Dict) -> Dict:
        """分配计算核心"""
        core_assignments = {}
        total_cores = self.hw_config.total_cores
        
        # 简单的核心分配：轮询分配算子
        ops_per_core = max(len(operators) // total_cores, 1)
        
        for core_idx in range(total_cores):
            core_x = core_idx % self.hw_config.core_grid[0]
            core_y = core_idx // self.hw_config.core_grid[0]
            core_id = ((0, 0), (core_x, core_y))  # ((chip_x, chip_y), (core_x, core_y))
            
            start_op = core_idx * ops_per_core
            end_op = min((core_idx + 1) * ops_per_core, len(operators))
            
            assigned_ops = operators[start_op:end_op]
            assigned_prims = primitives[start_op:end_op]
            
            # 计算该核心的负载
            total_cycles = 0
            for prim in assigned_prims:
                total_cycles += self._estimate_clock_cycles(prim)
            
            core_assignments[core_id] = {
                'operator_ids': list(range(start_op, end_op)),
                'operator_count': len(assigned_ops),
                'primitives_count': len(assigned_prims),
                'estimated_cycles': total_cycles,
                'operators': [op['type'] for op in assigned_ops],
                'memory_requirements': self._calculate_memory_requirements(assigned_ops)
            }
        
        return core_assignments
    
    def _calculate_memory_requirements(self, operators: List[Dict]) -> Dict:
        """计算内存需求"""
        # 简化实现
        return {
            'weight_memory': 4096,  # 4KB
            'activation_memory': 8192,  # 8KB
            'io_memory': 1024  # 1KB
        }
    
    def _generate_router_config(self, core_assignment: Dict) -> Dict:
        """生成路由配置"""
        router_config = {
            'routers': {},
            'routes': [],
            'bandwidth_requirements': {}
        }
        
        # 为每个核心生成路由器配置
        for core_id, core_info in core_assignment.items():
            # 将元组键转换为字符串
            router_id = str(core_id)
            
            router_config['routers'][router_id] = {
                'core_position': list(core_id[1]),  # 转换为列表
                'channels': self.hw_config.router_channels,
                'max_packet_size': self.hw_config.max_packet_size,
                'neighbors': self._get_neighbors(core_id[1])
            }
        
        # 生成默认路由
        router_config['routes'] = self._generate_default_routes(core_assignment)
        
        return router_config
    
    def _get_neighbors(self, core_pos: Tuple[int, int]) -> List[Dict]:
        """获取邻居核心"""
        neighbors = []
        x, y = core_pos
        
        # 上邻居
        if y > 0:
            neighbors.append({'direction': 'north', 'core': [x, y-1]})
        # 下邻居
        if y < self.hw_config.core_grid[1] - 1:
            neighbors.append({'direction': 'south', 'core': [x, y+1]})
        # 左邻居
        if x > 0:
            neighbors.append({'direction': 'west', 'core': [x-1, y]})
        # 右邻居
        if x < self.hw_config.core_grid[0] - 1:
            neighbors.append({'direction': 'east', 'core': [x+1, y]})
        
        return neighbors
    
    def _generate_default_routes(self, core_assignment: Dict) -> List[Dict]:
        """生成默认路由"""
        routes = []
        
        # 简单的路由：所有核心都可以互相通信
        for src_core in core_assignment.keys():
            for dst_core in core_assignment.keys():
                if src_core != dst_core:
                    routes.append({
                        'source': list(src_core[1]),
                        'destination': list(dst_core[1]),
                        'path': self._calculate_path(src_core[1], dst_core[1]),
                        'priority': 1
                    })
        
        return routes
    
    def _calculate_path(self, src: Tuple[int, int], dst: Tuple[int, int]) -> List[List[int]]:
        """计算路径（曼哈顿距离）"""
        path = []
        x1, y1 = src
        x2, y2 = dst
        
        # 先水平移动，再垂直移动
        if x1 < x2:
            for x in range(x1 + 1, x2 + 1):
                path.append([x, y1])
        elif x1 > x2:
            for x in range(x1 - 1, x2 - 1, -1):
                path.append([x, y1])
        
        if y1 < y2:
            for y in range(y1 + 1, y2 + 1):
                path.append([x2, y])
        elif y1 > y2:
            for y in range(y1 - 1, y2 - 1, -1):
                path.append([x2, y])
        
        return path
    
    def _calculate_hardware_stats(self, primitives: List[Dict], phases: List[Dict]) -> Dict:
        """计算硬件统计信息"""
        total_cycles = sum(phase.get('clock_cycles', 0) for phase in phases)
        
        # 计算MAC操作数
        total_macs = 0
        conv_primitives = [p for p in primitives if p.get('prim_type') in ['p81', 'p41']]
        
        for prim in conv_primitives:
            params = prim.get('params', {})
            h = params.get('px', 1)
            w = params.get('py', 1)
            cin = params.get('cin', 1)
            cout = params.get('cout', 1)
            kx = params.get('kx', 1)
            ky = params.get('ky', 1)
            total_macs += h * w * cin * cout * kx * ky
        
        # 计算性能指标
        total_mac_units = self.hw_config.total_cores * self.hw_config.mac_units_per_core
        
        return {
            'total_cycles': total_cycles,
            'total_macs': total_macs,
            'macs_per_cycle': total_macs / max(total_cycles, 1),
            'peak_performance_gops': total_mac_units * 0.000001,  # 假设1GHz时钟
            'utilization': min(total_macs / (total_cycles * total_mac_units), 1.0) if total_cycles > 0 else 0,
            'estimated_latency_ms': total_cycles / 200000 * 1000  # 假设200MHz时钟
        }


class ConfigGeneratorV2:
    """配置生成器V2"""
    
    def __init__(self, hardware_mapper: HardwareMapperV2):
        self.hardware_mapper = hardware_mapper
        self.hw_config = hardware_mapper.hw_config
        
    def generate_all_configs(self, mapped_model: Dict) -> Dict:
        """生成所有配置"""
        logger.info("生成硬件配置...")
        
        configs = {
            'g0_config': self.generate_g0_config(mapped_model),
            'g1_config': self.generate_g1_config(mapped_model),
            'g2_config': self.generate_g2_config(mapped_model),
            'global_config': self.generate_global_config(mapped_model)
        }
        
        return configs
    
    def generate_g0_config(self, mapped_model: Dict) -> Dict:
        """生成Group0（FPGA）配置"""
        # Group0通常处理数据输入输出
        config = {
            'type': 'g0',
            'description': 'FPGA数据输入输出组',
            'chip': [0, 0],  # 使用列表代替元组
            'cores': {
                '1_0': {  # 使用字符串键代替元组
                    'role': 'fpga_io',
                    'primitives': []
                }
            },
            'phases': []
        }
        
        # 查找数据搬运阶段
        data_transfer_phases = []
        for phase in mapped_model.get('phase_schedule', []):
            for prim in phase.get('primitives', []):
                if prim.get('prim_type') == 'p06':
                    data_transfer_phases.append(phase)
                    break
        
        # 为每个数据搬运阶段生成配置
        for i, phase in enumerate(data_transfer_phases):
            phase_config = {
                'phase_id': i,
                'clock': 200000,
                'mode': 1,
                'cores': {
                    '1_0': {  # 使用字符串键
                        'prims': self._extract_prims_for_core(phase, (1, 0))
                    }
                }
            }
            config['phases'].append(phase_config)
        
        return config
    
    def generate_g1_config(self, mapped_model: Dict) -> Dict:
        """生成Group1（1x1核心）配置"""
        config = {
            'type': 'g1',
            'description': '计算核心组 (1x1)',
            'chip': [0, 0],  # 使用列表代替元组
            'size_x': 1,
            'size_y': 1,
            'cores': {
                '0_0': {  # 使用字符串键
                    'role': 'computation',
                    'operator_count': len(mapped_model.get('primitives', [])),
                    'primitives': mapped_model.get('primitives', [])
                }
            },
            'phases': []
        }
        
        # 为每个阶段生成配置
        for i, phase in enumerate(mapped_model.get('phase_schedule', [])):
            phase_config = {
                'phase_id': i,
                'clock': phase.get('clock_cycles', 200000),
                'mode': 1,
                'cores': {
                    '0_0': {  # 使用字符串键
                        'prims': self._extract_prims_for_core(phase, (0, 0))
                    }
                }
            }
            config['phases'].append(phase_config)
        
        return config
    
    def generate_g2_config(self, mapped_model: Dict) -> Dict:
        """生成Group2（8x1核心）配置"""
        config = {
            'type': 'g2',
            'description': '路由核心组 (8x1)',
            'chip': [0, 0],  # 使用列表代替元组
            'size_x': 8,
            'size_y': 1,
            'cores': {},
            'phases': []
        }
        
        # 为每个路由核心生成配置
        for core_x in range(8):
            core_id = f'{core_x}_0'  # 使用字符串键
            config['cores'][core_id] = {
                'role': 'routing',
                'neighbors': self._get_core_neighbors((core_x, 0))
            }
        
        # 生成路由阶段
        router_phase = {
            'phase_id': 0,
            'clock': 100000,
            'mode': 1,
            'cores': {}
        }
        
        # 为每个核心添加路由器配置
        for core_id_str in config['cores'].keys():
            router_phase['cores'][core_id_str] = {
                'prims': [self._create_router_prim(core_id_str)]
            }
        
        config['phases'].append(router_phase)
        
        return config
    
    def generate_global_config(self, mapped_model: Dict) -> Dict:
        """生成全局配置"""
        memory_report = self.hardware_mapper.memory_allocator.get_allocation_report()
        
        config = {
            'sim_clock': 100000,
            'step_clock': {
                '0_0_0': (100000 - 1, 100000)  # 使用字符串键
            },
            'memory_map': self.hardware_mapper.memory_allocator.get_address_map(),
            'memory_usage': memory_report['mem_usage'],
            'hardware_stats': mapped_model.get('statistics', {}).get('hardware', {}),
            'model_stats': mapped_model.get('statistics', {}).get('model', {}),
            'phase_summary': {
                'total_phases': len(mapped_model.get('phase_schedule', [])),
                'total_cycles': sum(p.get('clock_cycles', 0) 
                                  for p in mapped_model.get('phase_schedule', [])),
                'estimated_latency_ms': sum(p.get('estimated_latency_ms', 0) 
                                          for p in mapped_model.get('phase_schedule', []))
            }
        }
        
        return config
    
    def _extract_prims_for_core(self, phase: Dict, core_id: Tuple[int, int]) -> List[Dict]:
        """为核心提取原语配置"""
        prims = []
        
        for primitive in phase.get('primitives', []):
            prim_type = primitive.get('prim_type', 'unknown')
            
            if prim_type == 'p06':
                prim = {
                    'axon': None,
                    'soma1': None,
                    'router': None,
                    'soma2': self._create_p06_config(primitive.get('params', {}))
                }
            elif prim_type == 'p09':
                prim = {
                    'axon': None,
                    'soma1': None,
                    'router': self._create_p09_config(primitive.get('params', {}), core_id),
                    'soma2': None
                }
            elif prim_type in ['p81', 'p41']:
                prim = {
                    'axon': self._create_conv_config(prim_type, primitive.get('params', {})),
                    'soma1': None,
                    'router': None,
                    'soma2': None
                }
            elif prim_type == 'pX5':
                prim = {
                    'axon': None,
                    'soma1': self._create_pX5_config(primitive.get('params', {})),
                    'router': None,
                    'soma2': None
                }
            else:
                prim = {'axon': None, 'soma1': None, 'router': None, 'soma2': None}
            
            prims.append(prim)
        
        return prims
    
    def _create_p06_config(self, params: Dict) -> Dict:
        """创建p06配置"""
        return {
            'addr_in': params.get('addr_in', 0),
            'addr_out': params.get('addr_out', 0),
            'addr_ciso': params.get('addr_ciso', 0),
            'length_in': params.get('length_in', 32),
            'num_in': params.get('num_in', 1),
            'length_ciso': params.get('length_ciso', 1),
            'num_ciso': params.get('num_ciso', 1),
            'length_out': params.get('length_out', 32),
            'num_out': params.get('num_out', 1),
            'type_in': params.get('type_in', 1),
            'type_out': params.get('type_out', 1),
            'data_in': None
        }
    
    def _create_p09_config(self, params: Dict, core_id: Tuple[int, int]) -> Dict:
        """创建p09配置"""
        router = {
            'rhead_mode': 1,
            'send_en': True,
            'receive_en': True,
            'send_num': params.get('send_num', 0),
            'receive_num': params.get('receive_num', 0),
            'addr_din_base': params.get('addr_din_base', 0x380),
            'addr_din_length': params.get('addr_din_length', 0),
            'addr_rhead_base': params.get('addr_rhead_base', 0),
            'addr_rhead_length': params.get('addr_rhead_length', 0),
            'addr_dout_base': params.get('addr_dout_base', 0x1000),
            'addr_dout_length': params.get('addr_dout_length', 0),
            'soma_in_en': params.get('soma_in_en', 1)
        }
        
        # 根据核心位置生成路由头
        x, y = core_id
        
        # 简化的路由：向右传递
        router['rheads'] = [{
            'S': 0, 'T': 1, 'P': 0, 'Q': 0,
            'X': 1,
            'Y': 0,
            'A': 0,
            'pack_per_Rhead': params.get('pack_per_Rhead', 0),
            'A_offset': 0,
            'Const': 0,
            'EN': 1
        }]
        
        return router
    
    def _create_conv_config(self, prim_type: str, params: Dict) -> Dict:
        """创建卷积配置"""
        if prim_type == 'p81':
            return {
                'px': params.get('px', 1),
                'py': params.get('py', 1),
                'cin': params.get('cin', 1),
                'cout': params.get('cout', 1),
                'kx': params.get('kx', 1),
                'ky': params.get('ky', 1),
                'sx': params.get('sx', 1),
                'sy': params.get('sy', 1),
                'addr_ina': params.get('addr_ina', 0),
                'addr_inb': params.get('addr_inb', 0),
                'addr_bias': params.get('addr_bias', 0),
                'addr_out': params.get('addr_out', 0),
                'ina_type': params.get('ina_type', 1),
                'inb_type': params.get('inb_type', 1),
                'load_bias': params.get('load_bias', 0)
            }
        else:  # p41
            return {
                'px': params.get('px', 1),
                'py': params.get('py', 1),
                'cin': params.get('cin', 1),
                'cout': params.get('cout', 1),
                'kx': params.get('kx', 1),
                'ky': params.get('ky', 1),
                'sx': params.get('sx', 1),
                'sy': params.get('sy', 1),
                'addr_ina': params.get('addr_ina', 0),
                'addr_inb': params.get('addr_inb', 0),
                'addr_bias': params.get('addr_bias', 0),
                'addr_out': params.get('addr_out', 0),
                'pad_top': params.get('pad_top', 0),
                'pad_down': params.get('pad_down', 0),
                'pad_left': params.get('pad_left', 0),
                'pad_right': params.get('pad_right', 0),
                'ina_type': params.get('ina_type', 1),
                'inb_type': params.get('inb_type', 1),
                'load_bias': params.get('load_bias', 0)
            }
    
    def _create_pX5_config(self, params: Dict) -> Dict:
        """创建pX5配置"""
        return {
            'mode': params.get('mode', 'max'),
            'addr_in': params.get('addr_in', 0),
            'addr_out': params.get('addr_out', 0),
            'cin': params.get('cin', 1),
            'cout': params.get('cout', 1),
            'px': params.get('px', 1),
            'py': params.get('py', 1),
            'kx': params.get('kx', 1),
            'ky': params.get('ky', 1),
            'sx': params.get('sx', 1),
            'sy': params.get('sy', 1),
            'cmp_c': params.get('cmp_c', 0x0),
            'type_in': params.get('type_in', 1),
            'type_out': params.get('type_out', 1),
            'in_cut_start': params.get('in_cut_start', 0),
            'row_ck_on': params.get('row_ck_on', 1),
            'in_row_max': params.get('in_row_max', 2)
        }
    
    def _get_core_neighbors(self, core_id: Tuple[int, int]) -> List[Dict]:
        """获取核心邻居"""
        x, y = core_id
        neighbors = []
        
        if x > 0:
            neighbors.append({'direction': 'west', 'core': [x-1, y]})
        if x < 7:
            neighbors.append({'direction': 'east', 'core': [x+1, y]})
        if y > 0:
            neighbors.append({'direction': 'north', 'core': [x, y-1]})
        if y < 0:  # 假设只有一行
            neighbors.append({'direction': 'south', 'core': [x, y+1]})
        
        return neighbors
    
    def _create_router_prim(self, core_id_str: str) -> Dict:
        """创建路由器原语"""
        # 解析核心ID
        try:
            x_str, y_str = core_id_str.split('_')
            x = int(x_str)
            y = int(y_str)
        except:
            x, y = 0, 0
        
        # 根据核心位置决定路由方向
        if x == 7:  # 最右边的核心
            # 向上或向下路由
            direction_x = 0
            direction_y = -1 if y > 0 else 1
        else:
            # 向右路由
            direction_x = 1
            direction_y = 0
        
        return {
            'axon': None,
            'soma1': None,
            'router': {
                'rhead_mode': 1,
                'send_en': True,
                'receive_en': True,
                'send_num': 15,  # 16字节数据
                'receive_num': 15,
                'addr_din_base': 0x380,
                'addr_din_length': 1,
                'addr_rhead_base': 0,
                'addr_rhead_length': 0,
                'addr_dout_base': 0x1000,
                'addr_dout_length': 0,
                'soma_in_en': 1,
                'rheads': [{
                    'S': 0, 'T': 1, 'P': 0, 'Q': 0,
                    'X': direction_x,
                    'Y': direction_y,
                    'A': 0,
                    'pack_per_Rhead': 15,
                    'A_offset': 0,
                    'Const': 0,
                    'EN': 1
                }]
            },
            'soma2': None
        }


class AutoMapperPipelineV2:
    """自动化映射流水线V2 - 适配TVM 0.20.0dev"""
    
    def __init__(self, hw_config: Optional[HardwareConfig] = None):
        self.hw_config = hw_config or HardwareConfig()
        self.onnx_analyzer = None
        self.hardware_mapper = HardwareMapperV2(self.hw_config)
        self.config_generator = ConfigGeneratorV2(self.hardware_mapper)
        
    def run_pipeline(self, onnx_path: str, output_dir: str = "./output", 
                    quantization: bool = False) -> Dict:
        """运行完整流水线"""
        import os
        import json
        import time
        
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("自动化映射流水线 V2.0")
        logger.info(f"输入: {onnx_path}")
        logger.info(f"输出: {output_dir}")
        logger.info(f"量化: {quantization}")
        logger.info("=" * 80)
        
        # 步骤1: 分析ONNX模型
        logger.info("步骤1: 分析ONNX模型...")
        self.onnx_analyzer = ONNXModelAnalyzer(onnx_path)
        model_analysis = self.onnx_analyzer.analyze()
        
        # 保存分析结果 - 使用自定义JSON编码器
        with open(f"{output_dir}/model_analysis.json", "w") as f:
            json.dump(model_analysis, f, indent=2, cls=JSONEncoder)
        
        logger.info(f"模型分析完成:")
        logger.info(f"  算子数量: {model_analysis['statistics']['total_operators']}")
        logger.info(f"  参数数量: {model_analysis['statistics']['total_parameters']}")
        logger.info(f"  参数大小: {model_analysis['statistics']['total_parameters_size'] / 1024:.2f} KB")
        
        # 步骤3: 硬件映射
        logger.info("步骤3: 硬件映射...")
        mapped_model = self.hardware_mapper.map_model(model_analysis, quantization)
        
        # 保存映射结果 - 使用自定义JSON编码器
        with open(f"{output_dir}/mapped_model.json", "w") as f:
            json.dump(mapped_model, f, indent=2, cls=JSONEncoder)
        
        # 步骤4: 生成配置
        logger.info("步骤4: 生成硬件配置...")
        all_configs = self.config_generator.generate_all_configs(mapped_model)
        
        # 保存配置 - 使用自定义JSON编码器
        with open(f"{output_dir}/all_configs.json", "w") as f:
            json.dump(all_configs, f, indent=2, cls=JSONEncoder)
        
        # 步骤5: 生成代码
        logger.info("步骤5: 生成可执行代码...")
        self._generate_code(all_configs, mapped_model, output_dir)
        
        # 步骤6: 生成报告
        logger.info("步骤6: 生成分析报告...")
        self._generate_report(mapped_model, all_configs, output_dir)
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info(f"自动化映射完成!")
        logger.info(f"总耗时: {elapsed_time:.2f} 秒")
        logger.info(f"结果保存到: {output_dir}")
        logger.info("=" * 80)
        
        return {
            'model_analysis': model_analysis,
            'mapped_model': mapped_model,
            'configs': all_configs,
            'performance': self._calculate_performance(mapped_model)
        }
    
    def _generate_code(self, configs: Dict, mapped_model: Dict, output_dir: str):
        """生成可执行代码"""
        import os
        
        # 生成Python配置文件
        for config_name, config in configs.items():
            if config_name != 'global_config':
                python_code = self._config_to_python(config_name, config)
                
                with open(f"{output_dir}/{config_name}.py", "w") as f:
                    f.write(python_code)
        
        # 生成主测试文件
        main_code = self._generate_main_test_file(configs, mapped_model)
        with open(f"{output_dir}/test_main.py", "w") as f:
            f.write(main_code)
        
        # 生成数据加载文件
        data_code = self._generate_data_loader(mapped_model)
        with open(f"{output_dir}/data_loader.py", "w") as f:
            f.write(data_code)
    
    def _config_to_python(self, config_name: str, config: Dict) -> str:
        """将配置转换为Python代码"""
        # 简化实现，生成基本的配置函数
        if config_name == 'g0_config':
            func_name = 'gen_0_map_config'
        elif config_name == 'g1_config':
            func_name = 'gen_1_map_config'
        elif config_name == 'g2_config':
            func_name = 'gen_2_map_config'
        else:
            func_name = f'gen_{config_name}'
        
        # 从配置中提取芯片信息
        chip_info = config.get('chip', [0, 0])
        
        code = f'''"""
自动生成的{config_name}配置
"""

import numpy as np
from itertools import product

def {func_name}(phase_en, clock_in_phase, size_x=1, size_y=1, data=None, **kwargs):
    """{config.get('description', '自动生成的配置')}"""
    
    # 硬件配置
    chip = ({chip_info[0]}, {chip_info[1]})  # 转换为元组
    
    # 初始化配置字典
    map_config = {{
        'sim_clock': 100000,
        (chip, 0): {{
            0: {{
                'clock': clock_in_phase,
                'mode': 1,
            }}
        }}
    }}
    
    # 为每个核心添加配置
    for core_y, core_x in product(range(size_y), range(size_x)):
        core_id = (chip, (core_x, core_y))
        map_config[(chip, 0)][0][core_id] = {{
            'prims': []
        }}
    
    phase_group = map_config[(chip, 0)][0]
    
    # 这里可以添加具体的原语配置
    # 示例: 添加一个数据搬运原语
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = (chip, (core_x, core_y))
            
            # 添加示例原语
            prim = {{
                'axon': None,
                'soma1': None,
                'router': None,
                'soma2': {{
                    'addr_in': 0x0000 >> 2,
                    'addr_out': 0x8400,
                    'addr_ciso': 0x10000 >> 2,
                    'length_in': 1024,
                    'num_in': 12,
                    'length_ciso': 1,
                    'num_ciso': 12,
                    'length_out': 1024,
                    'num_out': 12,
                    'type_in': 1,
                    'type_out': 1,
                    'data_in': None
                }}
            }}
            
            phase_group[core_id]['prims'].append(prim)
    
    return map_config

if __name__ == '__main__':
    # 测试配置生成
    phase = np.ones(10).astype(int)
    config = {func_name}(phase, 200000)
    print(f"{config_name} 生成成功")
'''
        
        return code
    
    def _generate_main_test_file(self, configs: Dict, mapped_model: Dict) -> str:
        """生成主测试文件"""
        model_stats = mapped_model.get('statistics', {}).get('model', {})
        hw_stats = mapped_model.get('statistics', {}).get('hardware', {})
        
        code = f'''"""
自动化生成的测试主程序
模型统计:
  算子数量: {model_stats.get('total_operators', 'N/A')}
  参数大小: {model_stats.get('total_parameters_size', 0) / 1024:.2f} KB
  
硬件统计:
  总周期数: {hw_stats.get('total_cycles', 0):,}
  总MAC操作: {hw_stats.get('total_macs', 0):,}
  估计延迟: {hw_stats.get('estimated_latency_ms', 0):.2f} ms
"""

import numpy as np
import sys
import os

sys.path.append(os.getcwd())

# 导入生成的配置
try:
    from g0_config import gen_0_map_config
    from g1_config import gen_1_map_config  
    from g2_config import gen_2_map_config
except ImportError as e:
    print(f"导入配置失败: {{e}}")
    print("请确保已运行自动化映射流水线生成配置文件")
    sys.exit(1)

from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch


def run_automated_test(case_file_name='auto_generated_test', send_to_fpga=True):
    """运行自动化测试"""
    
    print("=" * 60)
    print("开始自动化测试")
    print("=" * 60)
    
    # 初始化配置生成器
    config = MapConfigGen()
    
    # 时钟配置
    clock_in_phase = 200000
    
    # 阶段使能（所有阶段使能）
    total_phases = 10  # 假设10个阶段
    phase = np.ones(total_phases).astype(int)
    
    # 添加Group0配置 (FPGA)
    print("1. 添加Group0配置...")
    config_0 = gen_0_map_config(
        phase_en=phase,
        clock_in_phase=clock_in_phase,
        size_x=1,
        size_y=1,
        data=None,
        out_data_en=True,
        in_data_en=not send_to_fpga
    )
    config.add_config(config_0, core_offset=(1, 0))
    
    # 添加Group1配置 (计算核心)
    print("2. 添加Group1配置...")
    config_1 = gen_1_map_config(
        phase_en=phase,
        clock_in_phase=clock_in_phase,
        size_x=1,
        size_y=1,
        in_data_en=True,
        out_data_en=True
    )
    config.add_config(config_1, core_offset=(0, 0))
    
    # 添加Group2配置 (路由核心)
    print("3. 添加Group2配置...")
    config_2 = gen_2_map_config(
        phase_en=phase,
        clock_in_phase=clock_in_phase,
        size_x=8,
        size_y=1,
        in_data_en=True,
        out_data_en=True,
        send_to_fpga=send_to_fpga
    )
    config.add_config(config_2, core_offset=(0, 1))
    
    # 添加路由信息
    print("4. 添加路由信息...")
    MapConfigGen.add_router_info(map_config=config.map_config)
    
    # 添加初始数据传输prim
    print("5. 添加初始数据传输prim...")
    prim = {{
        'axon': None,
        'soma1': None,
        'router': None,
        'soma2': {{
            'addr_in': 0x0000 >> 2,
            'addr_out': 0x8400,
            'addr_ciso': 0x10000 >> 2,
            'length_in': 1024,
            'num_in': 12,
            'length_ciso': 1,
            'num_ciso': 12,
            'length_out': 1024,
            'num_out': 12,
            'type_in': 1,
            'type_out': 1,
            'data_in': None
        }}
    }}
    
    MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)
    
    # 配置时钟
    config.map_config['sim_clock'] = 100000
    config.map_config['step_clock'] = {{
        ((0, 0), 0): (100000 - 1, 100000)
    }}
    
    # 准备测试环境
    c_path = os.getcwd()
    out_files_path = os.getcwd() + "/simulator/Out_files/" + case_file_name + "/"
    
    if os.path.exists(out_files_path):
        os.chdir(out_files_path)
        if sys.platform.startswith('win'):
            os.system('rd/s/q cmp_out')
        else:
            os.system('rm -rf cmp_out')
        os.chdir(c_path)
    else:
        os.makedirs(out_files_path, exist_ok=True)
    
    # 配置测试参数
    test_config = {{
        'tb_name': case_file_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'test_group_phase': [(0, 1)]
    }}
    
    # 运行测试
    print("6. 运行硬件模拟测试...")
    tester = TestEngine(config.map_config, test_config)
    
    try:
        result = tester.run_test()
        if result:
            print("\\n✅ 测试通过!")
        else:
            print("\\n❌ 测试失败!")
        return result
    except Exception as e:
        print(f"\\n⚠️ 测试过程中出现错误: {{e}}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='运行自动化生成的测试')
    parser.add_argument('--name', type=str, default='auto_generated_test', 
                       help='测试案例名称')
    parser.add_argument('--no-fpga', action='store_true', 
                       help='不发送到FPGA')
    
    args = parser.parse_args()
    
    # 运行自动化测试
    success = run_automated_test(args.name, not args.no_fpga)
    
    if success:
        print("\\n🎉 自动化测试流程完成!")
    else:
        print("\\n💥 自动化测试流程失败!")
        sys.exit(1)
'''
        
        return code
    
    def _generate_data_loader(self, mapped_model: Dict) -> str:
        """生成数据加载器"""
        code = '''"""
自动化生成的数据加载器
用于加载和准备测试数据
"""

import numpy as np
import json
import os


class AutoDataLoader:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.parameters = {}
        self.input_data = None
        self.expected_output = None
        
    def load_from_config(self, config_path):
        """从配置文件加载数据"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 加载内存映射
            if 'memory_map' in config:
                self.memory_map = config['memory_map']
                print(f"加载内存映射: {len(self.memory_map)} 个张量")
            
            # 加载模型统计
            if 'model_stats' in config:
                self.model_stats = config['model_stats']
                print(f"模型参数大小: {self.model_stats.get('total_parameters_size', 0) / 1024:.2f} KB")
            
            return True
            
        except Exception as e:
            print(f"加载配置失败: {e}")
            return False
    
    def generate_test_data(self, input_shape=(1, 1, 28, 28), dtype=np.int8):
        """生成测试数据"""
        # 生成随机输入数据
        self.input_data = np.random.randint(
            -128, 128, 
            size=input_shape, 
            dtype=dtype
        )
        
        # 生成随机权重（简化）
        self.parameters = {
            'conv1': {
                'weight': np.random.randint(-128, 128, size=(6, 1, 5, 5), dtype=dtype),
                'bias': np.random.randint(-1000, 1000, size=(6,), dtype=np.int32)
            }
        }
        
        # 生成预期输出（简化）
        self.expected_output = np.random.randint(
            -128, 128,
            size=(1, 10),
            dtype=dtype
        )
        
        print(f"生成测试数据:")
        print(f"  输入形状: {input_shape}")
        print(f"  参数数量: {len(self.parameters)}")
        
        return self.input_data, self.parameters, self.expected_output
    
    def save_test_data(self, output_path):
        """保存测试数据"""
        data = {
            'input': self.input_data.tolist() if self.input_data is not None else [],
            'parameters': {k: v.tolist() for k, v in self.parameters.items()},
            'expected_output': self.expected_output.tolist() if self.expected_output is not None else []
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        print(f"测试数据保存到: {output_path}")
    
    def load_test_data(self, input_path):
        """加载测试数据"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.input_data = np.array(data['input'], dtype=np.int8)
        self.parameters = {k: np.array(v, dtype=np.int8 if 'weight' in k else np.int32) 
                          for k, v in data['parameters'].items()}
        self.expected_output = np.array(data['expected_output'], dtype=np.int8)
        
        print(f"从 {input_path} 加载测试数据")
        return self.input_data, self.parameters, self.expected_output


if __name__ == '__main__':
    # 示例用法
    loader = AutoDataLoader()
    
    # 生成测试数据
    input_data, parameters, expected_output = loader.generate_test_data()
    
    # 保存测试数据
    loader.save_test_data('./test_data.json')
    
    print("数据加载器测试完成")
'''
        
        return code
    
    def _generate_report(self, mapped_model: Dict, configs: Dict, output_dir: str):
        """生成分析报告"""
        import json
        import time
        
        # 生成JSON报告
        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_info': mapped_model.get('model_info', {}),
            'hardware_config': mapped_model.get('hardware_config', {}),
            'statistics': mapped_model.get('statistics', {}),
            'memory_allocation': self.hardware_mapper.memory_allocator.get_allocation_report(),
            'phase_summary': {
                'total_phases': len(mapped_model.get('phase_schedule', [])),
                'total_cycles': sum(p.get('clock_cycles', 0) 
                                  for p in mapped_model.get('phase_schedule', [])),
                'estimated_latency_ms': sum(p.get('estimated_latency_ms', 0) 
                                          for p in mapped_model.get('phase_schedule', []))
            },
            'config_summary': {
                'g0_cores': len(configs.get('g0_config', {}).get('cores', {})),
                'g1_cores': len(configs.get('g1_config', {}).get('cores', {})),
                'g2_cores': len(configs.get('g2_config', {}).get('cores', {}))
            }
        }
        
        # 保存报告 - 使用自定义JSON编码器
        with open(f"{output_dir}/analysis_report.json", "w") as f:
            json.dump(report, f, indent=2, cls=JSONEncoder)
        
        # 生成HTML报告
        self._generate_html_report(report, output_dir)
        
        # 生成简化的文本报告
        self._generate_text_report(report, output_dir)
    
    def _generate_html_report(self, report: Dict, output_dir: str):
        """生成HTML格式的报告"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>自动化映射分析报告</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; font-weight: bold; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f9f9f9; border-radius: 5px; min-width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .good {{ color: #4CAF50; }}
        .warning {{ color: #FF9800; }}
        .bad {{ color: #F44336; }}
        .summary {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
        .timestamp {{ color: #888; font-size: 14px; text-align: right; }}
    </style>
</head>
<body>
    <div class="timestamp">生成时间: {report['timestamp']}</div>
    <h1>自动化映射分析报告</h1>
    
    <div class="section summary">
        <h2>执行摘要</h2>
        <div class="metric">
            <div class="metric-value">{report['statistics']['model'].get('total_operators', 0)}</div>
            <div class="metric-label">算子数量</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['statistics']['model'].get('total_parameters_size', 0) / 1024:.1f} KB</div>
            <div class="metric-label">参数大小</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['phase_summary']['total_phases']}</div>
            <div class="metric-label">阶段数量</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['phase_summary']['total_cycles']:,}</div>
            <div class="metric-label">总时钟周期</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['phase_summary']['estimated_latency_ms']:.1f} ms</div>
            <div class="metric-label">估计延迟</div>
        </div>
    </div>
    
    <div class="section">
        <h2>性能指标</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>值</th>
                <th>说明</th>
            </tr>
            <tr>
                <td>总MAC操作</td>
                <td>{report['statistics']['hardware'].get('total_macs', 0):,}</td>
                <td>模型总计算量</td>
            </tr>
            <tr>
                <td>MAC/周期</td>
                <td>{report['statistics']['hardware'].get('macs_per_cycle', 0):.2f}</td>
                <td>每时钟周期MAC操作数</td>
            </tr>
            <tr>
                <td>硬件利用率</td>
                <td>{report['statistics']['hardware'].get('utilization', 0) * 100:.1f}%</td>
                <td>计算单元利用率</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>内存分配</h2>
        <table>
            <tr>
                <th>内存区域</th>
                <th>总大小</th>
                <th>已使用</th>
                <th>剩余</th>
                <th>使用率</th>
                <th>基地址</th>
            </tr>
"""
        
        # 添加内存使用情况
        mem_usage = report['memory_allocation']['mem_usage']
        for mem_type, info in mem_usage.items():
            usage_class = 'good'
            if info['usage_percent'] > 80:
                usage_class = 'bad'
            elif info['usage_percent'] > 60:
                usage_class = 'warning'
            
            html += f"""
            <tr>
                <td>{mem_type}</td>
                <td>{info['total']:,} B</td>
                <td>{info['used']:,} B</td>
                <td>{info['free']:,} B</td>
                <td class="{usage_class}">{info['usage_percent']:.1f}%</td>
                <td>{info['base_address']}</td>
            </tr>"""
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>硬件配置</h2>
        <table>
            <tr>
                <th>配置项</th>
                <th>值</th>
            </tr>
"""
        
        # 添加硬件配置
        hw_config = report.get('hardware_config', {})
        for key, value in hw_config.items():
            html += f"""
            <tr>
                <td>{key}</td>
                <td>{value}</td>
            </tr>"""
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>核心分配</h2>
        <table>
            <tr>
                <th>组</th>
                <th>核心数量</th>
                <th>描述</th>
            </tr>
            <tr>
                <td>Group0 (FPGA)</td>
                <td>{report['config_summary']['g0_cores']}</td>
                <td>数据输入输出</td>
            </tr>
            <tr>
                <td>Group1 (计算)</td>
                <td>{report['config_summary']['g1_cores']}</td>
                <td>神经网络计算</td>
            </tr>
            <tr>
                <td>Group2 (路由)</td>
                <td>{report['config_summary']['g2_cores']}</td>
                <td>数据路由和通信</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>建议和优化</h2>
        <ul>
            <li><strong>内存优化:</strong> 当前内存使用率正常，可以考虑进一步优化数据布局</li>
            <li><strong>性能优化:</strong> 硬件利用率有待提高，可以考虑算子融合或流水线优化</li>
            <li><strong>功耗优化:</strong> 可以根据实际部署场景调整时钟频率</li>
            <li><strong>面积优化:</strong> 对于资源受限场景，可以考虑减少计算核心数量</li>
        </ul>
    </div>
    
    <div class="timestamp">
        报告生成完成 - 自动化映射系统
    </div>
</body>
</html>"""
        
        with open(f"{output_dir}/analysis_report.html", "w") as f:
            f.write(html)
    
    def _generate_text_report(self, report: Dict, output_dir: str):
        """生成文本格式的报告"""
        text = f"""自动化映射分析报告
=====================

生成时间: {report['timestamp']}

1. 模型信息
-----------
算子数量: {report['statistics']['model'].get('total_operators', 0)}
参数大小: {report['statistics']['model'].get('total_parameters_size', 0) / 1024:.1f} KB
激活大小: {report['statistics']['model'].get('total_activations_size', 0) / 1024:.1f} KB

2. 硬件性能
-----------
总时钟周期: {report['phase_summary']['total_cycles']:,}
总MAC操作: {report['statistics']['hardware'].get('total_macs', 0):,}
MAC/周期: {report['statistics']['hardware'].get('macs_per_cycle', 0):.2f}
硬件利用率: {report['statistics']['hardware'].get('utilization', 0) * 100:.1f}%
估计延迟: {report['phase_summary']['estimated_latency_ms']:.1f} ms

3. 内存使用
-----------
"""
        
        # 内存使用情况
        mem_usage = report['memory_allocation']['mem_usage']
        for mem_type, info in mem_usage.items():
            text += f"{mem_type}: {info['used']:,} / {info['total']:,} B ({info['usage_percent']:.1f}%)\n"
        
        text += f"""
4. 核心分配
-----------
Group0 (FPGA): {report['config_summary']['g0_cores']} 核心
Group1 (计算): {report['config_summary']['g1_cores']} 核心  
Group2 (路由): {report['config_summary']['g2_cores']} 核心

5. 硬件配置
-----------
"""
        
        # 硬件配置
        hw_config = report.get('hardware_config', {})
        for key, value in hw_config.items():
            if key not in ['core_grid']:  # 跳过core_grid，因为它是一个列表
                text += f"{key}: {value}\n"
        
        text += f"""
6. 建议
-------
"""
        
        # 根据内存使用率给出建议
        max_usage = max(info['usage_percent'] for info in mem_usage.values())
        if max_usage > 90:
            text += "- ⚠️ 内存使用率较高，建议优化数据布局或增加内存\n"
        elif max_usage > 70:
            text += "- ℹ️ 内存使用率适中，有优化空间\n"
        else:
            text += "- ✅ 内存使用率良好\n"
        
        # 根据硬件利用率给出建议
        utilization = report['statistics']['hardware'].get('utilization', 0)
        if utilization < 0.3:
            text += "- ⚠️ 硬件利用率较低，建议进行算子融合或流水线优化\n"
        elif utilization < 0.6:
            text += "- ℹ️ 硬件利用率中等，可以考虑进一步优化\n"
        else:
            text += "- ✅ 硬件利用率良好\n"
        
        text += f"""
7. 生成的文件
-------------
- model_analysis.json: 模型分析结果
- mapped_model.json: 硬件映射结果  
- all_configs.json: 所有硬件配置
- g0_config.py: Group0配置代码
- g1_config.py: Group1配置代码
- g2_config.py: Group2配置代码
- test_main.py: 主测试程序
- data_loader.py: 数据加载器
- analysis_report.json: 详细分析报告
- analysis_report.html: HTML格式报告
- analysis_report.txt: 文本格式报告

报告生成完成!
"""
        
        with open(f"{output_dir}/analysis_report.txt", "w") as f:
            f.write(text)
    
    def _calculate_performance(self, mapped_model: Dict) -> Dict:
        """计算性能指标"""
        return {
            'score': self._calculate_performance_score(mapped_model),
            'recommendations': self._generate_recommendations(mapped_model)
        }
    
    def _calculate_performance_score(self, mapped_model: Dict) -> float:
        """计算性能评分（0-100）"""
        # 简化评分算法
        score = 100
        
        # 内存使用率扣分
        mem_report = self.hardware_mapper.memory_allocator.get_allocation_report()
        for mem_info in mem_report['mem_usage'].values():
            if mem_info['usage_percent'] > 90:
                score -= 20
            elif mem_info['usage_percent'] > 80:
                score -= 10
            elif mem_info['usage_percent'] > 70:
                score -= 5
        
        # 硬件利用率扣分
        hw_stats = mapped_model.get('statistics', {}).get('hardware', {})
        utilization = hw_stats.get('utilization', 0)
        if utilization < 0.3:
            score -= 20
        elif utilization < 0.5:
            score -= 10
        elif utilization < 0.7:
            score -= 5
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, mapped_model: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 内存优化建议
        mem_report = self.hardware_mapper.memory_allocator.get_allocation_report()
        for mem_type, mem_info in mem_report['mem_usage'].items():
            if mem_info['usage_percent'] > 80:
                recommendations.append(f"内存区域 {mem_type} 使用率较高 ({mem_info['usage_percent']:.1f}%)，建议优化数据布局")
        
        # 性能优化建议
        hw_stats = mapped_model.get('statistics', {}).get('hardware', {})
        if hw_stats.get('utilization', 0) < 0.5:
            recommendations.append("硬件利用率较低，建议进行算子融合或流水线优化")
        
        # 配置优化建议
        if len(mapped_model.get('phase_schedule', [])) > 20:
            recommendations.append("阶段数量较多，建议合并相关操作以减少阶段切换开销")
        
        return recommendations


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="基于ONNX的自动硬件映射系统 (适配TVM 0.20.0dev)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --onnx model.onnx
  %(prog)s --onnx model.onnx --output ./mapped --quantize
  %(prog)s --onnx model.onnx --cores 4 4 --mem0-size 128KB
        """
    )
    
    parser.add_argument("--onnx", type=str, required=True, 
                       help="ONNX模型文件路径")
    parser.add_argument("--output", type=str, default="./mapped_output",
                       help="输出目录 (默认: ./mapped_output)")
    parser.add_argument("--quantize", action="store_true",
                       help="启用量化 (INT8权重和激活)")
    parser.add_argument("--mem0-size", type=str, default="64KB",
                       help="Mem0大小 (默认: 64KB)")
    parser.add_argument("--mem1-size", type=str, default="64KB",
                       help="Mem1大小 (默认: 64KB)")
    parser.add_argument("--cores", type=int, nargs=2, default=[8, 1],
                       help="核心阵列尺寸 (默认: 8 1)")
    
    args = parser.parse_args()
    
    # 解析内存大小
    def parse_memory_size(size_str):
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('B'):
            return int(size_str[:-1])
        else:
            return int(size_str)
    
    # 创建硬件配置
    hw_config = HardwareConfig(
        mem0_size=parse_memory_size(args.mem0_size),
        mem1_size=parse_memory_size(args.mem1_size),
        core_grid=tuple(args.cores)
    )
    
    print(f"硬件配置:")
    print(f"  核心阵列: {hw_config.core_grid[0]}x{hw_config.core_grid[1]}")
    print(f"  Mem0大小: {hw_config.mem0_size / 1024:.0f} KB")
    print(f"  Mem1大小: {hw_config.mem1_size / 1024:.0f} KB")
    print(f"  量化模式: {'启用' if args.quantize else '禁用'}")
    print()
    
    # 运行自动化流水线
    pipeline = AutoMapperPipelineV2(hw_config)
    
    try:
        result = pipeline.run_pipeline(
            args.onnx, 
            args.output, 
            args.quantize
        )
        
        # 显示结果摘要
        print("\n" + "="*60)
        print("映射结果摘要")
        print("="*60)
        
        if 'performance' in result:
            score = result['performance'].get('score', 0)
            print(f"性能评分: {score}/100")
            
            if score >= 80:
                print("评级: ✅ 优秀")
            elif score >= 60:
                print("评级: ⚠️ 良好")
            else:
                print("评级: ❌ 需要优化")
            
            recommendations = result['performance'].get('recommendations', [])
            if recommendations:
                print("\n优化建议:")
                for rec in recommendations:
                    print(f"  • {rec}")
        
        print(f"\n输出文件保存在: {args.output}")
        print("="*60)
        
    except Exception as e:
        print(f"运行流水线时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
