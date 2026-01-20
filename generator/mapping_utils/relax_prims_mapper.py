"""
Relax IR到prims.py原语的映射模块
将TVM生成的Relax IR映射到众核芯片的原语配置
"""

import tvm
import tvm.relax as relax
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from .prims import p06, p26, pX5, p09, p81, p41, p83, p07, p08, p02, p04, p03, p43

class RelaxPrimsMapper:
    """Relax IR到prims.py原语的映射器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Relax IR操作到prims原语的映射表
        self.relax_to_prims_map = {
            # 卷积操作
            "nn.conv2d": self._map_conv2d,
            "conv2d": self._map_conv2d,
            
            # 池化操作
            "nn.max_pool2d": self._map_max_pool2d,
            "nn.avg_pool2d": self._map_avg_pool2d,
            "max_pool2d": self._map_max_pool2d,
            "avg_pool2d": self._map_avg_pool2d,
            
            # 激活函数
            "nn.relu": self._map_relu,
            "relu": self._map_relu,
            "sigmoid": self._map_sigmoid,
            "tanh": self._map_tanh,
            
            # 全连接/矩阵乘法
            "matmul": self._map_matmul,
            "dense": self._map_dense,
            "nn.dense": self._map_dense,
            
            # 数据搬移操作
            "reshape": self._map_reshape,
            "transpose": self._map_transpose,
            "flatten": self._map_flatten,
            
            # 算术操作
            "add": self._map_add,
            "multiply": self._map_multiply,
            "subtract": self._map_subtract,
            
            # 特殊操作
            "batch_norm": self._map_batch_norm,
            "dropout": self._map_dropout,
        }
        
        # 默认内存地址配置
        self.default_addr_config = {
            'input_base': 0x0000,
            'weight_base': 0x1000,
            'output_base': 0x2000,
            'bias_base': 0x3000,
            'temp_base': 0x4000
        }
        
    def map_relax_to_prims(self, mod: tvm.IRModule, input_shape: Tuple[int, ...]) -> List[Dict[str, Any]]:
        """
        将Relax IR模块映射到prims原语配置
        
        Args:
            mod: TVM Relax IR模块
            input_shape: 输入张量形状
            
        Returns:
            prims配置列表，每个元素包含原语类型和参数
        """
        prims_config = []
        current_addr = self.default_addr_config['input_base']
        
        # 获取主函数
        main_func = mod['main']
        
        # 提取层信息
        layers = self._extract_layers_from_relax(main_func)
        
        self.logger.info(f"从Relax IR中提取到 {len(layers)} 个层")
        
        # 为每个层生成prims配置
        for i, layer in enumerate(layers):
            layer_type = layer.get('type', 'unknown')
            op_name = layer.get('op_name', '')
            attrs = layer.get('attrs', {})
            input_shape = layer.get('input_shape', input_shape)
            output_shape = layer.get('output_shape', input_shape)
            
            self.logger.info(f"映射层 {i}: {layer_type} ({op_name})")
            
            # 根据操作类型映射到prims
            if op_name in self.relax_to_prims_map:
                prim_config = self.relax_to_prims_map[op_name](
                    layer_type, attrs, input_shape, output_shape, current_addr
                )
            else:
                # 默认映射
                prim_config = self._map_default(
                    layer_type, attrs, input_shape, output_shape, current_addr
                )
            
            if prim_config:
                prims_config.append(prim_config)
                # 更新地址
                current_addr = self._calculate_next_addr(current_addr, output_shape)
        
        return prims_config
    
    def _extract_layers_from_relax(self, func: relax.Function) -> List[Dict[str, Any]]:
        """从Relax函数中提取层信息"""
        layers = []
        
        def visit_binding(binding, parent_block=None):
            """访问绑定并提取层信息"""
            if not hasattr(binding, 'var') or not hasattr(binding, 'value'):
                return
            
            layer_info = {
                'name': str(binding.var.name_hint) if hasattr(binding.var, 'name_hint') else str(binding.var),
                'type': 'unknown',
                'op_name': '',
                'attrs': {},
                'input_shape': None,
                'output_shape': None
            }
            
            # 处理Call操作
            if isinstance(binding.value, relax.Call):
                call = binding.value
                
                # 获取操作名称
                if hasattr(call.op, 'name'):
                    layer_info['op_name'] = call.op.name
                elif hasattr(call.op, '__name__'):
                    layer_info['op_name'] = call.op.__name__
                
                # 提取属性
                if hasattr(call, 'attrs'):
                    attrs = call.attrs
                    if attrs:
                        for attr_name in dir(attrs):
                            if not attr_name.startswith('_'):
                                attr_value = getattr(attrs, attr_name)
                                if attr_value is not None:
                                    layer_info['attrs'][attr_name] = attr_value
                
                # 推断层类型
                layer_info['type'] = self._infer_layer_type(layer_info['op_name'])
                
                # 获取形状信息
                if hasattr(binding.var, 'struct_info'):
                    struct_info = binding.var.struct_info
                    if hasattr(struct_info, 'shape'):
                        if hasattr(struct_info.shape, 'values'):
                            layer_info['output_shape'] = [int(d) for d in struct_info.shape.values]
                        elif hasattr(struct_info.shape, 'fields'):
                            layer_info['output_shape'] = [int(d) for d in struct_info.shape.fields]
            
            layers.append(layer_info)
        
        # 遍历函数体中的绑定
        if hasattr(func, 'body'):
            if hasattr(func.body, 'bindings'):
                for binding in func.body.bindings:
                    visit_binding(binding)
            elif isinstance(func.body, relax.SeqExpr):
                for block in func.body.blocks:
                    if hasattr(block, 'bindings'):
                        for binding in block.bindings:
                            visit_binding(binding)
        
        return layers
    
    def _infer_layer_type(self, op_name: str) -> str:
        """根据操作名称推断层类型"""
        op_name_lower = op_name.lower()
        
        if any(conv in op_name_lower for conv in ['conv', 'conv2d']):
            return 'conv2d'
        elif any(pool in op_name_lower for pool in ['pool', 'max_pool', 'avg_pool']):
            return 'pool'
        elif any(act in op_name_lower for act in ['relu', 'sigmoid', 'tanh']):
            return 'activation'
        elif any(dense in op_name_lower for dense in ['dense', 'matmul', 'linear']):
            return 'dense'
        elif any(reshape in op_name_lower for reshape in ['reshape', 'flatten']):
            return 'reshape'
        elif any(bn in op_name_lower for bn in ['batch_norm', 'batchnorm']):
            return 'batch_norm'
        else:
            return 'unknown'
    
    def _map_conv2d(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...], 
                   output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射卷积操作到prims"""
        # 提取卷积参数
        kernel_size = attrs.get('kernel_size', [3, 3])
        strides = attrs.get('strides', [1, 1])
        padding = attrs.get('padding', [0, 0, 0, 0])
        groups = attrs.get('groups', 1)
        dilation = attrs.get('dilation', [1, 1])
        
        # 计算卷积参数
        batch_size, in_channels, in_height, in_width = input_shape
        _, out_channels, out_height, out_width = output_shape
        
        # 选择卷积原语（根据参数选择p81或p41）
        if sum(padding) > 0:
            # 使用带填充的卷积p41
            prim_func = p41
            prim_name = 'p41'
        else:
            # 使用标准卷积p81
            prim_func = p81
            prim_name = 'p81'
        
        # 配置卷积参数
        prim_config = {
            'type': prim_name,
            'params': {
                'addr_in': base_addr,
                'addr_out': base_addr + 0x1000,
                'cin': in_channels,
                'cout': out_channels,
                'tensor_px': in_width,
                'tensor_py': in_height,
                'kernel_x': kernel_size[0],
                'kernel_y': kernel_size[1],
                'stride_x': strides[0],
                'stride_y': strides[1],
                'pad_top': padding[0],
                'pad_down': padding[2],
                'pad_left': padding[1],
                'pad_right': padding[3],
                'type_in': 0,  # 输入数据类型
                'type_out': 1,  # 输出数据类型
            },
            'memory_blocks': [
                {'name': 'input', 'start': base_addr, 'size': np.prod(input_shape)},
                {'name': 'output', 'start': base_addr + 0x1000, 'size': np.prod(output_shape)}
            ]
        }
        
        return prim_config
    
    def _map_max_pool2d(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                       output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射最大池化操作到prims"""
        kernel_size = attrs.get('pool_size', [2, 2])
        strides = attrs.get('strides', [2, 2])
        padding = attrs.get('padding', [0, 0, 0, 0])
        
        batch_size, channels, in_height, in_width = input_shape
        _, _, out_height, out_width = output_shape
        
        # 使用pX5原语进行池化
        prim_config = {
            'type': 'pX5',
            'params': {
                'mode': 'max',
                'addr_in': base_addr,
                'addr_out': base_addr + 0x1000,
                'cin': channels,
                'cout': channels,
                'px': in_width,
                'py': in_height,
                'kx': kernel_size[0],
                'ky': kernel_size[1],
                'sx': strides[0],
                'sy': strides[1],
                'pad_top': padding[0],
                'pad_down': padding[2],
                'pad_left': padding[1],
                'pad_right': padding[3],
                'type_in': 0,
                'type_out': 1,
            },
            'memory_blocks': [
                {'name': 'input', 'start': base_addr, 'size': np.prod(input_shape)},
                {'name': 'output', 'start': base_addr + 0x1000, 'size': np.prod(output_shape)}
            ]
        }
        
        return prim_config
    
    def _map_avg_pool2d(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                       output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射平均池化操作到prims"""
        # 与最大池化类似，只是mode改为'avg'
        config = self._map_max_pool2d(layer_type, attrs, input_shape, output_shape, base_addr)
        config['params']['mode'] = 'avg'
        return config
    
    def _map_relu(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                 output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射ReLU激活函数到prims"""
        # 使用p07 LUT原语实现ReLU
        prim_config = {
            'type': 'p07',
            'params': {
                'addr_in': base_addr,
                'addr_out': base_addr + 0x1000,
                'addr_lut': base_addr + 0x2000,  # LUT表地址
                'group_num': 1,
                'neuron_num': np.prod(input_shape),
                'lut_dw': 8,  # LUT数据位宽
                'type_in': 0,
                'type_out': 1,
            },
            'memory_blocks': [
                {'name': 'input', 'start': base_addr, 'size': np.prod(input_shape)},
                {'name': 'output', 'start': base_addr + 0x1000, 'size': np.prod(output_shape)},
                {'name': 'lut', 'start': base_addr + 0x2000, 'size': 256}  # 256字节LUT表
            ]
        }
        
        return prim_config
    
    def _map_sigmoid(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                    output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射Sigmoid激活函数到prims"""
        # 使用p07 LUT原语实现Sigmoid
        prim_config = {
            'type': 'p07',
            'params': {
                'addr_in': base_addr,
                'addr_out': base_addr + 0x1000,
                'addr_lut': base_addr + 0x2000,  # LUT表地址
                'group_num': 1,
                'neuron_num': np.prod(input_shape),
                'lut_dw': 8,  # LUT数据位宽
                'type_in': 0,
                'type_out': 1,
            },
            'memory_blocks': [
                {'name': 'input', 'start': base_addr, 'size': np.prod(input_shape)},
                {'name': 'output', 'start': base_addr + 0x1000, 'size': np.prod(output_shape)},
                {'name': 'lut', 'start': base_addr + 0x2000, 'size': 256}  # 256字节LUT表
            ]
        }
        
        return prim_config
    
    def _map_tanh(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                 output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射Tanh激活函数到prims"""
        # 使用p07 LUT原语实现Tanh
        prim_config = {
            'type': 'p07',
            'params': {
                'addr_in': base_addr,
                'addr_out': base_addr + 0x1000,
                'addr_lut': base_addr + 0x2000,  # LUT表地址
                'group_num': 1,
                'neuron_num': np.prod(input_shape),
                'lut_dw': 8,  # LUT数据位宽
                'type_in': 0,
                'type_out': 1,
            },
            'memory_blocks': [
                {'name': 'input', 'start': base_addr, 'size': np.prod(input_shape)},
                {'name': 'output', 'start': base_addr + 0x1000, 'size': np.prod(output_shape)},
                {'name': 'lut', 'start': base_addr + 0x2000, 'size': 256}  # 256字节LUT表
            ]
        }
        
        return prim_config
    
    def _map_matmul(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                   output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射矩阵乘法到prims"""
        # 使用p04 MLP原语
        if len(input_shape) == 4:
            # 如果是4D输入，先展平
            batch_size, channels, height, width = input_shape
            input_size = channels * height * width
        else:
            input_size = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
        
        if len(output_shape) == 4:
            batch_size, out_channels, out_height, out_width = output_shape
            output_size = out_channels * out_height * out_width
        else:
            output_size = output_shape[-1] if len(output_shape) > 1 else output_shape[0]
        
        prim_config = {
            'type': 'p04',
            'params': {
                'addr_in': base_addr,
                'addr_out': base_addr + 0x1000,
                'addr_weight': base_addr + 0x2000,  # 权重地址
                'addr_bias': base_addr + 0x3000,    # 偏置地址
                'input_size': input_size,
                'output_size': output_size,
                'batch_size': 1,  # 默认批大小为1
                'type_in': 0,
                'type_out': 1,
            },
            'memory_blocks': [
                {'name': 'input', 'start': base_addr, 'size': np.prod(input_shape)},
                {'name': 'output', 'start': base_addr + 0x1000, 'size': np.prod(output_shape)},
                {'name': 'weight', 'start': base_addr + 0x2000, 'size': input_size * output_size},
                {'name': 'bias', 'start': base_addr + 0x3000, 'size': output_size}
            ]
        }
        
        return prim_config
    
    def _map_dense(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                  output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射全连接层操作到prims"""
        # 直接使用matmul的映射方法，因为功能相似
        return self._map_matmul(layer_type, attrs, input_shape, output_shape, base_addr)

    def _map_default(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                    output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """默认映射方法（使用p06数据搬移原语）"""
        prim_config = {
            'type': 'p06',
            'params': {
                'addr_in': base_addr,
                'addr_out': base_addr + 0x1000,
                'data_size': np.prod(input_shape),
                'type_in': 0,
                'type_out': 1,
            },
            'memory_blocks': [
                {'name': 'input', 'start': base_addr, 'size': np.prod(input_shape)},
                {'name': 'output', 'start': base_addr + 0x1000, 'size': np.prod(output_shape)}
            ]
        }
        
        return prim_config
    
    def _calculate_next_addr(self, current_addr: int, output_shape: Tuple[int, ...]) -> int:
        """计算下一个内存地址"""
        # 计算当前层输出数据的大小（字节）
        data_size = np.prod(output_shape) * 4  # 假设每个数据点4字节
        
        # 对齐到4KB边界
        block_size = 0x1000  # 4KB
        next_addr = current_addr + data_size
        
        # 对齐到下一个4KB边界
        if next_addr % block_size != 0:
            next_addr = ((next_addr // block_size) + 1) * block_size
        
        return next_addr

    def _map_reshape(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                    output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射reshape操作到prims"""
        # 使用p06数据搬移原语
        return self._map_default(layer_type, attrs, input_shape, output_shape, base_addr)
    
    def _map_transpose(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                      output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射transpose操作到prims"""
        # 使用p06数据搬移原语
        return self._map_default(layer_type, attrs, input_shape, output_shape, base_addr)
    
    def _map_flatten(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                    output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射flatten操作到prims"""
        # 使用p06数据搬移原语
        return self._map_default(layer_type, attrs, input_shape, output_shape, base_addr)
    
    def _map_add(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射加法操作到prims"""
        # 使用p06数据搬移原语（暂不支持算术操作）
        return self._map_default(layer_type, attrs, input_shape, output_shape, base_addr)
    
    def _map_multiply(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                     output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射乘法操作到prims"""
        # 使用p06数据搬移原语（暂不支持算术操作）
        return self._map_default(layer_type, attrs, input_shape, output_shape, base_addr)
    
    def _map_subtract(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                     output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射减法操作到prims"""
        # 使用p06数据搬移原语（暂不支持算术操作）
        return self._map_default(layer_type, attrs, input_shape, output_shape, base_addr)
    
    def _map_batch_norm(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                       output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射批归一化操作到prims"""
        # 使用p06数据搬移原语（暂不支持批归一化）
        return self._map_default(layer_type, attrs, input_shape, output_shape, base_addr)
    
    def _map_dropout(self, layer_type: str, attrs: Dict, input_shape: Tuple[int, ...],
                    output_shape: Tuple[int, ...], base_addr: int) -> Dict[str, Any]:
        """映射dropout操作到prims"""
        # 使用p06数据搬移原语（暂不支持dropout）
        return self._map_default(layer_type, attrs, input_shape, output_shape, base_addr)

# 简化版映射器（用于快速测试）
class SimpleRelaxPrimsMapper:
    """简化版Relax IR到prims映射器"""
    
    def __init__(self):
        self.op_to_prim = {
            'conv2d': 'p81',
            'max_pool2d': 'pX5',
            'relu': 'p07',
            'dense': 'p04',
            'matmul': 'p04',
            'add': 'p06',
            'reshape': 'p06',
        }
    
    def map_simple(self, layer_info: List[Dict]) -> List[Dict]:
        """简化映射方法"""
        prims_config = []
        base_addr = 0x0000
        
        for i, layer in enumerate(layer_info):
            op_type = layer.get('type', 'unknown')
            prim_type = self.op_to_prim.get(op_type, 'p06')  # 默认使用数据搬移
            
            prim_config = {
                'prim_type': prim_type,
                'layer_index': i,
                'base_addr': base_addr,
                'params': layer.get('attrs', {})
            }
            
            prims_config.append(prim_config)
            base_addr += 0x1000  # 每个层分配4KB空间
        
        return prims_config