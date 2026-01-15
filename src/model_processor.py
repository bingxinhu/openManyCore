import tvm
import tvm.relax as relax
import numpy as np
import logging
import onnx
import onnx.checker
from typing import Tuple, Dict, List, Optional
from tvm.relax.frontend.onnx import from_onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

class ModelProcessor:
    """模型处理类,负责ONNX模型的加载、转换和分析(兼容TVM Relax)"""
    
    def __init__(self, config):
        """初始化模型处理器
        
        Args:
            config: 众核架构配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        # 层类型映射表，与原语保持一致
        self.layer_type_map = {
            # 卷积层
            "conv": "conv2d",
            "conv2d": "conv2d",
            "conv3d": "conv3d",
            
            # 全连接层
            "dense": "fully_connected",
            "fc": "fully_connected",
            "linear": "fully_connected",
            "fully_connected": "fully_connected",
            
            # 激活函数层
            "relu": "ann_activation",
            "sigmoid": "ann_activation",
            "tanh": "ann_activation",
            "leaky_relu": "ann_activation",
            "elu": "ann_activation",
            
            # 池化层
            "pool": "vector_accumulate",
            "maxpool": "vector_accumulate",
            "avgpool": "vector_accumulate",
            "max_pool": "vector_accumulate",
            "avg_pool": "vector_accumulate",
            
            # 向量运算层
            "add": "vector_accumulate",
            "sum": "vector_accumulate",
            "sub": "vector_accumulate",
            "mul": "vector_multiply",
            "matmul": "matrix_multiply",
            "dot": "vector_dot",
            "scale": "vector_scale",
            
            # 数据操作层
            "split": "vector_split",
            "concat": "vector_merge",
            "reshape": "vector_scale",
            "transpose": "matrix_transpose",
            "flatten": "vector_merge",
            
            # 特殊层
            "batch_norm": "vector_scale",
            "batchnorm": "vector_scale",
            "dropout": "vector_scale",
            "lrn": "vector_accumulate",
            
            # SNN激活函数
            "lif": "snn_activation",
            "spike": "snn_activation",
            
            # 查找表
            "lut": "lookup_table",
            
            # 比较操作
            "max": "vector_max",
            "min": "vector_min",
            "abs": "vector_accumulate",
        }
        
    def load_and_convert(self, convert_format=True, quantize_to_uint8=False) -> Tuple[tvm.IRModule, str, onnx.ModelProto]:
        """加载ONNX模型并转换为TVM Relax IR
        
        Args:
            convert_format: 是否将NCHW格式转换为NCWH格式
            quantize_to_uint8: 是否将模型量化为UINT8格式
        
        Returns:
            tvm.IRModule: TVM Relax中间表示
            str: 输入名称
            onnx.ModelProto: ONNX模型对象
        """
        # 加载ONNX模型
        onnx_path = self.config.get_onnx_path()
        self.logger.info(f"加载ONNX模型: {onnx_path}")
        
        # 如果文件不存在，创建一个简单的测试模型
        if not os.path.exists(onnx_path):
            self.logger.warning(f"ONNX模型文件不存在: {onnx_path}，创建测试模型")
            onnx_model = self._create_test_model()
        else:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
        
        self.logger.info("ONNX模型格式验证通过")
        
        # 如果需要量化为UINT8
        if quantize_to_uint8:
            onnx_model = self.quantize_to_uint8(onnx_model)
        
        # 获取输入名称和形状
        input_name = onnx_model.graph.input[0].name
        input_shape = self.config.get_input_shape()
        self.logger.info(f"模型输入名称: {input_name}, 形状: {input_shape}")
        
        # 转换为Relax IR（兼容大多数TVM版本）
        shape_dict = {input_name: input_shape}
        try:
            mod = from_onnx(onnx_model, shape_dict={input_name: input_shape}, keep_params_in_input=False, opset=13)
            if not isinstance(mod, tvm.IRModule):
                mod = tvm.IRModule.from_expr(mod)
        except Exception as e:
            self.logger.warning(f"opset=13转换失败,尝试opset=11: {str(e)}")
            mod = from_onnx(onnx_model, shape_dict={input_name: input_shape}, keep_params_in_input=False, opset=11)
            if not isinstance(mod, tvm.IRModule):
                mod = tvm.IRModule.from_expr(mod)
        
        # 应用优化转换
        self.logger.info("优化Relax IR...")
        try:
            with tvm.transform.PassContext(opt_level=3):
               mod = relax.transform.ToNonDataflow()(mod)
               mod = relax.transform.FoldConstant()(mod)
               mod = relax.transform.FuseOps()(mod)
               mod = relax.transform.RemoveUnusedOutputs()(mod)
               mod = relax.transform.RemoveUnusedParameters()(mod)
               mod = relax.transform.CallTIRRewrite()(mod)
               mod = relax.transform.LegalizeOps()(mod)
        except Exception as e:
            self.logger.warning(f"优化过程中出现错误: {e}，继续使用未优化的模型")
        print(mod)
        return mod, input_name, onnx_model
    
    def _create_test_model(self) -> onnx.ModelProto:
        """创建简单的测试模型"""
        import onnx.helper as helper
        import onnx
        
        # 创建一个简单的卷积模型用于测试
        input_shape = self.config.get_input_shape()
        
        # 输入
        X = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, input_shape)
        
        # 输出  
        Y = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [input_shape[0], 10, 1, 1])
        
        # 创建权重
        conv1_weight = helper.make_tensor(
            'conv1_weight',
            onnx.TensorProto.FLOAT,
            [10, input_shape[1], 3, 3],
            np.random.randn(10, input_shape[1], 3, 3).astype(np.float32).tobytes(),
            raw=True
        )
        
        # 创建节点
        conv1_node = helper.make_node(
            'Conv',
            inputs=['input', 'conv1_weight'],
            outputs=['conv1_output'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1]
        )
        
        relu_node = helper.make_node(
            'Relu',
            inputs=['conv1_output'],
            outputs=['relu_output']
        )
        
        # 全局平均池化
        pool_node = helper.make_node(
            'GlobalAveragePool',
            inputs=['relu_output'],
            outputs=['output']
        )
        
        # 创建图
        graph = helper.make_graph(
            [conv1_node, relu_node, pool_node],
            'test_model',
            [X],
            [Y],
            [conv1_weight]
        )
        
        # 创建模型
        model = helper.make_model(graph, producer_name='test_producer')
        onnx.checker.check_model(model)
        
        return model
    
    def quantize_to_uint8(self, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """将ONNX模型量化为UINT8格式
        
        Args:
            onnx_model: 原始ONNX模型
        
        Returns:
            onnx.ModelProto: 量化后的ONNX模型
        """
        self.logger.info("开始量化ONNX模型为UINT8格式...")
        
        # 保存原始模型到临时文件
        temp_path = "temp_original.onnx"
        quantized_path = "temp_quantized_uint8.onnx"
        onnx.save(onnx_model, temp_path)
        
        try:
            # 使用onnxruntime进行动态量化
            quantize_dynamic(
                temp_path,
                quantized_path,
                weight_type=QuantType.QUInt8,
                per_channel=False,
                reduce_range=True
            )
            
            # 加载量化后的模型
            quantized_model = onnx.load(quantized_path)
            onnx.checker.check_model(quantized_model)
            
            self.logger.info("ONNX模型量化为UINT8格式完成")
            return quantized_model
        except Exception as e:
            self.logger.warning(f"量化失败: {e}，返回原始模型")
            return onnx_model
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(quantized_path):
                os.remove(quantized_path)
    
    def extract_weights(self, onnx_model: onnx.ModelProto) -> Dict[str, np.ndarray]:
        """从ONNX模型中提取权重参数"""
        self.logger.info("从ONNX模型提取权重...")
        weights = {}
        for init in onnx_model.graph.initializer:
            weights[init.name] = onnx.numpy_helper.to_array(init)
        self.logger.info(f"提取完成，共 {len(weights)} 个权重参数")
        return weights
    
    def analyze_layers(self, mod: tvm.IRModule) -> List[Tuple[str, str]]:
        """分析模型层类型 (基于Relax IR)"""
        self.logger.info("分析模型层结构...")
        layers = []
        
        # 获取全局函数列表
        global_vars = list(mod.get_global_vars())
        
        for var in global_vars:
            func = mod[var]
            layer_name = var.name_hint.lower()
            
            # 跳过主函数和匿名函数
            if layer_name == "main":
                continue
            
            # 初始化层类型为未知
            layer_type = None
            
            # 方法1: 通过函数名关键字匹配
            layer_type = self._match_layer_by_name(layer_name)
            
            # 方法2: 如果名称匹配失败，通过函数内容推断
            if layer_type is None:
                layer_type = self._infer_layer_type_from_func(func, layer_name)
            
            # 如果还是无法识别，使用默认类型并记录警告
            if layer_type is None:
                self.logger.warning(f"未识别的层类型: {layer_name}")
                # 根据常见模式进行猜测
                if "conv" in layer_name:
                    layer_type = "conv2d"
                elif any(op in layer_name for op in ["add", "sum", "plus"]):
                    layer_type = "vector_accumulate"
                elif any(op in layer_name for op in ["relu", "sigmoid", "tanh"]):
                    layer_type = "ann_activation"
                elif any(op in layer_name for op in ["pool", "max", "avg"]):
                    layer_type = "vector_accumulate"
                else:
                    layer_type = "conv2d"  # 默认使用卷积
            
            # 映射到原语 - 这里是关键修正点
            primitive = self.get_primitive_for_layer(layer_type)
            
            # 修正：我们需要返回层类型，而不是原语，因为调度器需要知道实际的层类型
            # 但是原语名称和层类型名称通常相同，除了映射表中有特殊映射的情况
            # 这里我们返回 (layer_name, layer_type)，然后在代码生成时再映射到原语
            layers.append((layer_name, layer_type))  # 修改这里，返回层类型而不是原语
            
            self.logger.debug(f"层分析: {layer_name} -> 层类型: {layer_type} -> 原语: {primitive}")
        
        # 如果没有分析出层，添加默认层
        if not layers:
            layers = [
                ("conv1", "conv2d"),
                ("relu1", "ann_activation"), 
                ("pool1", "vector_accumulate"),
                ("fc1", "fully_connected")
            ]
        
        self.logger.info(f"模型层分析完成，共 {len(layers)} 层")
        return layers
    
    def _match_layer_by_name(self, layer_name: str) -> Optional[str]:
        """通过层名称关键字匹配层类型"""
        # 优先级匹配：先精确匹配，再部分匹配
        for key, layer_type in self.layer_type_map.items():
            # 精确匹配
            if layer_name == key:
                return layer_type
            
            # 部分匹配（包含关键词）
            if key in layer_name:
                return layer_type
        
        return None
    
    def _infer_layer_type_from_func(self, func: relax.Function, layer_name: str) -> Optional[str]:
        """从函数内部推断层类型"""
        if not hasattr(func, "body"):
            return None
        
        # 获取函数体
        body = func.body
        
        # 如果是Call节点，检查操作符
        if isinstance(body, relax.Call):
            # 获取操作符名称
            if hasattr(body.op, "name"):
                op_name = body.op.name.lower()
                
                # 匹配常见的操作符
                if any(conv in op_name for conv in ["conv2d", "conv", "convolution"]):
                    return "conv2d"
                elif any(pool in op_name for pool in ["pool", "max_pool", "avg_pool"]):
                    return "vector_accumulate"
                elif any(act in op_name for act in ["relu", "sigmoid", "tanh"]):
                    return "ann_activation"
                elif any(op in op_name for op in ["add", "sum"]):
                    return "vector_accumulate"
                elif any(op in op_name for op in ["matmul", "dense", "linear"]):
                    return "fully_connected"
                elif any(op in op_name for op in ["reshape", "transpose"]):
                    return "vector_scale"
        
        # 如果是TIR PrimFunc，检查函数属性
        elif hasattr(func, "attrs"):
            attrs = func.attrs
            if hasattr(attrs, "op_name"):
                op_name = attrs.op_name.lower()
                if "conv" in op_name:
                    return "conv2d"
                elif "pool" in op_name:
                    return "vector_accumulate"
                elif "add" in op_name:
                    return "vector_accumulate"
        
        return None
    
    def get_layer_input_output_shapes(self, onnx_model: onnx.ModelProto) -> Dict[str, Dict[str, List[int]]]:
        """获取各层的输入输出形状"""
        layer_shapes = {}
        
        # 处理输入层
        for input_tensor in onnx_model.graph.input:
            layer_name = input_tensor.name
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            layer_shapes[layer_name] = {
                "input": None,
                "output": shape
            }
        
        # 处理中间层和输出层
        all_tensors = list(onnx_model.graph.value_info) + list(onnx_model.graph.input) + list(onnx_model.graph.output)
        tensor_map = {t.name: t for t in all_tensors}
        
        for node in onnx_model.graph.node:
            layer_name = node.name or node.output[0]  # 处理无名称节点
                
            # 获取输入形状
            input_shapes = []
            for input_name in node.input:
                if input_name in tensor_map and hasattr(tensor_map[input_name].type, 'tensor_type'):
                    shape = [dim.dim_value for dim in tensor_map[input_name].type.tensor_type.shape.dim]
                    input_shapes.append(shape)
            
            # 获取输出形状
            output_shapes = []
            for output_name in node.output:
                if output_name in tensor_map and hasattr(tensor_map[output_name].type, 'tensor_type'):
                    shape = [dim.dim_value for dim in tensor_map[output_name].type.tensor_type.shape.dim]
                    output_shapes.append(shape)
            
            layer_shapes[layer_name] = {
                "input": input_shapes[0] if input_shapes else None,
                "output": output_shapes[0] if output_shapes else None
            }
        
        # 处理输出层
        for output_tensor in onnx_model.graph.output:
            layer_name = output_tensor.name
            if layer_name not in layer_shapes:
                shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                layer_shapes[layer_name] = {
                    "input": None,
                    "output": shape
                }
        print(layer_shapes)    
        return layer_shapes
    
    def get_primitive_for_layer(self, layer_type: str) -> str:
        """获取层类型对应的原语
        
        Args:
            layer_type: 层类型字符串
            
        Returns:
            对应的原语名称
        """
        return self.layer_type_map.get(layer_type, "conv2d")

    def validate_layer_mapping(self, layer_mapping: Dict[str, List[int]]) -> bool:
        """验证层到核心的映射是否有效
        
        Args:
            layer_mapping: 层到核心的映射字典
            
        Returns:
            bool: 映射是否有效
        """
        compute_cores = set(self.config.get_core_ids_by_role("compute"))
        valid = True
        
        for layer_name, core_ids in layer_mapping.items():
            for cid in core_ids:
                if cid not in compute_cores:
                    self.logger.error(f"层 {layer_name} 映射到无效的计算核心 {cid}")
                    valid = False
        
        return valid
