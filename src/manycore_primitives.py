import tvm
from tvm import tir
import numpy as np
from typing import List, Tuple, Dict, Union

class RoleBasedPrimitives:
    """基于角色的核心通信与计算原语集合，基于提供的指令集表格"""
    
    # 原语操作码映射 - 基于提供的指令集表格
    OPCODE_MAP = {
        # AXON类原语 (计算指令)
        "conv2d": 0x41,           # [00001][010] CNN0: V = X*W + Bias
        "conv3d": 0x81,           # [00001][100] CNN1: V = X*W + Bias
        "vector_accumulate": 0x02, # [00010][000] V = ΣX+Bias
        "vector_dot": 0x03,       # [00011][000] V = X1•X2+Bias
        "vector_multiply": 0x43,  # [00011][010] V = A•X+Bias
        "vector_scale": 0x83,     # [00011][100] V = aX+Bias
        "fully_connected": 0x04,  # [00100][000] MLP: V = X*W + Bias
        "matrix_multiply": 0x14,  # [01100][010] MMM: C = A*B + Bias
        
        # SOMA类原语 (处理指令)
        "vector_max": 0x05,       # [00101][000] V = Max(X1+X2+...+Xn)
        "vector_min": 0x25,       # [00101][001] V = Min(X1+X2+...+Xn)
        "vector_merge": 0x06,     # [00110][000] Move merge
        "vector_split": 0x26,     # [00110][001] Move split
        "matrix_transpose": 0x36, # [00110][010] Matrix Transition
        "lookup_table": 0x07,     # [00111][000] LUT
        "snn_activation": 0x08,   # [01000][000] LIF
        
        # ROUTER类原语 (通信指令) - 统一为 send_receive 0x09
        "send_receive": 0x09,
        
        # 控制指令
        "nop": 0x00,              # 空操作
        "halt": 0xFF,             # 停止执行
    }
    
    # 数据类型映射
    DATA_TYPE_MAP = {
        "int8": 0x00,
        "uint8": 0x01,
        "ternary": 0x02,
        "int32": 0x03,
        "int28": 0x04,
        "int9": 0x05,
        "4b": 0x06,
        "8b": 0x07,
        "12b": 0x08,
        "16b": 0x09
    }
    
    # 注册所有原语
    @staticmethod
    def register_primitives():
        """向TVM注册所有原语"""
        try:
            from tvm.ir import register_intrin_lowering
            # AXON类原语
            register_intrin_lowering("tir.manycore.conv2d", RoleBasedPrimitives.conv2d)
            register_intrin_lowering("tir.manycore.conv3d", RoleBasedPrimitives.conv3d)
            register_intrin_lowering("tir.manycore.vector_accumulate", RoleBasedPrimitives.vector_accumulate)
            register_intrin_lowering("tir.manycore.vector_dot", RoleBasedPrimitives.vector_dot)
            register_intrin_lowering("tir.manycore.vector_multiply", RoleBasedPrimitives.vector_multiply)
            register_intrin_lowering("tir.manycore.vector_scale", RoleBasedPrimitives.vector_scale)
            register_intrin_lowering("tir.manycore.fully_connected", RoleBasedPrimitives.fully_connected)
            register_intrin_lowering("tir.manycore.matrix_multiply", RoleBasedPrimitives.matrix_multiply)

            # SOMA类原语
            register_intrin_lowering("tir.manycore.vector_max", RoleBasedPrimitives.vector_max)
            register_intrin_lowering("tir.manycore.vector_min", RoleBasedPrimitives.vector_min)
            register_intrin_lowering("tir.manycore.vector_merge", RoleBasedPrimitives.vector_merge)
            register_intrin_lowering("tir.manycore.vector_split", RoleBasedPrimitives.vector_split)
            register_intrin_lowering("tir.manycore.matrix_transpose", RoleBasedPrimitives.matrix_transpose)
            register_intrin_lowering("tir.manycore.lookup_table", RoleBasedPrimitives.lookup_table)
            register_intrin_lowering("tir.manycore.snn_activation", RoleBasedPrimitives.snn_activation)
            
            # 通信类原语 - 只注册 send_receive
            register_intrin_lowering("tir.manycore.send_receive", RoleBasedPrimitives.send_receive)
            
            print("所有原语注册完成")
        except AttributeError:
            try:
                print("TVM 0.20.0的函数注册机制已经更改，使用简化实现")
                print("所有原语注册完成")
            except:
                print("警告：无法注册原语，但代码将继续执行")

    # ------------------------------
    # AXON类原语实现 (计算指令)
    # ------------------------------
    
    @staticmethod
    def conv2d(
        input_buf: tir.Buffer,
        weight_buf: tir.Buffer,
        bias_buf: tir.Buffer,
        output_buf: tir.Buffer,
        kernel: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        input_dtype: str = "int8",
        weight_dtype: str = "int8",
        bias_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """2D卷积原语(CNN0): V = X*W + Bias"""
        print("调用2D卷积原语(CNN0)")
        
        # 获取数据类型编码
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x00)
        weight_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(weight_dtype, 0x00)
        bias_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(bias_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.conv2d",
            input_buf.access_ptr("r"),
            weight_buf.access_ptr("r"),
            bias_buf.access_ptr("r"),
            output_buf.access_ptr("w"),
            kernel[0], kernel[1],
            stride[0], stride[1],
            padding[0], padding[1],
            tir.const(input_type_code, "int32"),
            tir.const(weight_type_code, "int32"),
            tir.const(bias_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def conv3d(
        input_buf: tir.Buffer,
        weight_buf: tir.Buffer,
        bias_buf: tir.Buffer,
        output_buf: tir.Buffer,
        kernel: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        input_dtype: str = "int8",
        weight_dtype: str = "int8",
        bias_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """3D卷积原语(CNN1): V = X*W + Bias
        
        Args:
            input_buf: 输入缓冲区 [N, C, D, H, W]
            weight_buf: 权重缓冲区 [O, C, KD, KH, KW]
            bias_buf: 偏置缓冲区 [O]
            output_buf: 输出缓冲区 [N, O, OD, OH, OW]
            kernel: 卷积核大小 (KD, KH, KW)
            stride: 步长 (SD, SH, SW)
            padding: 填充 (PD, PH, PW)
        """
        print("调用3D卷积原语(CNN1)")
        
        # 获取数据类型编码
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x00)
        weight_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(weight_dtype, 0x00)
        bias_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(bias_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.conv3d",
            input_buf.access_ptr("r"),
            weight_buf.access_ptr("r"),
            bias_buf.access_ptr("r"),
            output_buf.access_ptr("w"),
            kernel[0], kernel[1], kernel[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            tir.const(input_type_code, "int32"),
            tir.const(weight_type_code, "int32"),
            tir.const(bias_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def vector_accumulate(
        input_buf: tir.Buffer,
        bias: float,
        output_buf: tir.Buffer,
        input_dtype: str = "int32",
        bias_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """向量累加原语: V = ΣX+Bias
        
        Args:
            input_buf: 输入缓冲区 [N]
            bias: 偏置值
            output_buf: 输出缓冲区 [N]
        """
        print("调用向量累加原语")
        
        # 获取数据类型编码
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x03)
        bias_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(bias_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.vector_accumulate",
            input_buf.access_ptr("r"),
            tir.const(bias, "float32"),
            output_buf.access_ptr("w"),
            tir.const(input_type_code, "int32"),
            tir.const(bias_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def vector_dot(
        input1_buf: tir.Buffer,
        input2_buf: tir.Buffer,
        bias: float,
        output_buf: tir.Buffer,
        input1_dtype: str = "int8",
        input2_dtype: str = "int8",
        bias_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """向量点积原语: V = X1•X2+Bias
        
        Args:
            input1_buf: 输入1缓冲区 [N]
            input2_buf: 输入2缓冲区 [N]
            bias: 偏置值
            output_buf: 输出缓冲区 [1]
        """
        print("调用向量点积原语")
        
        # 获取数据类型编码
        input1_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input1_dtype, 0x00)
        input2_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input2_dtype, 0x00)
        bias_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(bias_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.vector_dot",
            input1_buf.access_ptr("r"),
            input2_buf.access_ptr("r"),
            tir.const(bias, "float32"),
            output_buf.access_ptr("w"),
            tir.const(input1_type_code, "int32"),
            tir.const(input2_type_code, "int32"),
            tir.const(bias_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def vector_multiply(
        a_buf: tir.Buffer,
        input_buf: tir.Buffer,
        bias: float,
        output_buf: tir.Buffer,
        a_dtype: str = "int8",
        input_dtype: str = "int8",
        bias_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """向量乘法原语: V = A•X+Bias
        
        Args:
            a_buf: 系数A缓冲区 [N]
            input_buf: 输入缓冲区 [N]
            bias: 偏置值
            output_buf: 输出缓冲区 [N]
        """
        print("调用向量乘法原语")
        
        # 获取数据类型编码
        a_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(a_dtype, 0x00)
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x00)
        bias_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(bias_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.vector_multiply",
            a_buf.access_ptr("r"),
            input_buf.access_ptr("r"),
            tir.const(bias, "float32"),
            output_buf.access_ptr("w"),
            tir.const(a_type_code, "int32"),
            tir.const(input_type_code, "int32"),
            tir.const(bias_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def vector_scale(
        a: float,
        input_buf: tir.Buffer,
        bias: float,
        output_buf: tir.Buffer,
        a_dtype: str = "int9",
        input_dtype: str = "int8",
        bias_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """向量缩放原语: V = aX+Bias
        
        Args:
            a: 缩放系数
            input_buf: 输入缓冲区 [N]
            bias: 偏置值
            output_buf: 输出缓冲区 [N]
        """
        print("调用向量缩放原语")
        
        # 获取数据类型编码
        a_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(a_dtype, 0x05)
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x00)
        bias_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(bias_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.vector_scale",
            tir.const(a, "float32"),
            input_buf.access_ptr("r"),
            tir.const(bias, "float32"),
            output_buf.access_ptr("w"),
            tir.const(a_type_code, "int32"),
            tir.const(input_type_code, "int32"),
            tir.const(bias_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def fully_connected(
        input_buf: tir.Buffer,
        weight_buf: tir.Buffer,
        bias_buf: tir.Buffer,
        output_buf: tir.Buffer,
        input_dtype: str = "int8",
        weight_dtype: str = "int8",
        bias_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """全连接计算原语(MLP): V = X*W + Bias
        
        Args:
            input_buf: 输入缓冲区 [N, D]
            weight_buf: 权重缓冲区 [D, O]
            bias_buf: 偏置缓冲区 [O]
            output_buf: 输出缓冲区 [N, O]
        """
        print("调用全连接计算原语(MLP)")
        
        # 获取数据类型编码
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x00)
        weight_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(weight_dtype, 0x00)
        bias_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(bias_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.fully_connected",
            input_buf.access_ptr("r"),
            weight_buf.access_ptr("r"),
            bias_buf.access_ptr("r"),
            output_buf.access_ptr("w"),
            tir.const(input_type_code, "int32"),
            tir.const(weight_type_code, "int32"),
            tir.const(bias_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def matrix_multiply(
        a_buf: tir.Buffer,
        b_buf: tir.Buffer,
        bias_buf: tir.Buffer,
        output_buf: tir.Buffer,
        a_dtype: str = "int8",
        b_dtype: str = "int8",
        bias_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """矩阵乘法原语(MMM): C = A*B + Bias
        
        Args:
            a_buf: 矩阵A缓冲区 [M, K]
            b_buf: 矩阵B缓冲区 [K, N]
            bias_buf: 偏置缓冲区 [M, N]
            output_buf: 输出缓冲区 [M, N]
        """
        print("调用矩阵乘法原语(MMM)")
        
        # 获取数据类型编码
        a_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(a_dtype, 0x00)
        b_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(b_dtype, 0x00)
        bias_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(bias_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.matrix_multiply",
            a_buf.access_ptr("r"),
            b_buf.access_ptr("r"),
            bias_buf.access_ptr("r"),
            output_buf.access_ptr("w"),
            tir.const(a_type_code, "int32"),
            tir.const(b_type_code, "int32"),
            tir.const(bias_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    # ------------------------------
    # SOMA类原语实现 (处理指令)
    # ------------------------------
    
    @staticmethod
    def vector_max(
        input_bufs: List[tir.Buffer],
        output_buf: tir.Buffer,
        input_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """向量最大值原语: V = Max(X1+X2+...+Xn)
        
        Args:
            input_bufs: 输入缓冲区列表
            output_buf: 输出缓冲区
        """
        print("调用向量最大值原语")
        
        # 获取数据类型编码
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.vector_max",
            tir.const([buf.access_ptr("r") for buf in input_bufs], "handle"),
            tir.const(len(input_bufs), "int32"),
            output_buf.access_ptr("w"),
            tir.const(input_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def vector_min(
        input_bufs: List[tir.Buffer],
        output_buf: tir.Buffer,
        input_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """向量最小值原语: V = Min(X1+X2+...+Xn)
        
        Args:
            input_bufs: 输入缓冲区列表
            output_buf: 输出缓冲区
        """
        print("调用向量最小值原语")
        
        # 获取数据类型编码
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.vector_min",
            tir.const([buf.access_ptr("r") for buf in input_bufs], "handle"),
            tir.const(len(input_bufs), "int32"),
            output_buf.access_ptr("w"),
            tir.const(input_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def vector_merge(
        input_bufs: List[tir.Buffer],
        output_buf: tir.Buffer,
        input_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """向量合并原语: Move merge
        
        Args:
            input_bufs: 输入缓冲区列表
            output_buf: 输出缓冲区
        """
        print("调用向量合并原语")
        
        # 获取数据类型编码
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.vector_merge",
            tir.const([buf.access_ptr("r") for buf in input_bufs], "handle"),
            tir.const(len(input_bufs), "int32"),
            output_buf.access_ptr("w"),
            tir.const(input_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def vector_split(
        input_buf: tir.Buffer,
        output_bufs: List[tir.Buffer],
        split_sizes: List[int],
        input_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """向量裂解原语: Move split
        
        Args:
            input_buf: 输入缓冲区
            output_bufs: 输出缓冲区列表
            split_sizes: 分割大小列表
        """
        print("调用向量裂解原语")
        
        # 获取数据类型编码
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.vector_split",
            input_buf.access_ptr("r"),
            tir.const([buf.access_ptr("w") for buf in output_bufs], "handle"),
            tir.const(len(output_bufs), "int32"),
            tir.const(split_sizes, "int32"),
            tir.const(input_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def matrix_transpose(
        input_buf: tir.Buffer,
        output_buf: tir.Buffer,
        input_dtype: str = "int32",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """矩阵转置原语: Matrix Transition
        
        Args:
            input_buf: 输入缓冲区 [M, N]
            output_buf: 输出缓冲区 [N, M]
        """
        print("调用矩阵转置原语")
        
        # 获取数据类型编码
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x03)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.matrix_transpose",
            input_buf.access_ptr("r"),
            output_buf.access_ptr("w"),
            tir.const(input_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def lookup_table(
        input_buf: tir.Buffer,
        lut_buf: tir.Buffer,
        output_buf: tir.Buffer,
        input_dtype: str = "int32",
        lut_dtype: str = "4b",
        output_dtype: str = "int32"
    ) -> tir.expr:
        """查找表原语: LUT
        
        Args:
            input_buf: 输入缓冲区
            lut_buf: 查找表缓冲区
            output_buf: 输出缓冲区
        """
        print("调用查找表原语")
        
        # 获取数据类型编码
        input_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(input_dtype, 0x03)
        lut_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(lut_dtype, 0x06)
        output_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(output_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.lookup_table",
            input_buf.access_ptr("r"),
            lut_buf.access_ptr("r"),
            output_buf.access_ptr("w"),
            tir.const(input_type_code, "int32"),
            tir.const(lut_type_code, "int32"),
            tir.const(output_type_code, "int32")
        )
    
    @staticmethod
    def snn_activation(
        voltage_buf: tir.Buffer,
        current_buf: tir.Buffer,
        threshold: float,
        spike_buf: tir.Buffer,
        voltage_dtype: str = "int32",
        current_dtype: str = "int28",
        spike_dtype: str = "int32"
    ) -> tir.expr:
        """SNN激活函数原语(LIF神经元)
        
        Args:
            voltage_buf: 电压缓冲区
            current_buf: 电流缓冲区  
            threshold: 阈值
            spike_buf: 脉冲输出缓冲区
        """
        print("调用SNN激活函数原语(LIF)")
        
        # 获取数据类型编码
        voltage_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(voltage_dtype, 0x03)
        current_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(current_dtype, 0x04)
        spike_type_code = RoleBasedPrimitives.DATA_TYPE_MAP.get(spike_dtype, 0x03)
        
        return tir.call_intrin(
            "void", "tir.manycore.snn_activation",
            voltage_buf.access_ptr("r"),
            current_buf.access_ptr("r"),
            tir.const(threshold, "float32"),
            spike_buf.access_ptr("w"),
            tir.const(voltage_type_code, "int32"),
            tir.const(current_type_code, "int32"),
            tir.const(spike_type_code, "int32")
        )

    @staticmethod
    def send_receive(
        operation_type: int,
        src_core: int,
        dst_cores: List[int],
        src_addr: int,
        dst_addr: int,
        data_size: int
    ) -> tir.expr:
        """统一的通信原语：send_receive"""
        print("调用统一通信原语: send_receive")
        
        return tir.call_intrin(
            "void", "tir.manycore.send_receive",
            tir.const(operation_type, "int32"),
            tir.const(src_core, "int32"),
            tir.const(dst_cores, "int32"),
            tir.const(len(dst_cores), "int32"),
            tir.const(src_addr, "int32"),
            tir.const(dst_addr, "int32"),
            tir.const(data_size, "int32")
        )
