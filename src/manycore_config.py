import yaml
import logging
from typing import List, Dict, Tuple, Any, Optional
import os

class ManyCoreYAMLConfig:
    """众核架构YAML配置解析器 - 更新内存地址版本"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._validate_config()
        logging.info(f"配置加载完成: {config_path}")
        
        # 字地址到字节地址的转换因子
        self.WORD_SIZE = 4  # 32位架构，1字=4字节
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"配置文件解析错误: {str(e)}")
    
    def _validate_config(self) -> None:
        """验证配置的完整性和有效性"""
        required_sections = ["hardware", "core_roles", "model"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置缺少必要部分: {section}")
    
    # ------------------------------
    # 地址转换方法
    # ------------------------------
    
    def word_to_byte_addr(self, word_addr: int) -> int:
        """将字地址转换为字节地址"""
        return word_addr * self.WORD_SIZE
    
    def byte_to_word_addr(self, byte_addr: int) -> int:
        """将字节地址转换为字地址"""
        return byte_addr // self.WORD_SIZE
    
    def get_word_size(self) -> int:
        """获取字大小（字节）"""
        return self.WORD_SIZE
    
    # ------------------------------
    # 内存架构获取方法（字地址）
    # ------------------------------
    
    def get_memory_architecture(self) -> Dict[str, Any]:
        """获取完整的内存架构配置（字地址）"""
        return self.config["hardware"]["memory_arch"]
    
    def get_mem0_spec(self) -> Dict[str, Any]:
        """获取MEM0规格（字地址）"""
        return self.config["hardware"]["memory_arch"]["mem0"]
    
    def get_mem1_spec(self) -> Dict[str, Any]:
        """获取MEM1规格（字地址）"""
        return self.config["hardware"]["memory_arch"]["mem1"]
    
    def get_mem2_spec(self) -> Dict[str, Any]:
        """获取MEM2规格（字地址）"""
        return self.config["hardware"]["memory_arch"]["mem2"]
    
    def get_mem3_spec(self) -> Dict[str, Any]:
        """获取MEM3规格（字地址）"""
        return self.config["hardware"]["memory_arch"]["mem3"]
    
    def get_mem2_partitions(self) -> Dict[str, Any]:
        """获取MEM2分区信息"""
        return self.config["hardware"]["memory_arch"]["mem2"].get("partitions", {})
    
    def get_pi_b_addr(self) -> int:
        """获取PI_B区域起始地址（字地址）"""
        partitions = self.get_mem2_partitions()
        return partitions.get("pi_b", {}).get("addr", 0x8000)
    
    def get_pi_b_size(self) -> int:
        """获取PI_B区域大小（字）"""
        partitions = self.get_mem2_partitions()
        return partitions.get("pi_b", {}).get("size", 768)
    
    def get_r_lab_addr(self) -> int:
        """获取R_lab区域起始地址（字地址）"""
        partitions = self.get_mem2_partitions()
        return partitions.get("r_lab", {}).get("addr", 0x8300)
    
    def get_r_lab_size(self) -> int:
        """获取R_lab区域大小（字）"""
        partitions = self.get_mem2_partitions()
        return partitions.get("r_lab", {}).get("size", 256)
    
    def get_r_io_addr(self) -> int:
        """获取R_IO区域起始地址（字地址）"""
        partitions = self.get_mem2_partitions()
        return partitions.get("r_io", {}).get("addr", 0x8400)
    
    def get_r_io_size(self) -> int:
        """获取R_IO区域大小（字）"""
        partitions = self.get_mem2_partitions()
        return partitions.get("r_io", {}).get("size", 3072)
    
    # ------------------------------
    # LeNet模型内存布局
    # ------------------------------
    
    def get_lenet_memory_layout(self) -> Dict[str, Any]:
        """获取LeNet模型内存布局"""
        return self.config.get("lenet_memory_layout", {})
    
    def get_lenet_output_addr(self, layer_name: str) -> Optional[Tuple[int, int]]:
        """获取LeNet模型指定层的输出地址范围（字地址）"""
        layout = self.get_lenet_memory_layout()
        mem0_layout = layout.get("mem0", {})
        
        if layer_name == "fc2_output":
            fc2_info = mem0_layout.get("fc2_output", {})
            return tuple(fc2_info.get("addr_range", [0x6000, 0x7FFF]))
        elif layer_name == "fc3_output":
            fc3_info = mem0_layout.get("fc3_output", {})
            return tuple(fc3_info.get("addr_range", [0x2000, 0x23FF]))
        
        return None
    
    # ------------------------------
    # 其他原有方法（保持兼容性）
    # ------------------------------
    
    def get_total_cores(self) -> int:
        """获取总核心数"""
        return self.config["hardware"]["total_cores"]
    
    def get_array_shape(self) -> Tuple[int, int]:
        """获取核心阵列形状"""
        return tuple(self.config["hardware"]["array_shape"])
    
    def get_core_spec(self) -> Dict[str, Any]:
        """获取单核心规格"""
        return self.config["hardware"]["core_spec"]
    
    def get_noc_spec(self) -> Dict[str, Any]:
        """获取片上网络规格"""
        return self.config["hardware"]["noc"]
    
    def get_core_ids_by_role(self, role: str) -> List[int]:
        """根据角色获取核心ID列表"""
        if role not in self.config["core_roles"]:
            return []
            
        cores = self.config["core_roles"][role]
        
        # 处理特殊值-1（表示没有该角色核心）
        if cores == -1:
            return []
        
        # 确保返回列表
        if isinstance(cores, list):
            return cores
        elif isinstance(cores, int):
            return [cores]
        else:
            return []
    
    def get_active_core_count(self) -> int:
        """获取激活核心数量"""
        if "activation" in self.config and "active_core_count" in self.config["activation"]:
            return self.config["activation"]["active_core_count"]
        else:
            compute_cores = self.get_core_ids_by_role("compute")
            return len(compute_cores)
    
    def get_active_core_ids(self) -> List[int]:
        """获取激活核心ID列表"""
        if "activation" in self.config and "active_core_ids" in self.config["activation"]:
            active_ids = self.config["activation"]["active_core_ids"]
            if active_ids:
                return active_ids
        
        compute_cores = self.get_core_ids_by_role("compute")
        active_count = self.get_active_core_count()
        return compute_cores[:active_count]
    
    def get_layer_mapping(self) -> Dict[str, List[int]]:
        """获取层到计算核心的映射"""
        return self.config.get("layer_mapping", {})
    
    def get_onnx_path(self) -> str:
        """获取ONNX模型路径"""
        return self.config["model"]["onnx_path"]
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """获取输入形状"""
        return tuple(self.config["model"]["input_shape"])
    
    def get_input_dtype(self) -> str:
        """获取输入数据类型"""
        return self.config["model"]["input_dtype"]
    
    # ------------------------------
    # 新增方法用于修复pipeline_controller中的调用
    # ------------------------------
    
    def get_all_compute_cores(self) -> List[int]:
        """获取所有计算核心ID列表"""
        return self.get_core_ids_by_role("compute")
    
    def print_config_summary(self) -> None:
        """打印配置摘要"""
        print("\n=== 众核架构配置摘要 ===")
        print(f"总核心数: {self.get_total_cores()}")
        print(f"核心阵列形状: {self.get_array_shape()}")
        print(f"计算核心: {self.get_all_compute_cores()}")
        print(f"激活核心: {self.get_active_core_ids()}")
        print(f"激活核心数: {self.get_active_core_count()}")
        print(f"输入形状: {self.get_input_shape()}")
        print(f"输入数据类型: {self.get_input_dtype()}")
        print("="*30)
    
    def get_core_neighbors(self, core_id: int) -> List[int]:
        """获取指定核心的邻居核心ID列表"""
        rows, cols = self.get_array_shape()
        
        if not (0 <= core_id < rows * cols):
            return []
        
        x, y = core_id // cols, core_id % cols
        neighbors = []
        
        # 检查四个方向的邻居
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                neighbor_id = nx * cols + ny
                neighbors.append(neighbor_id)
        
        return neighbors
