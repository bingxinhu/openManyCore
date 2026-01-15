# manycore_memory_arch_updated.py
import logging
import numpy as np
import struct
from typing import List, Dict, Any, Optional, Tuple
from manycore_config_updated import ManyCoreYAMLConfig

class EnhancedMemoryArchitecture:
    """增强内存架构管理类 - 支持字地址操作"""
    
    def __init__(self, config: ManyCoreYAMLConfig):
        self.config = config
        self.total_cores = config.get_total_cores()
        self.word_size = config.get_word_size()
        
        # 获取内存架构规格（字地址）
        self.memory_arch = config.get_memory_architecture()
        self.mem0_spec = config.get_mem0_spec()
        self.mem1_spec = config.get_mem1_spec()
        self.mem2_spec = config.get_mem2_spec()
        self.mem3_spec = config.get_mem3_spec()
        
        # 初始化每个核心的SRAM（字节数组）
        self.sram_data = {}
        self._init_core_memory()
        
        # 内存分配跟踪（字地址）
        self.memory_allocation = {}
        self._init_memory_allocation()
        
        # 数据分布信息
        self.data_distribution = {}
        
        # 初始化PI_B区域（指令存储）
        self._init_pi_b_region()
        
        # 初始化R_lab区域（标签）
        self._init_r_lab_region()
        
        logging.info(f"增强内存架构初始化完成（字地址模式）")
        logging.info(f"字大小: {self.word_size}字节")
        logging.info(f"内存架构: MEM0({self.mem0_spec['size']}字), MEM1({self.mem1_spec['size']}字), "
                    f"MEM2({self.mem2_spec['size']}字), MEM3({self.mem3_spec['size']}字)")
    
    def _init_core_memory(self):
        """初始化每个核心的内存（字节数组）"""
        for core_id in range(self.total_cores):
            # 计算每个内存区域的字节大小
            mem0_bytes = self.mem0_spec["size"] * self.word_size
            mem1_bytes = self.mem1_spec["size"] * self.word_size
            mem2_bytes = self.mem2_spec["size"] * self.word_size
            mem3_bytes = self.mem3_spec["size"] * self.word_size
            
            # 为每个核心创建独立的内存空间
            core_memory = {}
            
            # MEM0: 数据存储区域
            core_memory["mem0"] = bytearray(mem0_bytes)
            
            # MEM1: 权重存储区域  
            core_memory["mem1"] = bytearray(mem1_bytes)
            
            # MEM2: 取指和结果输出区域
            core_memory["mem2"] = bytearray(mem2_bytes)
            
            # MEM3: 核间传输缓存
            core_memory["mem3"] = bytearray(mem3_bytes)
            
            self.sram_data[core_id] = core_memory
    
    def _init_memory_allocation(self):
        """初始化内存分配跟踪（字地址）"""
        for core_id in range(self.total_cores):
            self.memory_allocation[core_id] = {
                'mem0_current': self.mem0_spec["base_addr"],
                'mem1_current': self.mem1_spec["base_addr"],
                'mem2_pi_b_current': self.config.get_pi_b_addr(),
                'mem2_r_lab_current': self.config.get_r_lab_addr(),
                'mem2_r_io_current': self.config.get_r_io_addr(),
                'mem3_current': self.mem3_spec["base_addr"]
            }
    
    def _init_pi_b_region(self):
        """初始化PI_B区域（指令存储）"""
        pi_b_addr = self.config.get_pi_b_addr()
        pi_b_size = self.config.get_pi_b_size()
        
        # 创建简单的指令模板
        instruction_template = bytearray()
        
        # 添加一些示例指令
        for core_id in range(self.total_cores):
            # 添加NOP指令
            instruction_template.extend(b'\x00' * 4)  # 32位指令
            
            # 添加加载指令示例
            instruction_template.extend(b'\x01' * 4)  # 加载操作码
            
            # 添加计算指令示例
            instruction_template.extend(b'\x02' * 4)  # 计算操作码
        
        # 填充到PI_B区域大小
        if len(instruction_template) < pi_b_size * self.word_size:
            instruction_template.extend(b'\x00' * (pi_b_size * self.word_size - len(instruction_template)))
        elif len(instruction_template) > pi_b_size * self.word_size:
            instruction_template = instruction_template[:pi_b_size * self.word_size]
        
        # 加载到每个核心的MEM2 PI_B区域
        for core_id in range(self.total_cores):
            byte_addr = self.word_to_byte_addr(pi_b_addr)
            self.sram_data[core_id]["mem2"][byte_addr:byte_addr+pi_b_size*self.word_size] = instruction_template
        
        logging.info(f"PI_B区域初始化完成，大小: {pi_b_size}字")
    
    def _init_r_lab_region(self):
        """初始化R_lab区域（标签）"""
        r_lab_addr = self.config.get_r_lab_addr()
        r_lab_size = self.config.get_r_lab_size()
        
        # 创建标签数据
        label_data = bytearray()
        
        # 添加一些示例标签
        for i in range(min(r_lab_size, 256)):
            # 标签格式：[标签ID(2字节), 地址(2字节)]
            label_data.extend(i.to_bytes(2, 'little'))  # 标签ID
            label_data.extend((0x2000 + i * 4).to_bytes(2, 'little'))  # 目标地址
        
        # 填充到R_lab区域大小
        if len(label_data) < r_lab_size * self.word_size:
            label_data.extend(b'\x00' * (r_lab_size * self.word_size - len(label_data)))
        
        # 加载到每个核心的MEM2 R_lab区域
        for core_id in range(self.total_cores):
            byte_addr = self.word_to_byte_addr(r_lab_addr)
            self.sram_data[core_id]["mem2"][byte_addr:byte_addr+r_lab_size*self.word_size] = label_data
        
        logging.info(f"R_lab区域初始化完成，大小: {r_lab_size}字")
    
    def word_to_byte_addr(self, word_addr: int) -> int:
        """字地址转字节地址"""
        return word_addr * self.word_size
    
    def byte_to_word_addr(self, byte_addr: int) -> int:
        """字节地址转字地址"""
        return byte_addr // self.word_size
    
    def store_data_mem0_word(self, core_id: int, data: np.ndarray, data_type: str = "input") -> int:
        """在MEM0中存储数据（字地址操作）"""
        if core_id not in self.sram_data:
            raise ValueError(f"无效的核心ID: {core_id}")
        
        data_bytes = data.tobytes()
        data_size_bytes = len(data_bytes)
        
        # 转换为字大小
        data_size_words = (data_size_bytes + self.word_size - 1) // self.word_size
        
        current_word_addr = self.memory_allocation[core_id]['mem0_current']
        current_byte_addr = self.word_to_byte_addr(current_word_addr)
        
        # 检查空间（字地址）
        if current_word_addr + data_size_words > self.mem0_spec["base_addr"] + self.mem0_spec["size"]:
            raise MemoryError(f"核心{core_id}的MEM0空间不足")
        
        # 存储数据
        self.sram_data[core_id]["mem0"][current_byte_addr:current_byte_addr+data_size_bytes] = data_bytes
        
        # 更新分配指针（字地址）
        self.memory_allocation[core_id]['mem0_current'] += data_size_words
        
        # 记录数据分布
        if data_type not in self.data_distribution:
            self.data_distribution[data_type] = {}
        self.data_distribution[data_type][core_id] = (current_word_addr, data_size_words)
        
        logging.info(f"数据存储到核心{core_id}的MEM0，字地址: 0x{current_word_addr:x}，大小: {data_size_words}字")
        return current_word_addr
    
    def store_weights_mem1_word(self, core_id: int, weights: Dict[str, np.ndarray]) -> Dict[str, int]:
        """在MEM1中存储权重（字地址操作）"""
        if core_id not in self.sram_data:
            raise ValueError(f"无效的核心ID: {core_id}")
        
        weight_locations = {}
        current_word_addr = self.memory_allocation[core_id]['mem1_current']
        
        for weight_name, weight in weights.items():
            weight_bytes = weight.tobytes()
            weight_size_bytes = len(weight_bytes)
            weight_size_words = (weight_size_bytes + self.word_size - 1) // self.word_size
            
            # 检查空间（字地址）
            if current_word_addr + weight_size_words > self.mem1_spec["base_addr"] + self.mem1_spec["size"]:
                raise MemoryError(f"核心{core_id}的MEM1空间不足，无法存储权重{weight_name}")
            
            # 存储权重
            current_byte_addr = self.word_to_byte_addr(current_word_addr)
            self.sram_data[core_id]["mem1"][current_byte_addr:current_byte_addr+weight_size_bytes] = weight_bytes
            weight_locations[weight_name] = (current_word_addr, weight_size_words)
            
            # 更新分配指针（字地址）
            current_word_addr += weight_size_words
            
            logging.info(f"权重{weight_name}存储到核心{core_id}的MEM1，字地址: 0x{weight_locations[weight_name][0]:x}，大小: {weight_size_words}字")
        
        self.memory_allocation[core_id]['mem1_current'] = current_word_addr
        return weight_locations
    
    def allocate_r_io_memory(self, core_id: int, size_words: int) -> int:
        """在R_IO区域分配内存（字地址）"""
        current_word_addr = self.memory_allocation[core_id]['mem2_r_io_current']
        
        # 检查空间
        r_io_size = self.config.get_r_io_size()
        if current_word_addr + size_words > self.config.get_r_io_addr() + r_io_size:
            raise MemoryError(f"核心{core_id}的R_IO区域空间不足")
        
        # 更新分配指针
        self.memory_allocation[core_id]['mem2_r_io_current'] += size_words
        
        logging.info(f"在核心{core_id}的R_IO区域分配空间，字地址: 0x{current_word_addr:x}，大小: {size_words}字")
        return current_word_addr
    
    def load_instruction_pi_b(self, core_id: int, instruction_bytes: bytes) -> int:
        """加载指令到PI_B区域"""
        pi_b_addr = self.config.get_pi_b_addr()
        pi_b_size = self.config.get_pi_b_size()
        
        if len(instruction_bytes) > pi_b_size * self.word_size:
            raise ValueError(f"指令大小超出PI_B区域容量")
        
        # 存储指令
        byte_addr = self.word_to_byte_addr(pi_b_addr)
        self.sram_data[core_id]["mem2"][byte_addr:byte_addr+len(instruction_bytes)] = instruction_bytes
        
        logging.info(f"指令加载到核心{core_id}的PI_B区域，字地址: 0x{pi_b_addr:x}，大小: {len(instruction_bytes)}字节")
        return pi_b_addr
    
    def cache_intercore_data(self, src_core: int, dst_core: int, data: np.ndarray) -> bool:
        """使用MEM3缓存核间传输数据（字地址操作）"""
        if src_core not in self.sram_data or dst_core not in self.sram_data:
            return False
        
        data_bytes = data.tobytes()
        data_size_bytes = len(data_bytes)
        data_size_words = (data_size_bytes + self.word_size - 1) // self.word_size
        
        # 检查MEM3大小（字）
        if data_size_words > self.mem3_spec["size"]:
            logging.warning(f"数据大小{data_size_words}字超过MEM3缓存容量{self.mem3_spec['size']}字")
            return False
        
        # 存储到源核心的MEM3
        mem3_word_addr = self.mem3_spec["base_addr"]
        mem3_byte_addr = self.word_to_byte_addr(mem3_word_addr)
        self.sram_data[src_core]["mem3"][mem3_byte_addr:mem3_byte_addr+data_size_bytes] = data_bytes
        
        # 标记缓存数据
        cache_key = f"{src_core}_to_{dst_core}"
        if "intercore_cache" not in self.data_distribution:
            self.data_distribution["intercore_cache"] = {}
        self.data_distribution["intercore_cache"][cache_key] = (src_core, mem3_word_addr, data_size_words)
        
        logging.info(f"核间传输数据缓存到核心{src_core}的MEM3，目标核心: {dst_core}，字地址: 0x{mem3_word_addr:x}，大小: {data_size_words}字")
        return True
    
    def get_memory_usage_word(self, core_id: int) -> Dict[str, Any]:
        """获取指定核心的内存使用情况（字地址）"""
        if core_id not in self.memory_allocation:
            raise ValueError(f"无效的核心ID: {core_id}")
        
        alloc = self.memory_allocation[core_id]
        
        # MEM0使用情况
        mem0_used_words = alloc['mem0_current'] - self.mem0_spec["base_addr"]
        mem0_total_words = self.mem0_spec["size"]
        
        # MEM1使用情况
        mem1_used_words = alloc['mem1_current'] - self.mem1_spec["base_addr"]
        mem1_total_words = self.mem1_spec["size"]
        
        # MEM2 R_IO使用情况
        r_io_used_words = alloc['mem2_r_io_current'] - self.config.get_r_io_addr()
        r_io_total_words = self.config.get_r_io_size()
        
        return {
            'mem0': {
                'total_words': mem0_total_words,
                'total_bytes': mem0_total_words * self.word_size,
                'used_words': mem0_used_words,
                'used_bytes': mem0_used_words * self.word_size,
                'free_words': mem0_total_words - mem0_used_words,
                'free_bytes': (mem0_total_words - mem0_used_words) * self.word_size,
                'usage_percent': (mem0_used_words / mem0_total_words) * 100 if mem0_total_words > 0 else 0
            },
            'mem1': {
                'total_words': mem1_total_words,
                'total_bytes': mem1_total_words * self.word_size,
                'used_words': mem1_used_words,
                'used_bytes': mem1_used_words * self.word_size,
                'free_words': mem1_total_words - mem1_used_words,
                'free_bytes': (mem1_total_words - mem1_used_words) * self.word_size,
                'usage_percent': (mem1_used_words / mem1_total_words) * 100 if mem1_total_words > 0 else 0
            },
            'mem2_r_io': {
                'total_words': r_io_total_words,
                'total_bytes': r_io_total_words * self.word_size,
                'used_words': r_io_used_words,
                'used_bytes': r_io_used_words * self.word_size,
                'free_words': r_io_total_words - r_io_used_words,
                'free_bytes': (r_io_total_words - r_io_used_words) * self.word_size,
                'usage_percent': (r_io_used_words / r_io_total_words) * 100 if r_io_total_words > 0 else 0
            },
            'mem3': {
                'total_words': self.mem3_spec["size"],
                'total_bytes': self.mem3_spec["size"] * self.word_size,
                'used_words': 0,  # 动态使用，不跟踪
                'used_bytes': 0,
                'free_words': self.mem3_spec["size"],
                'free_bytes': self.mem3_spec["size"] * self.word_size,
                'usage_percent': 0
            }
        }
