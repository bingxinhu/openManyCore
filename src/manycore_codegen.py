import logging
from typing import List, Dict, Tuple, Any
import numpy as np

from manycore_config import ManyCoreYAMLConfig
from manycore_scheduler import RoleBasedScheduler
from manycore_primitives import RoleBasedPrimitives
from manycore_runtime import EnhancedMemoryRuntime  # 导入运行时类

class EnhancedMemoryCodeGenerator:
    """增强内存架构代码生成器（支持部分核心激活）"""
    
    def __init__(self, config: ManyCoreYAMLConfig, scheduler: RoleBasedScheduler, runtime: EnhancedMemoryRuntime):
        self.config = config
        self.scheduler = scheduler
        self.runtime = runtime  # 存储运行时实例（用于获取激活核心）
        self.core_programs = [[] for _ in range(config.get_total_cores())]  # 每个核心的指令列表
        self.opcode_map = RoleBasedPrimitives.OPCODE_MAP  # 原语操作码映射
        self.memory_allocation = {}  # 内存分配跟踪
        self._init_memory_allocation()  # 初始化内存分配
        
        logging.info("增强内存架构代码生成器初始化完成")

    def _init_memory_allocation(self):
        """初始化每个核心的内存分配地址（MEM0/MEM1分区）"""
        for core_id in range(self.config.get_total_cores()):
            # MEM0分为输入区（前32KB）和输出区（后32KB）
            self.memory_allocation[core_id] = {
                'mem0_input': self.config.get_mem0_spec()["base_addr"],
                'mem0_output': self.config.get_mem0_spec()["base_addr"] + 32768,  # 32KB分界
                'mem1_weights': self.config.get_mem1_spec()["base_addr"],
                'mem3_cache': self.config.get_mem3_spec()["base_addr"]  # MEM3缓存地址
            }

    def generate_enhanced_memory_code(self, layer_name: str, layer_type: str, 
                                    input_data: np.ndarray, weights: Dict[str, np.ndarray]) -> None:
        """生成增强内存架构的完整代码（适配部分核心）"""
        active_cores = sorted(self.runtime.active_cores)
        if not active_cores:
            raise ValueError("未激活任何核心，请先在运行时中激活核心")
        
        logging.info(f"开始为层 {layer_name} 生成代码，激活核心数: {len(active_cores)}")
        
        # 1. 分布式存储输入数据到激活核心的MEM0
        input_chunk_size = self._distribute_input_data(layer_name, input_data, active_cores)
        
        # 2. 分布式存储权重到激活核心的MEM1
        weight_chunk_size = self._distribute_weights(layer_name, weights, active_cores)
        
        # 3. 生成计算指令（存内计算）
        self._generate_computation_instructions(layer_name, layer_type, 
                                               input_chunk_size, weight_chunk_size, active_cores)
        
        # 4. 生成核间通信指令（优化通信拓扑）
        self._generate_intercore_communication_optimized(layer_name, active_cores)
        
        logging.info(f"层 {layer_name} 代码生成完成")

    def _distribute_input_data(self, layer_name: str, input_data: np.ndarray, active_cores: List[int]) -> int:
        """将输入数据分布式加载到激活核心的MEM0输入区（优化版本）"""
        core_count = len(active_cores)
        
        # 优化：直接使用numpy分割
        data_chunks = np.array_split(input_data.flatten(), core_count)
        
        for i, core_id in enumerate(active_cores):
            chunk_array = data_chunks[i]
            chunk_size_bytes = chunk_array.nbytes
            
            # 获取当前核心的MEM0输入区地址
            current_addr = self.memory_allocation[core_id]['mem0_input']
            
            # 生成数据加载指令（操作码：send_receive，源：0x0000=输入设备）
            encoded_instr = self._encode_data_load_instruction(
                core_id, current_addr, chunk_size_bytes
            )
            
            # 添加指令到核心程序
            self.core_programs[core_id].append({
                "encoded": encoded_instr,
                "opcode": self.opcode_map["send_receive"],
                "operands": [0x0000, current_addr, chunk_size_bytes],
                "comment": f"加载{layer_name}输入数据到MEM0（大小：{chunk_size_bytes}B）"
            })
            
            # 更新下一次分配的地址
            self.memory_allocation[core_id]['mem0_input'] += chunk_size_bytes
        
        return data_chunks[0].nbytes if data_chunks else 0

    def _distribute_weights(self, layer_name: str, weights: Dict[str, np.ndarray], active_cores: List[int]) -> int:
        """将权重分布式加载到激活核心的MEM1权重区（优化版本）"""
        core_count = len(active_cores)
        total_weight_size = 0
        
        # 遍历所有权重（支持多权重字典）
        for weight_name, weight in weights.items():
            # 优化：直接使用numpy分割
            weight_chunks = np.array_split(weight, core_count, axis=0)
            
            for i, core_id in enumerate(active_cores):
                chunk_array = weight_chunks[i]
                chunk_size_bytes = chunk_array.nbytes
                total_weight_size += chunk_size_bytes
                
                # 获取当前核心的MEM1权重区地址
                current_addr = self.memory_allocation[core_id]['mem1_weights']
                
                # 生成权重加载指令（操作码：send_receive，源：0x0001=权重设备）
                encoded_instr = self._encode_weight_load_instruction(
                    core_id, current_addr, chunk_size_bytes, weight_name
                )
                
                # 添加指令到核心程序
                self.core_programs[core_id].append({
                    "encoded": encoded_instr,
                    "opcode": self.opcode_map["send_receive"],
                    "operands": [0x0001, current_addr, chunk_size_bytes],
                    "comment": f"加载权重{weight_name}到MEM1（大小：{chunk_size_bytes}B）"
                })
                
                # 更新下一次分配的地址
                self.memory_allocation[core_id]['mem1_weights'] += chunk_size_bytes
        
        # 返回平均每个核心的权重总大小
        return total_weight_size // core_count if core_count > 0 else 0

    def _generate_computation_instructions(self, layer_name: str, layer_type: str, 
                                         input_size: int, weight_size: int, active_cores: List[int]) -> None:
        """生成存内计算指令（基于层类型匹配原语）"""
        # 获取当前层对应的计算原语
        primitive = self._get_primitive_for_layer(layer_type)
        
        for core_id in active_cores:
            # 计算输入/权重/输出的地址（回退到当前块的起始地址）
            input_addr = self.memory_allocation[core_id]['mem0_input'] - input_size
            weight_addr = self.memory_allocation[core_id]['mem1_weights'] - weight_size
            output_addr = self.memory_allocation[core_id]['mem0_output']
            
            # 获取原语对应的操作码（从运行时的内存架构中获取预加载的原语）
            primitive_opcode = self._get_primitive_opcode(core_id, primitive)
            
            if primitive_opcode is not None:
                # 生成计算指令（操作码+地址参数）
                encoded_instr = self._encode_compute_instruction(
                    primitive_opcode, input_addr, weight_addr, output_addr
                )
                
                # 添加指令到核心程序
                self.core_programs[core_id].append({
                    "encoded": encoded_instr,
                    "opcode": primitive_opcode,
                    "operands": [input_addr, weight_addr, output_addr],
                    "comment": f"{layer_name} 存内计算（原语：{primitive}）"
                })
                
                # 更新输出地址（为下一层预留空间）
                self.memory_allocation[core_id]['mem0_output'] += input_size
            
            logging.debug(f"核心{core_id}生成{layer_name}计算指令（原语：{primitive}，操作码：{primitive_opcode}）")

    def _generate_intercore_communication_optimized(self, layer_name: str, active_cores: List[int]) -> None:
        """生成激活核心间的通信指令（优化通信拓扑）"""
        core_count = len(active_cores)
        if core_count <= 1:
            logging.info(f"层 {layer_name} 激活核心数≤1，无需核间通信")
            return
        
        # 优化：根据核心数量选择不同的通信拓扑
        if core_count <= 4:
            # 小规模核心使用全连接通信（减少跳数）
            self._generate_full_connect_communication(active_cores, layer_name)
        else:
            # 大规模核心使用更高效的网格通信
            self._generate_mesh_communication(active_cores, layer_name)
        
        logging.info(f"层 {layer_name} 核间通信指令生成完成（优化拓扑，{core_count}个核心参与）")

    def _generate_full_connect_communication(self, active_cores: List[int], layer_name: str) -> None:
        """全连接通信拓扑（适用于小规模核心）"""
        core_count = len(active_cores)
        
        # 每个核心将结果广播给所有其他核心
        for i in range(core_count):
            src_core = active_cores[i]
            
            # 源核心：将MEM0输出区数据复制到MEM3（核间共享缓存）
            src_output_addr = self.memory_allocation[src_core]['mem0_output'] - 512  # 减少通信数据量
            mem3_addr = self.memory_allocation[src_core]['mem3_cache']
            
            # 生成缓存复制指令
            encoded_instr = self._encode_cache_instruction(
                src_output_addr, mem3_addr, 512
            )
            
            # 源核心添加通信指令
            self.core_programs[src_core].append({
                "encoded": encoded_instr,
                "opcode": self.opcode_map["send_receive"],
                "operands": [src_output_addr, mem3_addr, 512],
                "comment": f"核心{src_core}→所有核心：MEM0→MEM3（大小：512B）"
            })
            
            # 所有其他核心从MEM3读取数据
            for j in range(core_count):
                if i != j:  # 跳过自己
                    dst_core = active_cores[j]
                    dst_input_addr = self.memory_allocation[dst_core]['mem0_input']
                    
                    # 生成数据读取指令
                    encoded_instr = self._encode_data_load_instruction(
                        dst_core, dst_input_addr, 512
                    )
                    
                    # 目标核心添加通信指令
                    self.core_programs[dst_core].append({
                        "encoded": encoded_instr,
                        "opcode": self.opcode_map["send_receive"],
                        "operands": [0x0002, dst_input_addr, 512],  # 0x0002=MEM3缓存
                        "comment": f"核心{dst_core}←核心{src_core}：MEM3→MEM0（大小：512B）"
                    })

    def _generate_mesh_communication(self, active_cores: List[int], layer_name: str) -> None:
        """网格通信拓扑（适用于大规模核心）"""
        core_count = len(active_cores)
        
        # 简单的网格通信：每个核心与邻居通信
        rows, cols = self.config.get_array_shape()
        
        for core_id in active_cores:
            x, y = core_id // cols, core_id % cols
            
            # 与四个方向的邻居通信
            neighbors = []
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                neighbor_id = nx * cols + ny
                if 0 <= nx < rows and 0 <= ny < cols and neighbor_id in active_cores:
                    neighbors.append(neighbor_id)
            
            # 源核心：将数据发送给邻居
            if neighbors:
                src_output_addr = self.memory_allocation[core_id]['mem0_output'] - 256  # 进一步减少通信数据量
                mem3_addr = self.memory_allocation[core_id]['mem3_cache']
                
                # 生成缓存复制指令
                encoded_instr = self._encode_cache_instruction(
                    src_output_addr, mem3_addr, 256
                )
                
                self.core_programs[core_id].append({
                    "encoded": encoded_instr,
                    "opcode": self.opcode_map["send_receive"],
                    "operands": [src_output_addr, mem3_addr, 256],
                    "comment": f"核心{core_id}→邻居：MEM0→MEM3（大小：256B）"
                })
                
                # 邻居核心从MEM3读取数据
                for neighbor_id in neighbors:
                    dst_input_addr = self.memory_allocation[neighbor_id]['mem0_input']
                    
                    encoded_instr = self._encode_data_load_instruction(
                        neighbor_id, dst_input_addr, 256
                    )
                    
                    self.core_programs[neighbor_id].append({
                        "encoded": encoded_instr,
                        "opcode": self.opcode_map["send_receive"],
                        "operands": [0x0002, dst_input_addr, 256],
                        "comment": f"核心{neighbor_id}←核心{core_id}：MEM3→MEM0（大小：256B）"
                    })

    def _get_primitive_for_layer(self, layer_type: str) -> str:
        """根据层类型匹配对应的计算原语"""
        primitive_map = {
            "conv2d": "conv2d",
            "conv3d": "conv3d",
            "relu": "ann_activation",
            "fully_connected": "fully_connected",
            "pool": "vector_accumulate",
            "batch_norm": "vector_accumulate"
        }
        # 默认使用全连接原语
        return primitive_map.get(layer_type, "fully_connected")

    def _get_primitive_opcode(self, core_id: int, primitive_name: str) -> int:
        """从核心的MEM2中获取预加载的原语操作码（模拟硬件预加载）"""
        # 实际硬件中，原语操作码会预加载到MEM2，此处简化为从映射表获取
        opcode = self.opcode_map.get(primitive_name)
        if opcode is None:
            logging.warning(f"原语 {primitive_name} 未找到操作码，使用默认值 0x41（conv2d）")
            return 0x41
        return opcode

    def _encode_data_load_instruction(self, core_id: int, addr: int, size: int) -> List[int]:
        """编码数据加载指令（格式：[操作码, 核心ID高8位, 核心ID低8位, 地址高24位, 地址低8位, 大小高8位, 大小低8位]）"""
        return [
            self.opcode_map["send_receive"],
            (core_id >> 8) & 0xFF,  # 核心ID高字节
            core_id & 0xFF,         # 核心ID低字节
            (addr >> 24) & 0xFF,    # 地址第3字节
            (addr >> 16) & 0xFF,    # 地址第2字节
            (addr >> 8) & 0xFF,     # 地址第1字节
            addr & 0xFF,            # 地址第0字节
            (size >> 8) & 0xFF,     # 大小高字节
            size & 0xFF             # 大小低字节
        ]

    def _encode_weight_load_instruction(self, core_id: int, addr: int, size: int, weight_name: str) -> List[int]:
        """编码权重加载指令（与数据加载指令格式一致，仅注释区分）"""
        return self._encode_data_load_instruction(core_id, addr, size)

    def _encode_compute_instruction(self, opcode: int, input_addr: int, weight_addr: int, output_addr: int) -> List[int]:
        """编码计算指令（格式：[操作码, 输入地址高16位, 输入地址低16位, 权重地址高16位, 权重地址低16位, 输出地址高16位, 输出地址低16位]）"""
        return [
            opcode,
            (input_addr >> 16) & 0xFF,  # 输入地址第3字节
            (input_addr >> 8) & 0xFF,   # 输入地址第2字节
            input_addr & 0xFF,          # 输入地址第1-0字节（16位）
            (weight_addr >> 16) & 0xFF, # 权重地址第3字节
            (weight_addr >> 8) & 0xFF,  # 权重地址第2字节
            weight_addr & 0xFF,         # 权重地址第1-0字节（16位）
            (output_addr >> 16) & 0xFF, # 输出地址第3字节
            (output_addr >> 8) & 0xFF,  # 输出地址第2字节
            output_addr & 0xFF          # 输出地址第1-0字节（16位）
        ]

    def _encode_cache_instruction(self, src_addr: int, dst_addr: int, size: int) -> List[int]:
        """编码缓存复制指令（格式：[操作码, 源地址高16位, 源地址低16位, 目标地址高16位, 目标地址低16位, 大小高8位, 大小低8位]）"""
        return [
            self.opcode_map["send_receive"],
            (src_addr >> 16) & 0xFF,  # 源地址第3字节
            (src_addr >> 8) & 0xFF,   # 源地址第2字节
            src_addr & 0xFF,          # 源地址第1-0字节（16位）
            (dst_addr >> 16) & 0xFF,  # 目标地址第3字节
            (dst_addr >> 8) & 0xFF,   # 目标地址第2字节
            dst_addr & 0xFF,          # 目标地址第1-0字节（16位）
            (size >> 8) & 0xFF,       # 大小高字节
            size & 0xFF               # 大小低字节
        ]

    def generate_binary(self) -> bytearray:
        """将所有核心的指令转换为二进制可执行文件"""
        binary = bytearray()
        
        # 遍历每个核心的程序
        for core_id, program in enumerate(self.core_programs):
            if not program:
                continue  # 跳过空程序（未激活的核心）
            
            # 写入核心ID（2字节，小端序）
            binary.extend(core_id.to_bytes(2, byteorder='little', signed=False))
            
            # 写入指令数量（2字节，小端序）
            instr_count = len(program)
            binary.extend(instr_count.to_bytes(2, byteorder='little', signed=False))
            
            # 写入每条指令（每条指令的字节数由编码长度决定）
            for instr in program:
                instr_bytes = bytes(instr["encoded"])
                # 写入指令长度（1字节）
                binary.append(len(instr_bytes))
                # 写入指令内容
                binary.extend(instr_bytes)
        
        logging.info(f"二进制文件生成完成，总大小：{len(binary)} 字节")
        return binary

    def get_core_program_stats(self) -> Dict[str, int]:
        """获取程序统计信息（总指令数、激活核心数）"""
        total_instructions = 0
        active_cores = 0
        
        for program in self.core_programs:
            if program:
                active_cores += 1
                total_instructions += len(program)
        
        stats = {
            "total_instructions": total_instructions,
            "active_cores": active_cores
        }
        logging.info(f"程序统计：总指令数={total_instructions}，激活核心数={active_cores}")
        return stats

    # 保留向后兼容的通信方法
    def _generate_intercore_communication(self, layer_name: str, active_cores: List[int]) -> None:
        """向后兼容的通信方法"""
        self._generate_intercore_communication_optimized(layer_name, active_cores)
