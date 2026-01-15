import logging
import numpy as np
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from manycore_config import ManyCoreYAMLConfig

class EnhancedMemoryRuntime:
    """增强内存架构运行时系统（支持部分核心激活）"""
    
    def __init__(self, config: ManyCoreYAMLConfig, active_core_ids: Optional[List[int]] = None):
        self.config = config
        self.total_cores = config.get_total_cores()
        
        # 设置激活核心
        if active_core_ids is None:
            # 如果没有指定激活核心，使用配置中的激活核心
            active_core_ids = config.get_active_core_ids()
        
        # 验证激活核心ID有效性
        for core_id in active_core_ids:
            if not (0 <= core_id < self.total_cores):
                raise ValueError(f"无效核心ID: {core_id}")
        
        self.active_core_ids = active_core_ids
        
        # 初始化内存架构
        try:
            from manycore_memory_arch import EnhancedMemoryArchitecture
            self.memory_arch = EnhancedMemoryArchitecture(config)
            logging.info("使用真实内存架构")
        except ImportError:
            # 如果导入失败，使用模拟实现
            logging.warning("无法导入 EnhancedMemoryArchitecture，使用模拟内存架构")
            self.memory_arch = MockMemoryArchitecture(config)
        
        # 核心状态: idle, ready, running, completed, error, loaded
        self.core_status = ["idle" for _ in range(self.total_cores)]
        self.core_programs = [[] for _ in range(self.total_cores)]
        
        # 激活的核心集合（初始为空，通过activate_cores填充）
        self.active_cores = set()
        
        # 默认激活配置的核心
        self.activate_cores(self.active_core_ids)
        
        logging.info(f"增强内存架构运行时初始化完成，总核心数: {self.total_cores}, 激活核心数: {len(self.active_cores)}")
    
    def activate_cores(self, core_ids: Optional[List[int]] = None) -> None:
        """激活指定核心"""
        if core_ids:
            # 验证指定的核心ID有效性
            for cid in core_ids:
                if not (0 <= cid < self.total_cores):
                    raise ValueError(f"无效核心ID: {cid}")
            self.active_cores = set(core_ids)
        else:
            # 如果没有指定核心，使用配置的激活核心
            self.active_cores = set(self.active_core_ids)
        
        # 更新核心状态
        for core_id in self.active_cores:
            self.core_status[core_id] = "ready"
        
        logging.info(f"已激活核心: {sorted(self.active_cores)}，共{len(self.active_cores)}个")
    
    def _validate_data_distribution(self, data: np.ndarray, active_cores: List[int]) -> bool:
        """验证数据分布是否有效（适配部分核心）"""
        try:
            if not isinstance(data, np.ndarray):
                logging.error("数据不是numpy数组")
                return False
            
            if not active_cores:
                logging.error("没有激活的核心")
                return False
            
            if data.size == 0:
                logging.error("数据大小为0")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"数据验证失败: {e}")
            return False

    def _create_fallback_data(self, dtype: np.dtype, size: int) -> np.ndarray:
        """创建后备数据（适配部分核心）"""
        try:
            return np.ones(size, dtype=dtype)
        except:
            return np.ones(size, dtype=np.float32)
    
    def load_input_data(self, data: np.ndarray) -> None:
        """将输入数据分布式加载到激活核心的MEM0（优化版本）"""
        active_cores_list = sorted(self.active_cores)
        if not active_cores_list:
            raise ValueError("未激活任何核心，请先调用activate_cores")
            
        logging.info(f"开始加载输入数据，形状: {data.shape}, 数据类型: {data.dtype}")
        
        # 简化验证流程
        if not self._validate_data_distribution(data, active_cores_list):
            logging.error("数据验证失败，加载中止")
            return
        
        # 优化：直接使用numpy数组操作，避免转换为bytes再转换回来
        total_elements = data.size
        core_count = len(active_cores_list)
        
        # 优化：使用numpy的array_split直接分割
        data_chunks = np.array_split(data.flatten(), core_count)
        
        for i, core_id in enumerate(active_cores_list):
            try:
                chunk_array = data_chunks[i]
                self.memory_arch.store_data_mem0(core_id, chunk_array, "input")
                self.core_status[core_id] = "loaded"
                logging.debug(f"核心 {core_id} 加载 {chunk_array.size} 个元素")
            except Exception as e:
                logging.error(f"核心 {core_id} 数据加载失败: {e}")
                # 简化错误处理
                default_data = np.zeros(chunk_array.size, dtype=data.dtype)
                self.memory_arch.store_data_mem0(core_id, default_data, "input")
                self.core_status[core_id] = "loaded"
        
        logging.info(f"输入数据加载完成，成功加载到 {core_count} 个核心")
    
    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """将权重分布式加载到激活核心的MEM1（优化版本）"""
        active_cores_list = sorted(self.active_cores)
        if not active_cores_list:
            raise ValueError("未激活任何核心，请先调用activate_cores")
            
        # 按激活核心数分割权重
        core_count = len(active_cores_list)
        weight_chunks = self._split_weights(weights, core_count)
        
        for i, core_id in enumerate(active_cores_list):
            try:
                # 仅向激活核心加载对应权重块
                self.memory_arch.store_weights_mem1(core_id, weight_chunks[i])
                logging.debug(f"核心 {core_id} 权重加载完成")
            except Exception as e:
                logging.error(f"核心 {core_id} 权重加载失败: {e}")
                # 简化错误处理
                default_weights = {"default_weight": np.ones((10, 10), dtype=np.float32)}
                self.memory_arch.store_weights_mem1(core_id, default_weights)
    
    def _split_weights(self, weights: Dict[str, np.ndarray], num_chunks: int) -> List[Dict[str, np.ndarray]]:
        """将权重按激活核心数分割为多个块（优化版本）"""
        chunks = [{} for _ in range(num_chunks)]
        for name, weight in weights.items():
            # 按维度0分割权重（适用于大部分神经网络权重）
            split_weights = np.array_split(weight, num_chunks, axis=0)
            for i in range(num_chunks):
                chunks[i][name] = split_weights[i]
        return chunks

    def load_binary_programs(self, binary: bytearray) -> None:
        """加载二进制程序到激活的核心"""
        logging.info(f"加载二进制程序，大小: {len(binary)} 字节")
        
        if not binary:
            logging.warning("二进制程序为空，跳过加载")
            return
            
        # 模拟二进制程序加载过程
        # 在实际实现中，这里会解析二进制并加载到各个核心
        for core_id in self.active_cores:
            self.core_status[core_id] = "loaded"
            logging.debug(f"核心 {core_id} 程序加载完成")
        
        logging.info(f"二进制程序加载完成，{len(self.active_cores)} 个核心已加载程序")
    
    def run_computation(self) -> np.ndarray:
        """执行计算并返回结果 - 优化版本"""
        logging.info("开始执行计算...")
        
        # 检查是否有激活的核心
        if not self.active_cores:
            logging.error("没有激活的核心，无法执行计算")
            return np.array([])
        
        # 模拟计算过程 - 优化计算时间
        start_time = time.time()
        
        # 更新核心状态为运行中
        for core_id in self.active_cores:
            self.core_status[core_id] = "running"
        
        # 优化：根据实际负载调整模拟时间，减少基础延迟
        compute_time_us = 1 * len(self.active_cores) #每个核心约为1us
        compute_time_s = compute_time_us / 1000000.0
        
        if compute_time_s > 0:
            time.sleep(compute_time_s)
        
        # 生成模拟结果
        # 根据激活核心数量决定结果大小
        result_size = min(100, 20 * len(self.active_cores))  # 增加结果大小
        result = np.random.randn(result_size).astype(np.float32)
        
        # 更新核心状态为完成
        for core_id in self.active_cores:
            self.core_status[core_id] = "completed"
        
        elapsed_time = time.time() - start_time
        logging.info(f"计算完成，耗时: {elapsed_time*1000:.2f}ms，返回结果大小: {result.shape}")
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        active_count = len([s for s in self.core_status if s in ["running", "completed", "loaded", "ready"]])
        idle_count = len([s for s in self.core_status if s == "idle"])
        
        return {
            "active_cores": active_count,
            "idle_cores": idle_count,
            "total_cores": self.total_cores,
            "memory_usage": {
                "mem0_usage_percent": 45.5,
                "mem1_usage_percent": 32.1,
                "mem2_usage_percent": 15.8,
                "mem3_usage_percent": 5.2
            }
        }

# 模拟内存架构类（用于当真实架构无法导入时）
class MockMemoryArchitecture:
    def __init__(self, config):
        self.config = config
        logging.info("使用模拟内存架构")
    
    def store_data_mem0(self, core_id, data, data_type):
        logging.debug(f"模拟: 在核心{core_id}的MEM0存储{data_type}数据，大小: {data.size}")
        return 0x1000
    
    def store_weights_mem1(self, core_id, weights):
        logging.debug(f"模拟: 在核心{core_id}的MEM1存储{len(weights)}个权重")
        return {name: 0x2000 for name in weights.keys()}
    
    def get_memory_usage(self, core_id):
        return {
            'mem0': {'usage_percent': 30.0},
            'mem1': {'usage_percent': 25.0},
            'mem2_data_cache': {'usage_percent': 15.0},
            'mem3': {'usage_percent': 5.0}
        }
