import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np

from manycore_config import ManyCoreYAMLConfig

class RoleBasedScheduler:
    """基于核心角色的任务调度器，支持所有原语的任务分配"""
    def __init__(self, config: ManyCoreYAMLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.layer_mapping = self._init_layer_mapping()  # 网络层到计算核心的映射
        self.layer_order = []  # 层执行顺序
        self.cache_map = self.map_cache_cores()  # 计算核心到缓存核心的映射
        
    def _init_layer_mapping(self) -> Dict[str, List[int]]:
        """从配置初始化层映射"""
        config_mapping = self.config.get_layer_mapping()
        compute_cores = self.config.get_core_ids_by_role("compute")
        
        # 如果配置中没有层映射，为所有层分配所有计算核心
        if not config_mapping:
            return {"default": compute_cores}
        
        return config_mapping
    
    def assign_layer_to_cores(self, layer_name: str, core_ids: Optional[List[int]] = None) -> None:
        """将网络层分配给指定计算核心"""
        if core_ids is None:
            # 如果未指定核心，使用默认映射
            core_ids = self.layer_mapping.get("default", self.config.get_core_ids_by_role("compute"))
        
        # 检查核心是否已配置为计算角色
        compute_cores = self.config.get_core_ids_by_role("compute")
        for cid in core_ids:
            if cid not in compute_cores:
                self.logger.warning(f"核心 {cid} 未配置为计算角色，但将用于层 {layer_name}")
            
        self.layer_mapping[layer_name] = core_ids
        if layer_name not in self.layer_order:
            self.layer_order.append(layer_name)
            
        self.logger.info(f"层 {layer_name} 分配到核心: {core_ids}")
    
    def split_workload(self, layer_name: str, data_size: int) -> Dict[int, Tuple[int, int]]:
        """将层工作负载分配到指定计算核心"""
        core_ids = self.get_layer_compute_cores(layer_name)
        core_count = len(core_ids)
        
        if core_count == 0:
            logging.warning(f"未为层 {layer_name} 分配计算核心，使用默认核心")
            compute_cores = self.config.get_core_ids_by_role("compute")
            if compute_cores:
                core_ids = compute_cores[:1]  # 至少使用一个核心
                core_count = len(core_ids)
                self.logger.info(f"使用默认核心 {core_ids} 用于层 {layer_name}")
            else:
                raise ValueError(f"未为层 {layer_name} 分配计算核心且无默认核心可用")
                
        # 根据层类型调整分配策略
        layer_type = self._infer_layer_type(layer_name)
        
        # 确保data_size是有效的
        if data_size <= 0:
            data_size = 100  # 默认大小
            logging.warning(f"数据大小无效，使用默认值: {data_size}")
        
        if layer_type in ["conv2d", "conv3d"]:
            # 卷积层按输出通道分配
            return self._split_by_output_channels(layer_name, data_size)
        elif layer_type in ["fully_connected", "dense"]:
            # 全连接层按输出神经元分配
            return self._split_by_output_dim(layer_name, data_size)
        else:
            # 默认均匀分配
            return self._split_uniformly(core_ids, data_size)
    
    def _split_uniformly(self, core_ids: List[int], data_size: int) -> Dict[int, Tuple[int, int]]:
        """均匀分配工作负载 - 修复版本"""
        core_count = len(core_ids)
        
        if data_size <= 0:
            # 当data_size为0或负数时，返回一个默认的工作负载分配
            self.logger.warning(f"data_size ({data_size}) 小于等于0，使用默认分配")
            workload = {cid: (0, 10) for cid in core_ids}  # 每个核心10个元素
            return workload
        
        base_size = data_size // core_count 
        remainder = data_size % core_count
        
        workload = {}
        current = 0
        for i, cid in enumerate(core_ids):
            end = current + base_size + (1 if i < remainder else 0)
            workload[cid] = (current, end)
            current = end
            
        self.logger.debug(f"工作负载分配: 数据大小={data_size}, 核心数={core_count}, 分配={workload}")
        return workload
    
    def _split_by_output_channels(self, layer_name: str, data_size: int) -> Dict[int, Tuple[int, int]]:
        """按输出通道分配卷积层工作负载"""
        core_ids = self.get_layer_compute_cores(layer_name)
        # 假设通道数是核心数的倍数
        channels_per_core = max(1, data_size // len(core_ids))  # 确保至少1个通道
        
        workload = {}
        current = 0
        for cid in core_ids:
            end = current + channels_per_core
            workload[cid] = (current, end)
            current = end
            
        self.logger.debug(f"按通道分配工作负载: 层={layer_name}, 数据大小={data_size}, 分配={workload}")
        return workload
    
    def _split_by_output_dim(self, layer_name: str, data_size: int) -> Dict[int, Tuple[int, int]]:
        """按输出维度分配全连接层工作负载"""
        core_ids = self.get_layer_compute_cores(layer_name)
        # 假设输出维度是核心数的倍数
        dims_per_core = max(1, data_size // len(core_ids))  # 确保至少1个维度
        
        workload = {}
        current = 0
        for cid in core_ids:
            end = current + dims_per_core
            workload[cid] = (current, end)
            current = end
            
        self.logger.debug(f"按维度分配工作负载: 层={layer_name}, 数据大小={data_size}, 分配={workload}")
        return workload
    
    def map_cache_cores(self) -> Dict[int, int]:
        """为每个计算核心分配一个缓存核心（优先邻居）"""
        cache_map = {}
        compute_cores = self.config.get_core_ids_by_role("compute")
        cache_cores = self.config.get_core_ids_by_role("cache")
        
        if not cache_cores:
            logging.warning("未配置缓存核心，将跳过缓存同步")
            return {}
        
        for cid in compute_cores:
            # 优先选择邻居作为缓存核心
            neighbors = self.config.get_core_neighbors(cid)
            for neighbor in neighbors:
                if neighbor in cache_cores:
                    cache_map[cid] = neighbor
                    break
            # 如果没有邻居缓存核心，从缓存核心列表中选第一个
            if cid not in cache_map:
                cache_map[cid] = cache_cores[0]
                
        self.logger.info(f"缓存核心映射: {cache_map}")
        return cache_map
    
    def get_layer_compute_cores(self, layer_name: str) -> List[int]:
        """获取指定层的计算核心"""
        cores = self.layer_mapping.get(layer_name, self.layer_mapping.get("default", []))
        # 确保返回的列表不为空
        if not cores:
            compute_cores = self.config.get_core_ids_by_role("compute")
            if compute_cores:
                cores = compute_cores[:1]  # 至少返回一个计算核心
                self.logger.warning(f"层 {layer_name} 没有分配计算核心，使用默认核心: {cores}")
        return cores
    
    def _infer_layer_type(self, layer_name: str) -> str:
        """从层名称推断层类型"""
        layer_name = layer_name.lower()
        if "conv2d" in layer_name or "conv" in layer_name:
            return "conv2d"
        elif "conv3d" in layer_name:
            return "conv3d"
        elif "fc" in layer_name or "dense" in layer_name or "linear" in layer_name:
            return "fully_connected"
        elif "relu" in layer_name or "activation" in layer_name or "sigmoid" in layer_name or "tanh" in layer_name:
            return "ann_activation"
        elif "pool" in layer_name:
            return "pooling"
        elif "concat" in layer_name:
            return "concat"
        elif "split" in layer_name:
            return "split"
        elif "batch_norm" in layer_name or "batchnorm" in layer_name:
            return "batch_norm"
        elif "lif" in layer_name or "spike" in layer_name:
            return "snn_activation"
        else:
            return "unknown"
    
    def get_layer_index(self, layer_name: str) -> int:
        """获取层在执行顺序中的索引"""
        try:
            return self.layer_order.index(layer_name)
        except ValueError:
            return -1
    
    def get_layer_name(self, index: int) -> Optional[str]:
        """根据索引获取层名称"""
        if 0 <= index < len(self.layer_order):
            return self.layer_order[index]
        return None
    
    def estimate_layer_performance(self, layer_name: str, input_size: int) -> Dict[str, float]:
        """估算层性能指标"""
        core_ids = self.get_layer_compute_cores(layer_name)
        layer_type = self._infer_layer_type(layer_name)
        
        # 基础性能参数
        mac_units = self.config.get_core_spec()["compute_capability"]["mac_units"]
        freq = 1.0  # 1GHz假设频率
        
        # 根据层类型估算计算量
        if layer_type in ["conv2d", "conv3d"]:
            # 卷积层计算量估算: 2 * N * C * H * W * K^2 / S
            compute_ops = 2 * input_size * 9  # 简化估算
        elif layer_type in ["fully_connected", "dense"]:
            # 全连接层计算量估算: 2 * input_size * output_size
            compute_ops = 2 * input_size * (input_size // 2)  # 假设输出是输入的一半
        else:
            # 其他层简化估算
            compute_ops = input_size
        
        # 计算时间 (秒)
        compute_time = compute_ops / (len(core_ids) * mac_units * freq * 1e9)
        
        # 通信时间估算
        comm_time = 0.0
        if len(core_ids) > 1:
            # 假设需要在核心间交换一半的数据
            data_size = input_size * 4  # 4字节/元素
            max_hops = self.config.get_noc_spec()["max_hops"]
            hop_latency = self.config.get_noc_spec()["latency_per_hop"]
            bandwidth = self.config.get_noc_spec()["bandwidth"]
            
            # 通信时间 (秒) = 数据传输时间 + 路由延迟
            transfer_time = (data_size * 0.5) / (bandwidth * 1e9)
            routing_delay = max_hops * hop_latency * 1e-9
            comm_time = transfer_time + routing_delay
        
        return {
            "compute_cores": len(core_ids),
            "compute_ops": compute_ops,
            "compute_time_s": compute_time,
            "comm_time_s": comm_time,
            "total_time_s": compute_time + comm_time
        }
    
    def get_scheduling_summary(self) -> Dict[str, Any]:
        """获取调度摘要信息"""
        summary = {
            "total_layers": len(self.layer_order),
            "layer_mapping": {},
            "cache_mapping": self.cache_map,
            "compute_cores_used": set(),
            "performance_estimates": {}
        }
        
        for layer_name in self.layer_order:
            cores = self.get_layer_compute_cores(layer_name)
            summary["layer_mapping"][layer_name] = cores
            summary["compute_cores_used"].update(cores)
            
            # 估算性能
            input_size = np.prod(self.config.get_input_shape())
            try:
                perf = self.estimate_layer_performance(layer_name, input_size)
                summary["performance_estimates"][layer_name] = perf
            except Exception as e:
                self.logger.warning(f"无法估算层 {layer_name} 的性能: {e}")
        
        summary["compute_cores_used"] = list(summary["compute_cores_used"])
        summary["total_compute_cores_used"] = len(summary["compute_cores_used"])
        
        return summary
    
    def print_scheduling_summary(self) -> None:
        """打印调度摘要"""
        summary = self.get_scheduling_summary()
        
        print("\n=== 任务调度摘要 ===")
        print(f"总层数: {summary['total_layers']}")
        print(f"使用的计算核心数: {summary['total_compute_cores_used']}")
        print(f"使用的计算核心: {sorted(summary['compute_cores_used'])}")
        
        print("\n层到核心的映射:")
        for layer_name, cores in summary["layer_mapping"].items():
            print(f"  {layer_name}: {cores}")
        
        print("\n缓存核心映射:")
        for compute_core, cache_core in summary["cache_mapping"].items():
            print(f"  计算核心{compute_core} -> 缓存核心{cache_core}")
        
        if summary["performance_estimates"]:
            print("\n性能估算:")
            for layer_name, perf in summary["performance_estimates"].items():
                print(f"  {layer_name}:")
                print(f"    计算核心: {perf['compute_cores']}")
                print(f"    计算量: {perf['compute_ops']:.0f} ops")
                print(f"    计算时间: {perf['compute_time_s']*1e6:.2f} μs")
                print(f"    通信时间: {perf['comm_time_s']*1e6:.2f} μs")
                print(f"    总时间: {perf['total_time_s']*1e6:.2f} μs")
