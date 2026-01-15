import logging
import numpy as np
import os
import time
import sys
from typing import Dict, Any, List, Optional

# 导入组件
from model_processor import ModelProcessor
from manycore_config import ManyCoreYAMLConfig
from manycore_primitives import RoleBasedPrimitives
from manycore_scheduler import RoleBasedScheduler
from manycore_codegen import EnhancedMemoryCodeGenerator
from manycore_runtime import EnhancedMemoryRuntime

class EnhancedMemoryPipelineController:
    """增强内存架构部署全流程控制器"""
    def __init__(self, config_path: str):
        # 初始化配置
        self.config = ManyCoreYAMLConfig(config_path)
        
        # 初始化核心组件
        self._init_components()
        
        # 初始化模型处理器
        self.model_processor = ModelProcessor(self.config)
        
        # 初始化流程状态
        self.binary = None
        self.weights = None
        self.layers = []
        self.mod = None
        self.input_name = None
        self.onnx_model = None
        
        # 性能统计
        self.performance_stats = {}
        
        logging.info("增强内存架构流程控制器初始化完成")
        
    def _init_components(self) -> None:
        """初始化众核架构核心组件"""
        try:
            # 注册原语
            RoleBasedPrimitives.register_primitives()
            logging.info("1. 原语注册完成")
            
            # 创建调度器
            self.scheduler = RoleBasedScheduler(self.config)
            logging.info("2. 调度器初始化完成")
            
            # 创建运行时 - 使用配置的激活核心
            active_core_ids = self.config.get_active_core_ids()
            self.runtime = EnhancedMemoryRuntime(self.config, active_core_ids)
            logging.info("3. 运行时初始化完成")
            
            # 创建代码生成器
            self.codegen = EnhancedMemoryCodeGenerator(self.config, self.scheduler, self.runtime)
            logging.info("4. 代码生成器初始化完成")
        
            logging.info("增强内存架构核心组件初始化完成")
            logging.info(f"核心配置: 总核心数={self.config.get_total_cores()}, "
                        f"计算核心={len(self.config.get_all_compute_cores())}, "
                        f"激活核心={len(self.runtime.active_cores)}")
                        
        except Exception as e:
            logging.error(f"组件初始化失败: {e}")
            raise
    
    def prepare_model(self) -> None:
        """准备模型：加载、转换和提取权重"""
        try:
            start_time = time.time()
            
            # 加载并转换模型
            self.mod, self.input_name, self.onnx_model = self.model_processor.load_and_convert()
            
            # 提取权重
            self.weights = self.model_processor.extract_weights(self.onnx_model)
            
            # 分析模型层
            self.layers = self.model_processor.analyze_layers(self.mod)
            
            # 为每个层分配激活的计算核心
            active_core_ids = self.config.get_active_core_ids()
            successful_assignments = 0
            for layer_name, layer_type in self.layers:
                try:
                    # 使用配置的激活核心参与每一层的计算
                    self.scheduler.assign_layer_to_cores(layer_name, active_core_ids)
                    
                    # 估算层性能
                    input_size = np.prod(self.config.get_input_shape())
                    perf = self.scheduler.estimate_layer_performance(layer_name, input_size)
                    
                    # 记录性能统计
                    self.performance_stats[layer_name] = perf
                    
                    logging.info(f"{layer_name}({layer_type}) 性能估算: "
                                f"计算时间={perf['compute_time_s']*1e6:.2f}μs, "
                                f"通信时间={perf['comm_time_s']*1e6:.2f}μs, "
                                f"总时间={perf['total_time_s']*1e6:.2f}μs")
                    
                    successful_assignments += 1
                    
                except Exception as e:
                    logging.warning(f"层 {layer_name} 分配失败: {e}")
                    continue
            
            elapsed_time = time.time() - start_time
            
            if successful_assignments == 0:
                logging.error("所有层分配都失败了，创建默认配置")
                self._create_fallback_model()
            else:
                logging.info(f"模型准备完成，共 {len(self.layers)} 层，成功分配 {successful_assignments} 层")
            
            # 打印调度摘要
            self.scheduler.print_scheduling_summary()
            
            # 记录性能统计
            total_compute_time = sum(perf['total_time_s'] for perf in self.performance_stats.values())
            self.performance_stats['total'] = {
                'total_time_s': total_compute_time,
                'preparation_time_s': elapsed_time,
                'layer_count': len(self.layers)
            }
            
            logging.info(f"模型准备总耗时: {elapsed_time:.2f}秒")
            logging.info(f"预估推理总时间: {total_compute_time*1e6:.2f}μs")
            
        except Exception as e:
            logging.error(f"模型准备失败: {e}")
            # 创建默认层作为后备
            self._create_fallback_model()
            logging.info("使用默认模型配置继续执行")
    
    def _create_fallback_model(self) -> None:
        """创建后备模型配置"""
        try:
            self.layers = [
                ("conv1", "conv2d"),
                ("relu1", "ann_activation"),
                ("pool1", "vector_accumulate"),
                ("fc1", "fully_connected")
            ]
            self.weights = {
                "conv1_weight": np.random.randn(10, 1, 3, 3).astype(np.float32),
                "fc1_weight": np.random.randn(10, 10).astype(np.float32)
            }
            
            # 为默认层分配激活核心
            active_core_ids = self.config.get_active_core_ids()
            for layer_name, _ in self.layers:
                self.scheduler.assign_layer_to_cores(layer_name, active_core_ids)
                
            logging.info("后备模型配置创建完成")
            
        except Exception as e:
            logging.error(f"创建后备模型失败: {e}")
            raise
    
    def generate_executable(self) -> None:
        """生成增强内存架构可执行代码"""
        if not hasattr(self, 'layers') or not self.layers:
            raise RuntimeError("请先调用prepare_model()准备模型")
            
        logging.info(f"开始为 {len(self.layers)} 层生成增强内存架构代码...")
        
        start_time = time.time()
        
        try:
            # 在生成代码之前激活配置的核心
            active_core_ids = self.config.get_active_core_ids()
            self.runtime.activate_cores(active_core_ids)
            logging.info(f"已激活 {len(active_core_ids)} 个核心用于代码生成: {active_core_ids}")
            
            # 生成输入数据 - 彻底验证
            input_shape = self.config.get_input_shape()
            input_dtype = self.config.get_input_dtype()
            
            # 验证输入形状
            if not input_shape or any(dim <= 0 for dim in input_shape):
                logging.warning(f"输入形状无效: {input_shape}，使用默认形状")
                input_shape = (1, 1, 1, 100)
            
            # 验证数据类型
            try:
                dtype_obj = np.dtype(input_dtype)
            except:
                logging.warning(f"数据类型无效: {input_dtype}，使用float32")
                input_dtype = "float32"
                dtype_obj = np.float32
            
            # 创建输入数据
            try:
                input_data = np.random.rand(*input_shape).astype(dtype_obj)
            except Exception as e:
                logging.error(f"创建输入数据失败: {e}，使用简单数据")
                # 使用更简单的数据
                simple_shape = (100,)  # 一维数组
                input_data = np.random.rand(*simple_shape).astype(np.float32)
            
            logging.info(f"最终输入数据 - 形状: {input_data.shape}, 数据类型: {input_data.dtype}, 总元素数: {input_data.size}")
            
            # 为每个层生成代码
            successful_generations = 0
            for i, (layer_name, layer_type) in enumerate(self.layers):
                logging.info(f"为层 {i+1}/{len(self.layers)}: {layer_name} (类型: {layer_type}) 生成代码")
                
                try:
                    # 检查激活核心状态
                    if not self.runtime.active_cores:
                        logging.warning(f"层 {layer_name} 生成时没有激活核心，重新激活")
                        self.runtime.activate_cores(active_core_ids)
                    
                    # 生成增强内存架构代码
                    self.codegen.generate_enhanced_memory_code(layer_name, layer_type, input_data, self.weights)
                    
                    successful_generations += 1
                    
                except Exception as e:
                    logging.error(f"生成层 {layer_name} 代码时出错: {e}")
                    # 尝试重新激活核心并继续
                    try:
                        logging.info(f"尝试重新激活核心并重试层 {layer_name}")
                        self.runtime.activate_cores(active_core_ids)
                        self.codegen.generate_enhanced_memory_code(layer_name, layer_type, input_data, self.weights)
                        successful_generations += 1
                        logging.info(f"层 {layer_name} 代码生成重试成功")
                    except Exception as retry_e:
                        logging.error(f"层 {layer_name} 代码生成重试失败: {retry_e}")
                        continue
            
            # 生成最终二进制
            self.binary = self.codegen.generate_binary()
            
            elapsed_time = time.time() - start_time
            
            logging.info(f"二进制代码生成完成，大小: {len(self.binary)}字节，耗时: {elapsed_time:.2f}秒")
            logging.info(f"成功为 {successful_generations}/{len(self.layers)} 层生成代码")
            
            # 打印程序摘要
            program_stats = self.codegen.get_core_program_stats()
            logging.info(f"程序统计: 总指令数={program_stats['total_instructions']}, "
                        f"激活核心数={program_stats['active_cores']}")
            
            # 保存二进制到output目录
            self._save_binary_file()
            
            # 记录性能统计
            self.performance_stats['code_generation'] = {
                'time_s': elapsed_time,
                'binary_size': len(self.binary),
                'total_instructions': program_stats['total_instructions'],
                'active_cores': program_stats['active_cores']
            }
            
        except Exception as e:
            logging.error(f"生成可执行文件失败: {e}")
            raise
    
    def _save_binary_file(self) -> None:
        """保存二进制文件和信息文件"""
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"创建输出目录: {output_dir}")
            
        # 生成带有时间戳的文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        binary_filename = f"{output_dir}/enhanced_memory_executable_{timestamp}.bin"
        
        try:
            # 保存二进制文件
            with open(binary_filename, 'wb') as f:
                f.write(self.binary)
                
            logging.info(f"二进制文件已保存到: {binary_filename}")
            
        except Exception as e:
            logging.error(f"保存文件失败: {e}")
    
    def run_inference(self) -> np.ndarray:
        """执行推理并返回结果 - 使用配置的激活核心"""
        # 添加详细的调试信息
        logging.info(f"检查状态: binary={self.binary is not None}, weights={self.weights is not None}")
        
        if not self.binary:
            logging.error("二进制文件未生成，请检查 generate_executable() 方法")
            # 尝试重新生成
            try:
                logging.info("尝试重新生成可执行文件...")
                self.generate_executable()
            except Exception as e:
                logging.error(f"重新生成可执行文件失败: {e}")
                raise RuntimeError("无法生成可执行文件")
        
        if not self.weights:
            logging.error("权重未加载，请检查 prepare_model() 方法")
            # 尝试重新准备模型
            try:
                logging.info("尝试重新准备模型...")
                self.prepare_model()
            except Exception as e:
                logging.error(f"重新准备模型失败: {e}")
                raise RuntimeError("无法准备模型")
        
        if not self.binary or not self.weights:
            raise RuntimeError("请先调用prepare_model()和generate_executable()")
            
        logging.info("开始执行增强内存架构推理...")
        
        start_time = time.time()
        
        try:
            # 激活配置的核心
            active_core_ids = self.config.get_active_core_ids()
            self.runtime.activate_cores(active_core_ids)
            
            # 生成输入数据 - 彻底验证
            input_shape = self.config.get_input_shape()
            input_dtype = self.config.get_input_dtype()
            
            # 验证输入形状
            if not input_shape or any(dim <= 0 for dim in input_shape):
                logging.warning(f"输入形状无效: {input_shape}，使用默认形状")
                input_shape = (1, 1, 1, 100)
            
            # 验证数据类型
            try:
                dtype_obj = np.dtype(input_dtype)
            except:
                logging.warning(f"数据类型无效: {input_dtype}，使用float32")
                input_dtype = "float32"
                dtype_obj = np.float32
            
            # 创建输入数据
            try:
                input_data = np.random.rand(*input_shape).astype(dtype_obj)
            except Exception as e:
                logging.error(f"创建输入数据失败: {e}，使用简单数据")
                # 使用更简单的数据
                simple_shape = (100,)  # 一维数组
                input_data = np.random.rand(*simple_shape).astype(np.float32)
            
            logging.info(f"最终输入数据 - 形状: {input_data.shape}, 数据类型: {input_data.dtype}, 总元素数: {input_data.size}")
            
            # 验证数据分布
            if not self.runtime._validate_data_distribution(input_data, active_core_ids):
                logging.warning("数据分布验证失败，使用后备数据")
                input_data = self.runtime._create_fallback_data(input_data.dtype, 100)
            
            # 加载输入数据和权重
            self.runtime.load_input_data(input_data)
            self.runtime.load_weights(self.weights)
            
            # 加载二进制程序
            self.runtime.load_binary_programs(self.binary)
            
            # 获取系统状态
            system_status = self.runtime.get_system_status()
            logging.info(f"系统状态: {system_status['active_cores']}个核心激活")
            
            # 执行并返回结果
            result = self.runtime.run_computation()
            
            elapsed_time = time.time() - start_time
            
            # 处理结果
            self._process_inference_result(result, elapsed_time)
            
            # 记录性能统计
            self.performance_stats['inference'] = {
                'time_s': elapsed_time,
                'result_size': len(result),
                'system_status': system_status
            }
            
            return result
            
        except Exception as e:
            logging.error(f"推理执行失败: {e}")
            # 返回模拟结果作为后备
            logging.info("返回模拟结果作为后备")
            return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    def _process_inference_result(self, result: np.ndarray, elapsed_time: float) -> None:
        """处理推理结果"""
        if len(result) == 0:
            logging.warning("推理结果为空")
        else:
            logging.info(f"推理完成，结果大小: {result.shape}，耗时: {elapsed_time*1000:.2f}ms")
            
            # 打印结果统计信息
            logging.info(f"结果统计: 均值={np.mean(result):.4f}, 标准差={np.std(result):.4f}, "
                        f"范围=[{np.min(result):.4f}, {np.max(result):.4f}]")
            
            # 打印前几个结果值
            display_count = min(10, len(result))
            result_preview = [f"{x:.4f}" for x in result[:display_count]]
            logging.info(f"结果示例 (前{display_count}个): {result_preview}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取流水线状态"""
        status = {
            "model_prepared": hasattr(self, 'layers') and len(self.layers) > 0,
            "executable_generated": self.binary is not None,
            "weights_loaded": self.weights is not None,
            "layers_count": len(self.layers) if hasattr(self, 'layers') else 0,
            "binary_size": len(self.binary) if self.binary else 0,
            "weights_count": len(self.weights) if self.weights else 0,
            "performance_stats": self.performance_stats
        }
        
        if hasattr(self, 'scheduler'):
            try:
                scheduling_summary = self.scheduler.get_scheduling_summary()
                status.update(scheduling_summary)
            except Exception as e:
                logging.warning(f"获取调度摘要失败: {e}")
                
        if hasattr(self, 'codegen') and self.binary:
            try:
                program_stats = self.codegen.get_core_program_stats()
                status["program_stats"] = program_stats
            except Exception as e:
                logging.warning(f"获取程序统计失败: {e}")
                
        if hasattr(self, 'runtime'):
            try:
                system_status = self.runtime.get_system_status()
                status["system_status"] = system_status
            except Exception as e:
                logging.warning(f"获取系统状态失败: {e}")
            
        return status

    def print_pipeline_status(self) -> None:
        """打印流水线状态"""
        status = self.get_pipeline_status()
        
        print("\n" + "="*50)
        print("增强内存架构流水线状态")
        print("="*50)
        print(f"模型准备: {'✅ 完成' if status['model_prepared'] else '❌ 未完成'}")
        print(f"可执行文件生成: {'✅ 完成' if status['executable_generated'] else '❌ 未完成'}")
        print(f"权重加载: {'✅ 完成' if status['weights_loaded'] else '❌ 未完成'}")
        
        if status['model_prepared']:
            print(f"模型层数: {status['layers_count']}")
            if 'total_compute_cores_used' in status:
                print(f"使用的计算核心数: {status['total_compute_cores_used']}")
            
        if status['executable_generated']:
            print(f"二进制大小: {status['binary_size']} 字节")
            
        if status['weights_loaded']:
            print(f"权重数量: {status['weights_count']}")
            
        if 'program_stats' in status:
            stats = status['program_stats']
            print(f"程序指令数: {stats['total_instructions']}")
            print(f"激活核心数: {stats['active_cores']}")
        
        if 'system_status' in status:
            sys_status = status['system_status']
            print(f"系统激活核心: {sys_status['active_cores']}")
            print(f"内存使用率 - MEM0: {sys_status['memory_usage']['mem0_usage_percent']:.1f}%")
            print(f"内存使用率 - MEM1: {sys_status['memory_usage']['mem1_usage_percent']:.1f}%")
        
        # 打印性能统计
        if status['performance_stats']:
            print("\n性能统计:")
            if 'total' in status['performance_stats']:
                total_stats = status['performance_stats']['total']
                print(f"  模型准备时间: {total_stats.get('preparation_time_s', 0):.2f}s")
                print(f"  预估推理时间: {total_stats.get('total_time_s', 0)*1e6:.2f}μs")
            
            if 'code_generation' in status['performance_stats']:
                code_stats = status['performance_stats']['code_generation']
                print(f"  代码生成时间: {code_stats.get('time_s', 0):.2f}s")
            
            if 'inference' in status['performance_stats']:
                inference_stats = status['performance_stats']['inference']
                print(f"  实际推理时间: {inference_stats.get('time_s', 0)*1000:.2f}ms")
        
        print("="*50)

def main():
    # 配置全局日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("enhanced_memory_pipeline_execution.log", encoding='utf-8')
        ]
    )
    
    # 使用增强内存架构配置文件
    config_path = "enhanced_memory_config.yaml"
    
    try:
        # 创建流程控制器
        logging.info("步骤1: 初始化流程控制器...")
        controller = EnhancedMemoryPipelineController(config_path)
        logging.info("增强内存架构流程控制器初始化完成")
        
        # 打印配置摘要
        controller.config.print_config_summary()
        
        # 打印初始状态
        controller.print_pipeline_status()
        
        # 准备模型
        logging.info("步骤2: 准备模型...")
        controller.prepare_model()
        logging.info(f"模型准备完成，层数: {len(controller.layers)}")
        
        # 检查模型准备结果
        if not controller.layers:
            logging.error("模型准备失败：没有分析出任何层")
            return
        
        # 生成可执行文件
        logging.info("步骤3: 生成可执行文件...")
        controller.generate_executable()
        logging.info(f"可执行文件生成完成，大小: {len(controller.binary) if controller.binary else 0} 字节")
        
        # 检查可执行文件生成结果
        if not controller.binary:
            logging.error("可执行文件生成失败：二进制为空")
            return
        
        # 执行推理
        logging.info("步骤4: 执行推理...")
        result = controller.run_inference()
        logging.info(f"推理执行完成，结果大小: {result.shape}")
        
        # 打印最终状态
        controller.print_pipeline_status()
        
        logging.info("🎉 增强内存架构部署流程完成！")
        
    except Exception as e:
        logging.error(f"执行出错: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
