"""
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
