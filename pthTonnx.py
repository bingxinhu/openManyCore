#!/usr/bin/env python3
"""
LeNet模型转换工具 - 修复版
自动处理依赖问题
"""

import sys
import subprocess
import importlib

def check_and_install_deps():
    """检查并安装依赖"""
    
    required_packages = [
        'torch',
        'numpy',
        # 'onnx',      # 可选
        # 'onnxruntime', # 可选
    ]
    
    print("检查依赖...")
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少 {len(missing_packages)} 个包，正在安装...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"  ✓ 已安装 {package}")
            except:
                print(f"  ✗ 无法安装 {package}")
    
    # 特别检查onnxscript
    try:
        import onnxscript
        print("  ✓ onnxscript")
    except ImportError:
        print("  ✗ onnxscript (如果使用PyTorch 2.0+可能需要)")

def convert_model():
    """转换模型的主函数"""
    
    import torch
    import torch.nn as nn
    
    # 简单的LeNet定义
    class MinimalLeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
            self.pool1 = nn.AvgPool2d(2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.pool2 = nn.AvgPool2d(2)
            self.fc1 = nn.Linear(400, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.pool1(self.relu(self.conv1(x)))
            x = self.pool2(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 加载模型
    try:
        model = MinimalLeNet()
        state_dict = torch.load("./onnx_model/lenet.pth", map_location='cpu')
        
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            # 可能是完整的模型
            model = state_dict
        
        model.eval()
        print("✓ 模型加载成功")
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        # 创建示例模型
        print("创建示例模型...")
        model = MinimalLeNet()
        model.eval()
        torch.save(model.state_dict(), "./onnx_model/lenet.pth")
    
    # 导出ONNX - 使用最简单的方法
    try:
        dummy_input = torch.randn(1, 1, 28, 28)
        
        # 方法1: 使用最简单的export
        torch.onnx.export(
            model,
            dummy_input,
            "./onnx_model/lenet_simple.onnx",
            export_params=True,
            input_names=['input'],
            output_names=['output']
        )
        print("✓ ONNX导出成功: ./onnx_model/lenet_simple.onnx")
        
    except Exception as e:
        print(f"✗ ONNX导出失败: {e}")
        
        # 方法2: 使用更基础的方法
        try:
            print("尝试备选方法...")
            import warnings
            warnings.filterwarnings("ignore")
            
            # 使用_export（内部函数）
            torch.onnx._export(
                model,
                dummy_input,
                "./onnx_model/lenet_fallback.onnx",
                export_params=True,
            )
            print("✓ 备选方法成功: ./onnx_model/lenet_fallback.onnx")
            
        except Exception as e2:
            print(f"✗ 所有方法都失败: {e2}")
            
            # 最后的方法：保存为TorchScript
            traced = torch.jit.trace(model, dummy_input)
            traced.save("./onnx_model/lenet_traced.pt")
            print("✓ 已保存为TorchScript: ./onnx_model/lenet_traced.pt")
            print("  可以使用其他工具转换为ONNX")

def main():
    print("=" * 60)
    print("LeNet模型转换工具")
    print("=" * 60)
    
    # 检查依赖
    check_and_install_deps()
    
    print("\n" + "=" * 60)
    print("开始转换模型...")
    print("=" * 60)
    
    # 执行转换
    convert_model()
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
