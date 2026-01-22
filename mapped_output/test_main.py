"""
自动化生成的测试主程序
模型统计:
  算子数量: 12
  参数大小: 241.05 KB
  
硬件统计:
  总周期数: 118,958
  总MAC操作: 118,214
  估计延迟: 594.79 ms
"""

import numpy as np
import sys
import os

sys.path.append(os.getcwd())

# 导入生成的配置
try:
    from g0_config import gen_0_map_config
    from g1_config import gen_1_map_config  
    from g2_config import gen_2_map_config
except ImportError as e:
    print(f"导入配置失败: {e}")
    print("请确保已运行自动化映射流水线生成配置文件")
    sys.exit(1)

from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch


def run_automated_test(case_file_name='auto_generated_test', send_to_fpga=True):
    """运行自动化测试"""
    
    print("=" * 60)
    print("开始自动化测试")
    print("=" * 60)
    
    # 初始化配置生成器
    config = MapConfigGen()
    
    # 时钟配置
    clock_in_phase = 200000
    
    # 阶段使能（所有阶段使能）
    total_phases = 10  # 假设10个阶段
    phase = np.ones(total_phases).astype(int)
    
    # 添加Group0配置 (FPGA)
    print("1. 添加Group0配置...")
    config_0 = gen_0_map_config(
        phase_en=phase,
        clock_in_phase=clock_in_phase,
        size_x=1,
        size_y=1,
        data=None,
        out_data_en=True,
        in_data_en=not send_to_fpga
    )
    config.add_config(config_0, core_offset=(1, 0))
    
    # 添加Group1配置 (计算核心)
    print("2. 添加Group1配置...")
    config_1 = gen_1_map_config(
        phase_en=phase,
        clock_in_phase=clock_in_phase,
        size_x=1,
        size_y=1,
        in_data_en=True,
        out_data_en=True
    )
    config.add_config(config_1, core_offset=(0, 0))
    
    # 添加Group2配置 (路由核心)
    print("3. 添加Group2配置...")
    config_2 = gen_2_map_config(
        phase_en=phase,
        clock_in_phase=clock_in_phase,
        size_x=8,
        size_y=1,
        in_data_en=True,
        out_data_en=True,
        send_to_fpga=send_to_fpga
    )
    config.add_config(config_2, core_offset=(0, 1))
    
    # 添加路由信息
    print("4. 添加路由信息...")
    MapConfigGen.add_router_info(map_config=config.map_config)
    
    # 添加初始数据传输prim
    print("5. 添加初始数据传输prim...")
    prim = {
        'axon': None,
        'soma1': None,
        'router': None,
        'soma2': {
            'addr_in': 0x0000 >> 2,
            'addr_out': 0x8400,
            'addr_ciso': 0x10000 >> 2,
            'length_in': 1024,
            'num_in': 12,
            'length_ciso': 1,
            'num_ciso': 12,
            'length_out': 1024,
            'num_out': 12,
            'type_in': 1,
            'type_out': 1,
            'data_in': None
        }
    }
    
    MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)
    
    # 配置时钟
    config.map_config['sim_clock'] = 100000
    config.map_config['step_clock'] = {
        ((0, 0), 0): (100000 - 1, 100000)
    }
    
    # 准备测试环境
    c_path = os.getcwd()
    out_files_path = os.getcwd() + "/simulator/Out_files/" + case_file_name + "/"
    
    if os.path.exists(out_files_path):
        os.chdir(out_files_path)
        if sys.platform.startswith('win'):
            os.system('rd/s/q cmp_out')
        else:
            os.system('rm -rf cmp_out')
        os.chdir(c_path)
    else:
        os.makedirs(out_files_path, exist_ok=True)
    
    # 配置测试参数
    test_config = {
        'tb_name': case_file_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'test_group_phase': [(0, 1)]
    }
    
    # 运行测试
    print("6. 运行硬件模拟测试...")
    tester = TestEngine(config.map_config, test_config)
    
    try:
        result = tester.run_test()
        if result:
            print("\n✅ 测试通过!")
        else:
            print("\n❌ 测试失败!")
        return result
    except Exception as e:
        print(f"\n⚠️ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='运行自动化生成的测试')
    parser.add_argument('--name', type=str, default='auto_generated_test', 
                       help='测试案例名称')
    parser.add_argument('--no-fpga', action='store_true', 
                       help='不发送到FPGA')
    
    args = parser.parse_args()
    
    # 运行自动化测试
    success = run_automated_test(args.name, not args.no_fpga)
    
    if success:
        print("\n🎉 自动化测试流程完成!")
    else:
        print("\n💥 自动化测试流程失败!")
        sys.exit(1)
