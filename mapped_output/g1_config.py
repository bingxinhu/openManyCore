"""
自动生成的g1_config配置
"""

import numpy as np
from itertools import product

def gen_1_map_config(phase_en, clock_in_phase, size_x=1, size_y=1, data=None, **kwargs):
    """计算核心组 (1x1)"""
    
    # 硬件配置
    chip = (0, 0)  # 转换为元组
    
    # 初始化配置字典
    map_config = {
        'sim_clock': 100000,
        (chip, 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    
    # 为每个核心添加配置
    for core_y, core_x in product(range(size_y), range(size_x)):
        core_id = (chip, (core_x, core_y))
        map_config[(chip, 0)][0][core_id] = {
            'prims': []
        }
    
    phase_group = map_config[(chip, 0)][0]
    
    # 这里可以添加具体的原语配置
    # 示例: 添加一个数据搬运原语
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            core_id = (chip, (core_x, core_y))
            
            # 添加示例原语
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
            
            phase_group[core_id]['prims'].append(prim)
    
    return map_config

if __name__ == '__main__':
    # 测试配置生成
    phase = np.ones(10).astype(int)
    config = gen_1_map_config(phase, 200000)
    print(f"g1_config 生成成功")
