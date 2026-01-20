import pandas
import json


def gen_router_info_from_matrix(dir='', core_w=16, core_h=10):
    router_json = {
        "chip_w": 1,
        "chip_h": 1,
        "core_w": core_w,
        "core_h": core_h,
        'mode': 1,
        'phase_list': {
            0: {  # phase idx
                0: []  # group idx
            }
        }
    }
    df = pandas.read_csv(dir, sep='\t', dtype=str)
    for core_x_idx in range(core_w):
        for core_y_idx in range(core_h):
            id = core_x_idx + core_y_idx * core_w
            router_json['phase_list'][0][0].append(
                {
                    "core_id": {
                        "chip_x": 0,
                        "chip_y": 0,
                        "core_x": core_x_idx,
                        "core_y": core_y_idx
                    },
                    "A2S2_mode": 0,
                    "A_time": 0,
                    "S1_time": 0,
                    "S2_time": 0,
                    "multicast_core": False,
                    "packets": []
                })
            line = df.loc[id]
            for dst_core_x_idx in range(core_w):
                for dst_core_y_idx in range(core_h):
                    dst_id = dst_core_x_idx + dst_core_y_idx * core_w
                    if int(line[dst_id + 1]) > 0:
                        router_json['phase_list'][0][0][-1]['packets'].append(
                            {
                                "dst_core_id": {
                                    "chip_x": 0,
                                    "chip_y": 0,
                                    "core_x": dst_core_x_idx,
                                    "core_y": dst_core_y_idx
                                },
                                "pack_num": int(line[dst_id + 1]),
                                "multicast_pack": False
                            }
                        )
    return router_json


if __name__ == '__main__':
    result = gen_router_info_from_matrix(dir='temp/router/ex1.log', core_w=5, core_h=5)
    with open('./temp/router/log.json', 'w') as f:
        json.dump(result, f, indent=4, separators=(',', ':'), sort_keys=True)
