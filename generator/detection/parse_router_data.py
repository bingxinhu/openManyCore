import os


def parse_router_data(file_name, packet_header_dxy: tuple, core_id=None, t_mode=1):
    """
        t_mode: 1 - 多包；  0 - 单包
    """
    result = []
    with open(file_name, 'r') as rf:
        for line in rf.readlines():
            if line == '\n':
                continue
            line = line[:-1] if line[-1] == '\n' else line
            p_m = (int(line[0], 16) >> 2) & 0x3
            p_1 = int(line[0], 16)
            p_core_id = ((int(line[0], 16) & 0x3) << 2) + ((int(line[1], 16) >> 2) & 0x3)
            x = (-128 + int(line[6], 16) + ((int(line[5], 16) & 0x7) << 4)) if (int(line[5], 16) > 7) else (
                    (int(line[5], 16) << 4) + int(line[6], 16))
            y = (-128 + int(line[8], 16) + ((int(line[7], 16) & 0x7) << 4)) if (int(line[7], 16) > 7) else (
                (int(line[7], 16) << 4) + int(line[8], 16))
            if p_m == 3 and x == packet_header_dxy[0] and y == packet_header_dxy[1]:  # 跨片数据包
                # print(line, "// x =", x, "y =", y)
                temp_str = line[20:28] + "\n" + line[12:20] + "\n"
                result.append(temp_str)
                pass
    with open(file_name + '-parsed-dx{:}-dy{:d}.txt'.format(packet_header_dxy[0], packet_header_dxy[1]), 'w') as wf:
        for str_iter in result:
            wf.write(str_iter)


if __name__ == '__main__':
    name = 'temp/recv04-58-35.txt'
    parse_router_data(file_name=name, packet_header_dxy=(0, -1))
    parse_router_data(file_name=name, packet_header_dxy=(0, -2))
    parse_router_data(file_name=name, packet_header_dxy=(0, -3))
