import os
import platform
import errno

from generator.util.name_generator import NameGenerator


class Path(object):
    # 使用os.path.sep确保路径分隔符兼容当前系统
    WORK_DIR = os.path.abspath(os.curdir)

    MAPPER_DIR = os.path.join('mapper', '')
    GENERATOR_DIR = os.path.join('generator', '')
    HARDWARE_DIR = os.path.join('simulator', '')
    TEMP_DIR = os.path.join('temp', '')

    COMPARE_RESULT_DIR = os.path.join(TEMP_DIR, 'compare_result', '')
    SIMULATION_LOG_DIR = os.path.join(TEMP_DIR, 'out_files', '')
    JSON_CONFID_DIR = os.path.join(TEMP_DIR, 'config', '')

    JSON_CONFIG_ABSOLUTE_PATH = ""

    @staticmethod
    def json_config_path(tb_name):
        """获取JSON配置文件路径"""
        Path._create_dir_if_not_exists(Path.JSON_CONFID_DIR)
        path = os.path.join(Path.JSON_CONFID_DIR, f"{tb_name}Config.json")
        Path.JSON_CONFIG_ABSOLUTE_PATH = os.path.abspath(path)
        return path

    @staticmethod
    def json_config_absolute_path(tb_name):
        """获取JSON配置文件的绝对路径"""
        return os.path.abspath(Path.json_config_path(tb_name))

    @staticmethod
    def simulation_case_dir(tb_name):
        """获取仿真案例目录"""
        return os.path.join(Path.SIMULATION_LOG_DIR, tb_name, '')

    @staticmethod
    def simulation_out_dir(tb_name):
        """获取仿真输出目录"""
        return os.path.join(Path.simulation_case_dir(tb_name), "output", '')

    @staticmethod
    def simulation_input_dir(tb_name, x, y):
        """获取仿真输入目录"""
        input_dir = NameGenerator.input_dir_name(x, y)
        return os.path.join(Path.simulation_case_dir(tb_name), input_dir, '')

    @staticmethod
    def simulation_cmp_out_dir(tb_name):
        """获取仿真比较输出目录"""
        return os.path.join(Path.simulation_case_dir(tb_name), "cmp_out", '')

    @staticmethod
    def hardware_cmp_out_dir(tb_name):
        """获取硬件比较输出目录"""
        return os.path.join(Path.hardware_out_dir(tb_name), "cmp_out", '')

    @staticmethod
    def hardware_out_dir(tb_name=''):
        """获取硬件输出目录"""
        if tb_name:
            dir_path = os.path.join(Path.HARDWARE_DIR, "Out_files", tb_name, '')
        else:
            dir_path = os.path.join(Path.HARDWARE_DIR, "Out_files", '')
            
        Path._create_dir_if_not_exists(dir_path)
        return dir_path

    @staticmethod
    def hardware_debug_message_dir():
        """获取硬件调试消息目录"""
        return os.path.join(Path.HARDWARE_DIR, "Debug_files", '')

    @staticmethod
    def create_compare_result_dir():
        """创建比较结果目录"""
        Path._create_dir_if_not_exists(Path.COMPARE_RESULT_DIR)
        return Path.COMPARE_RESULT_DIR

    @staticmethod
    def create_simulation_case_dir(tb_name):
        """创建仿真案例目录"""
        Path._create_dir_if_not_exists(Path.SIMULATION_LOG_DIR)
        case_dir = Path.simulation_case_dir(tb_name)
        Path._create_dir_if_not_exists(case_dir)
        return case_dir

    @staticmethod
    def create_simulation_input_dir(tb_name, x, y):
        """创建仿真输入目录"""
        Path.create_simulation_case_dir(tb_name)
        input_dir = Path.simulation_input_dir(tb_name, x, y)
        Path._create_dir_if_not_exists(input_dir)
        return input_dir

    @staticmethod
    def create_simulation_out_dir(tb_name):
        """创建仿真输出目录"""
        Path.create_simulation_case_dir(tb_name)
        out_dir = Path.simulation_out_dir(tb_name)
        Path._create_dir_if_not_exists(out_dir)
        return out_dir

    @staticmethod
    def create_simulation_cmp_out_dir(tb_name):
        """创建仿真比较输出目录"""
        Path.create_simulation_case_dir(tb_name)
        cmp_dir = Path.simulation_cmp_out_dir(tb_name)
        Path._create_dir_if_not_exists(cmp_dir)
        return cmp_dir

    @staticmethod
    def delete_all(path):
        """删除路径下的所有内容"""
        if not os.path.exists(path):
            return
            
        if os.path.isdir(path):
            for name in os.listdir(path):
                sub_path = os.path.join(path, name)
                if os.path.isdir(sub_path):
                    Path.delete_all(sub_path)
                    try:
                        os.rmdir(sub_path)
                    except OSError as e:
                        if e.errno != errno.ENOENT:  # 忽略目录不存在的错误
                            raise
                else:
                    try:
                        os.remove(sub_path)
                    except OSError as e:
                        if e.errno != errno.ENOENT:  # 忽略文件不存在的错误
                            raise
        else:
            try:
                os.remove(path)
            except OSError as e:
                if e.errno != errno.ENOENT:  # 忽略文件不存在的错误
                    raise

    @staticmethod
    def _create_dir_if_not_exists(directory):
        """创建目录（包括父目录），如果不存在"""
        if not directory:
            return
            
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            # 处理权限问题等异常
            if e.errno != errno.EEXIST:
                raise
