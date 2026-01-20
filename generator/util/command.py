import os
import time
import platform

from generator.util.path import Path
from typing import overload


class Command(object):
    @staticmethod
    def execute_hardware(tb_name, executable=True, exe_name='TianjicX1_SIM'):
        print("excute_hardware conmmand.......", tb_name)
        if executable:
            print(exe_name + " " + Path.JSON_CONFIG_ABSOLUTE_PATH + " " + tb_name)
            return os.system(exe_name + " " + Path.JSON_CONFIG_ABSOLUTE_PATH + " " + tb_name)
        else:
            print("top.bat " + tb_name + " " + Path.JSON_CONFIG_ABSOLUTE_PATH)
            return os.system("top.bat " + tb_name + " " + Path.JSON_CONFIG_ABSOLUTE_PATH)

    @staticmethod
    def output_time_to_file(file_path):
        os.system("echo " + time.strftime('%y-%m-%d %H:%M:%S',
                                          time.localtime()) + "> " + file_path)

    @staticmethod
    def compare_files(file1, file2, result_file=None):
        os_name = platform.system()
        if result_file != None:
            print(result_file)
            if os_name == "Windows":
                return os.system(f"fc {file1} {file2} /N >> {result_file}")
            else:
                return os.system(f"diff -uN {file1} {file2} >> {result_file} 2>&1")
        else:
            if os_name == "Windows":
                return os.system(f"fc {file1} {file2} /N >> nul")
            else:
                return os.system(f"diff -uN {file1} {file2}  >> /dev/null")
    @staticmethod
    def open_and_show_file(file_path):
        os.system(file_path)