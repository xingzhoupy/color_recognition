# -*- coding: utf-8 -*- 
# @Time : 2020/3/2 9:22 
# @Author : Allen 
# @Site :  检查文件是否存在
import os


def exists_file(path):
    if os.path.exists(path):
        return path
    return None
