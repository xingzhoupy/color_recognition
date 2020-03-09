#!/user/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   config
   Description:
   Author:      allen
   Date:        4/14/19 4:45 PM

---------------------
"""
__author__ = "zx"

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_YAML = os.path.join(BASE_DIR, 'log.yaml')
RESOURCE_PATH = os.path.join(BASE_DIR, 'resources')
# 百度在线翻译账户密码
baidu_key = "20190808000324931"
baidu_passwd = "K52NIO04wx20O8gV1YF7"

class Config(object):
    DEBUG = False
    TESTING = False

    JSONSCHEMA_DIR = os.path.join(BASE_DIR, "schemas")


def read_file(path):
    with open(path, encoding="utf8") as file:
        data = ''.join(file.readlines())
    return data


def save_excel(model, path):
    df = pd.DataFrame(model)
    df.to_excel(path, index=False)
