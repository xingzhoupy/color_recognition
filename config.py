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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_YAML = os.path.join(BASE_DIR, 'log.yaml')
RESOURCE_PATH = os.path.join(BASE_DIR, 'resources')


class Config(object):
    DEBUG = False
    TESTING = False

    JSONSCHEMA_DIR = os.path.join(BASE_DIR, "schemas")

