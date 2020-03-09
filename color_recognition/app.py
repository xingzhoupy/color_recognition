# -*- coding: utf-8 -*- 
# @Time : 2019/12/23 9:20 
# @Author : Allen 
# @Site :
import json
import numpy as np
import cv2
from flask import request
from jsonschema import ValidationError
import traceback
from color_recognition import app, jsonschema
from flask import jsonify

from color_recognition.helper import exists_file

__author__ = "zhouxing"


@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World!"


@app.route('/colorDiscrimination', methods=['POST'])
@jsonschema.validate('api', 'recognition')
def recognition():
    """OA接口"""
    try:
        request_data = json.loads(request.data)
        app.logger.info("请求参数：{}".format(request_data))
        # 获取参数
        img = request_data["img"]

        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        result = ci.predict(image)
    except Exception:
        traceback.print_exc()
        app.logger.exception(f"{request.data},异常：{traceback.print_exc()}")
        return jsonify(code=500, msg="内部错误")
    return jsonify(code=1, color=result["color"], type=result["type"])


@app.errorhandler(ValidationError)
def on_validation_error(e):
    return jsonify(code=401, msg=f"参数错误")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
