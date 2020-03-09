# -*- coding: utf-8 -*- 
# @Time : 2019/12/23 9:20 
# @Author : Allen 
# @Site :
import base64
import json
import numpy as np
import cv2
from flask import request
from jsonschema import ValidationError
import traceback
from color_recognition import app, jsonschema, ci
from flask import jsonify
import base64
from color_recognition.helper import exists_file

__author__ = "zhouxing"


@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World!"


@app.route('/colorDiscrimination', methods=['POST'])
# @jsonschema.validate('api', 'recognition')
def recognition():
    """OA接口"""
    try:
        request_data = json.loads(request.data)
        app.logger.info("请求参数：{}".format(request_data))
        # 获取参数
        image = request_data["img"]
        img = base64.b64decode(image)
        nparr = np.fromstring(img, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = ci.predict(img_np)
    except Exception as e:
        traceback.print_exc()
        app.logger.exception(f"{request.data},异常：{traceback.print_exc()}")
        return jsonify(code=500, msg="内部错误")
    return jsonify(code=1, color=result["color"], type=result["type"])


@app.errorhandler(ValidationError)
def on_validation_error(e):
    return jsonify(code=401, msg=f"参数错误")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
