# -*- coding: utf-8 -*- 
# @Time 2020/3/9 13:34
# @Author wcy
import base64
import json
import numpy as np
import cv2
import requests
import pprint
if __name__ == '__main__':
    image_url = "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1583742211078&di=e4ddad6dc96329fc07db1631ce7e77ad&imgtype=0&src=http%3A%2F%2Fa.vpimg4.com%2Fupload%2Fmerchandise%2F162065%2FQBH-B351716011-2.jpg"
    # url = "http://127.0.0.1:5050/colorDiscrimination"
    url = "http://222.85.230.14:12347/colorDiscrimination"
    file_path = r"D:\pySpace\color_recognition\resources\color2\薰衣草红2\薰衣草红2_9090.jpg"
    file_path = file_path.encode('utf-8').decode('utf-8')
    frame = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    retval, buffer = cv2.imencode('.jpg', frame)
    base64_pic = base64.b64encode(buffer)
    base64_pic = base64_pic.decode()
    data = {"img": base64_pic}
    r = requests.post(url, data=json.dumps(data))
    pprint.pprint(r.json())