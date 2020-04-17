# -*- coding: utf-8 -*- 
# @Time 2020/4/17 12:30
# @Author wcy

# -*- coding: utf-8 -*-
# @Time 2020/3/18 18:06
# @Author wcy
import time

import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
import sys
sys.path.append("../")
from config import style_model_path, style_class_name_path


class CostumeStyle(object):
    def __init__(self):
        self.class_name_dict = []
        with open(style_class_name_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                self.class_name_dict.append(line.strip('\n') )
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        tf.saved_model.loader.load(self.sess, ["serve"], style_model_path)
        graph = tf.get_default_graph()

        self.input = self.sess.graph.get_tensor_by_name('inputs:0')
        self.output = self.sess.graph.get_tensor_by_name('outputs:0')
        self.sess.run(self.output, feed_dict={self.input: np.zeros((200, 128, 3), dtype=np.uint8)})

    def get_image(self, frame):
        if len(frame.shape) < 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        image = frame[..., :3]
        image = image.astype(np.float)
        return image

    def predict(self, image):
        """
        返回类别名 类别名参考 style_class_name_path
        :param image: BGR图片
        :return: {"class_name": class_name, "score": score, "all_class": all_class}
        """
        images = self.get_image(image)
        softmax = self.sess.run(self.output, feed_dict={self.input:images[..., ::1]})
        type = np.argmax(softmax)
        score = float(softmax.max())
        class_name = self.class_name_dict[int(type)]
        all_class = {class_name: float(score) for class_name, score in zip(self.class_name_dict, softmax[0])}
        return {"class_name": class_name, "score": score, "all_class": all_class}


if __name__ == '__main__':
    ct = CostumeStyle()
    while True:
        res = ct.predict(cv2.imread("../test/image/111.jpg"))
        time.sleep(0.1)
        print(res)
