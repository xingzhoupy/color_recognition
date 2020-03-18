# -*- coding: utf-8 -*- 
# @Time 2020/3/18 18:06
# @Author wcy
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np


class CostumeType(object):
    def __init__(self):
        self.sess = tf.Session()
        with tf.gfile.FastGFile('../resources/pb_model/model.pb', 'rb') as f:
            graph_pen = tf.GraphDef()
            graph_pen.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_pen, name='')  # 导入计算图
            self.input = self.sess.graph.get_tensor_by_name('input/input_image:0')
            self.softmax = self.sess.graph.get_tensor_by_name('losses/Softmax:0')
        self.sess.run(self.softmax, feed_dict={self.input: np.zeros((1, 128, 128, 3))})

    def resize_image(self, image, width, height):
        top, bottom, left, right = (0, 0, 0, 0)
        # 获取图像尺寸
        h, w, _ = image.shape
        if h / w < height / width:
            # "填充上下"
            hd = int(w * height / width - h + 0.5)
            top = hd // 2
            bottom = hd - top
            scale = w / width
        elif h / w > height / width:
            # "填充左右"
            wd = int(h * width / height - w + 0.5)
            left = wd // 2
            right = wd - left
            scale = h / height
        else:
            scale = 1
        # RGB颜色
        BLACK = [255, 255, 255]

        # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
        constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

        # 调整图像大小并返回
        return cv2.resize(constant, (width, height)), [top / scale, bottom / scale, left / scale, right / scale]

    def get_image(self, frame):
        if len(frame.shape) < 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = frame[..., :3]
        image = frame[..., ::1]
        image, edge = self.resize_image(image, 128, 128)
        image = image.astype(np.float)
        image_norm = (image - 127) / 127
        image_norm = np.expand_dims(image_norm, axis=0)
        return image_norm

    def predict(self, image):
        images = self.get_image(image)
        res = self.sess.run(self.softmax, feed_dict={self.input:images})
        type = np.argmax(res)
        return type


if __name__ == '__main__':
    ct = CostumeType()
    res = ct.predict(cv2.imread("../test/10.png"))
    print()
