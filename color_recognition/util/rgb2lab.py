# -*- coding: utf-8 -*- 
# @Time 2020/3/2 17:30
# @Author wcy

import numpy as np
import cv2


# region 辅助函数
# RGB2XYZ空间的系数矩阵
M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])

def f(im_channel):
    return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931


def anti_f(im_channel):
    return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
# endregion


# region RGB 转 Lab
# 像素值RGB转XYZ空间，pixel格式:(B,G,R)
# 返回XYZ空间下的值
def __rgb2xyz__(pixel):
    b, g, r = pixel[0], pixel[1], pixel[2]
    rgb = np.array([r, g, b])
    # rgb = rgb / 255.0
    # RGB = np.array([gamma(c) for c in rgb])
    XYZ = np.dot(M, rgb.T)
    XYZ = XYZ / 255.0
    return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)


def __xyz2lab__(xyz):
    """
    XYZ空间转Lab空间
    :param xyz: 像素xyz空间下的值
    :return: 返回Lab空间下的值
    """
    F_XYZ = [f(x) for x in xyz]
    L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return (L, a, b)


def RGB2Lab(pixel):
    """
    RGB空间转Lab空间
    :param pixel: RGB空间像素值，格式：[G,B,R]
    :return: 返回Lab空间下的值
    """
    xyz = __rgb2xyz__(pixel)
    Lab = __xyz2lab__(xyz)
    return Lab
