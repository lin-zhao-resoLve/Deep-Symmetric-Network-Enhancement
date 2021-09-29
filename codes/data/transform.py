# -*- coding: utf-8 -*-
# 提供图像处理辅助函数，色彩空间转换
# @Author  : BQH
# @File    : tools.py
# @Date    : 2018-11-07

import numpy as np
import cv2


# region 辅助函数
# RGB2XYZ空间的系数矩阵
M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])


# im_channel取值范围：[0,1]
def f(im_channel):
    return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931


def anti_f(im_channel):
    return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
# endregion


# region RGB 转 Lab
# 像素值RGB转XYZ空间，pixel格式:(R,G,B)
# 返回XYZ空间下的值
def __rgb2xyz__(pixel):
    b, g, r = pixel[2], pixel[1], pixel[0]
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


# endregion

# region Lab 转 RGB
def __lab2xyz__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0

    x = anti_f(fX)
    y = anti_f(fY)
    z = anti_f(fZ)

    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883

    return (x, y, z)


def __xyz2rgb(xyz):
    xyz = np.array(xyz)
    xyz = xyz * 255
    rgb = np.dot(np.linalg.inv(M), xyz.T)
    # rgb = rgb * 255
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return rgb


def Lab2RGB(Lab):
    xyz = __lab2xyz__(Lab)
    rgb = __xyz2rgb(xyz)
    return rgb
# endregion

# if __name__ == '__main__':
#     img = cv2.imread(r'E:\code\collor_recorrect\test_1.jpg')
#     w = img.shape[0]
#     h = img.shape[1]
#     img_new = np.zeros((w,h,3))
#     lab = np.zeros((w,h,3))
#     for i in range(w):
#         for j in range(h):
#             Lab = RGB2Lab(img[i,j])
#             lab[i, j] = (Lab[0], Lab[1], Lab[2])
#
#     for i in range(w):
#         for j in range(h):
#             rgb = Lab2RGB(lab[i,j])
#             img_new[i, j] = (rgb[2], rgb[1], rgb[0])
#
# 	cv2.imwrite(r'E:\code\collor_recorrect\test.jpg', img_new)

