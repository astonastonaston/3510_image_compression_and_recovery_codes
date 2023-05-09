# -*- coding: utf-8 -*-
import cv2
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

# 转为u8类型
def restore1(u, sigma, v, k):
    m = len(u)
    n = len(v)
    a = np.zeros((m, n))
    a = np.dot(u[:, :k], np.diag(sigma[:k])).dot(v[:k, :])
    # s1 =  np.size(u[:, :k])
    # s1 += k
    # s1 += np.size(v[:k, :])
    # s2 = m * n
    # # print(s2)
    # # print(s1)
    # print("压缩率：",s2/s1)
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')

def SVD(frame,K=10):
    a = np.array(frame)
    # 由于是彩色图像，所以3通道。a的最内层数组为三个数，分别表示RGB，用来表示一个像素
    u, sigma, v = np.linalg.svd(a[:, :])
    I = restore1(u, sigma, v, K)
    return I

def svd(img, ratio):
    """
    use SVD to compress image given ratio
    """
    ratio_k_dict = {
        1/1: 255, 
        2/1: 127, 
        4/1: 63, 
        8/1: 31, 
        16/1: 16, 
        32/1: 8, 
        64/1: 4, 
        128/1: 2, 
        256/1: 1
    }
    ratio = ratio[0]/ratio[1]
    k = ratio_k_dict[ratio]
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    I = SVD(img, k)
    return I
