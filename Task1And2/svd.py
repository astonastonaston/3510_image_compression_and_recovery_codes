# -*- coding: utf-8 -*-
import cv2
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

def decode(u, sigma, v, k):
    """
    decode the compressed image
    """
    m = len(u)
    n = len(v)
    a = np.zeros((m, n))
    a = np.dot(u[:, :k], np.diag(sigma[:k])).dot(v[:k, :])
    # s1 =  np.size(u[:, :k])
    # s1 += k
    # s1 += np.size(v[:k, :])
    # s2 = m * n
    # print(s2)
    # print(s1)
    # print("compression ratio：",s2/s1)
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')

def SVD(frame,K=10):
    """
    perform svd on a given frame, and compress the image
    """
    a = np.array(frame)
    u, sigma, v = np.linalg.svd(a[:, :])
    I = decode(u, sigma, v, K)
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
