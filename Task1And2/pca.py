import numpy as np
import cv2

def per_block_pca(k, img):
    img = img / 255 - 0.5   # 灰度图 X 进行正规化  normalization
    img_T = img.transpose()
    Mat = img_T.dot(img)  # 得到 X * X_T

    # 求 X * X_T 的前k大特征向量
    eigvals, vecs = np.linalg.eig(Mat)  # 注意求出的eigvals是特征值，vecs是标准化后的特征向量
    indexs = np.argsort(eigvals)

    # 编码矩阵 D 是前k大特征向量组成的矩阵(正交矩阵)——topk_evecs
    topk_evecs:np.ndarray = vecs[:,indexs[:-k-1:-1]]

    # X * D = 维度压缩后的图片信息——encode  （信息由 512 x 512 压缩为 512 x 64）
    encode = img.dot(topk_evecs)

    # 译码矩阵即 D_T
    img_new = ((encode.dot(topk_evecs.transpose()) + 0.5) * 255).astype(np.uint8)  # 译码得到的新图片
    # print(img_new)
    img_new[img_new < 0] = 0
    img_new[img_new > 255] = 255
    return img_new


def block_pca(img, block_size=16, ratio=[1,1]):
    """
    compress img using given ratio and block size
    """
    block_y = img.shape[1] // block_size
    block_x = img.shape[0] // block_size
    
    # align the image
    height_align = block_y * block_size
    width_align = block_x * block_size
    img_align = np.zeros((height_align, width_align), dtype=np.float32)
    
    # compute k
    ratio = ratio[0]/ratio[1]
    k = int(np.int(np.sqrt(block_size * block_size / ratio)))
    if (ratio in [8, 16, 64, 128]):
        k += 1
    if (ratio in [32]):
        k += 2
    print(k)
    
    # compress each block
    for h in range(block_y):
        for w in range(block_x):
            # 预处理
            img_block = img[block_size*h: block_size*(h+1), block_size*w: block_size*(w+1)]
            img_pca = per_block_pca(k, img_block)
            img_align[block_size*h: block_size*(h+1), block_size*w: block_size*(w+1)] = img_pca
    return img_align








