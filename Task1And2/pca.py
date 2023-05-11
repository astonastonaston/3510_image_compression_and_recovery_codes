import numpy as np
import cv2

def per_block_pca(k, img):
    """
    compress image using pca for each block
    """
    # normalization
    img = img / 255 - 0.5   
    img_T = img.transpose()

    # compute X_T*X's k-biggest eigen-vectors 
    M = img_T.dot(img)
    eigvals, vecs = np.linalg.eig(M)      
    indexs = np.argsort(eigvals)

    # computer D containing top-k eigen-vectors
    topk_evecs:np.ndarray = vecs[:,indexs[:-k-1:-1]]

    # compute X*D: encoded image information, reducing from 512*512 to 512*64
    encode = img.dot(topk_evecs)

    # compute D^T: decoder, and decode the image
    img_new = ((encode.dot(topk_evecs.transpose()) + 0.5) * 255).astype(np.uint8) 
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
    # print(k)
    
    # compress each block
    for h in range(block_y):
        for w in range(block_x):
            # processing each block
            img_block = img[block_size*h: block_size*(h+1), block_size*w: block_size*(w+1)]
            img_pca = per_block_pca(k, img_block)
            img_align[block_size*h: block_size*(h+1), block_size*w: block_size*(w+1)] = img_pca
    return img_align








