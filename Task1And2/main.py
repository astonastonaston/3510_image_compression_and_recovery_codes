import cv2
import numpy as np
import matplotlib.pyplot as plt
from svd import svd
from pca import block_pca
import time

dct_time, svd_time, pca_time = [], [], []

# compute MSE for two images
def MSE(imA, imB):
	"""
    Compute images' MSE, and return the error
    """
	err = np.sum((imA.astype("float") - imB.astype("float")) ** 2)
	err /= float(imA.shape[0] * imA.shape[1])
	return err

# whole-image SCD
def img_svd(img_f32):
    """
    SVD the input image
    """
    compr_ratios = [[1, 1], [2, 1], [4, 1], [8, 1], [16, 1], [32, 1], [64, 1], [128, 1], [256, 1]]
    img_svd_all = []
    for i in range(len(compr_ratios)):
        ratio = compr_ratios[i]
        T1 = time.clock()
        img_svd = svd(img_f32, ratio=ratio)
        T2 = time.clock()
        img_svd_all.append(img_svd)
        svd_time.append((T2 - T1)*1000)
    return compr_ratios, img_svd_all

# block-based PCA compression
def block_img_pca(img_f32):
    """
    perform block-based image PCA
    """
    compr_ratios = [[1, 1], [2, 1], [4, 1], [8, 1], [16, 1], [32, 1], [64, 1], [128, 1], [256, 1]]
    img_pca_all = []
    for i in range(len(compr_ratios)): 
        ratio = compr_ratios[i]
        T1 = time.clock()
        img_pca = block_pca(img_f32, block_size=16, ratio=ratio)
        T2 = time.clock()
        img_pca_all.append(img_pca)
        pca_time.append((T2 - T1)*1000)
    return compr_ratios, img_pca_all

# block-based DCT compression
def block_img_dct(img_f32, img_size):
    """
    perform block-based dct on image 
    """
    (width, height) = img_size
    compr_ratios = [[1, 1], [2, 1], [4, 1], [8, 1], [16, 1], [32, 1], [64, 1], [128, 1], [256, 1]]
    block_size = 16
    block_y = height // block_size
    block_x = width // block_size
    height_align = block_y * block_size
    width_align = block_x * block_size
    img_f32_cut = img_f32[:height_align, :width_align]
    img_dct_log, img_idct = [], []

    # compress with different ratios
    for i in range(len(compr_ratios)):
        T1 = time.clock()
        
        new_img = np.zeros((height_align, width_align), dtype=np.float32)
        new_img_log_dct = np.zeros((height_align, width_align), dtype=np.float32)
        ratio = compr_ratios[i][0]/compr_ratios[i][1]
        compr_width = int(np.int(np.sqrt(block_size * block_size / ratio)))
        if (compr_ratios[i][0] in [8, 16, 64, 128]):
            compr_width += 1
        if (compr_ratios[i][0] in [32]):
            compr_width += 2
        # print(compr_width)

        for h in range(block_y):
            for w in range(block_x):
                # block-based DCT
                img_block = img_f32_cut[block_size*h: block_size*(h+1), block_size*w: block_size*(w+1)]
                img_block_dct = cv2.dct(img_block)

                # Compress the spectrum
                img_block_dct_cpr = np.zeros((block_size, block_size))
                img_block_dct_cpr[0:compr_width, 0:compr_width] = img_block_dct[0:compr_width, 0:compr_width]
                img_block_log_dct_cpr = np.log(abs(img_block_dct_cpr))    # log the spectrum to visualize

                # IDCT to obtain compressed results
                img_block = cv2.idct(img_block_dct_cpr)
                new_img[block_size*h: block_size*(h+1), block_size*w: block_size*(w+1)] = img_block
                new_img_log_dct[block_size*h: block_size*(h+1), block_size*w: block_size*(w+1)] = img_block_log_dct_cpr
        img_idct.append(new_img)
        img_dct_log.append(new_img_log_dct)
        
        T2 = time.clock()
        dct_time.append((T2 - T1)*1000)
    return compr_ratios, img_dct_log, img_idct

def visualize_ratio_images(compr_ratios, img_dct_log, img_idct, algo):
    """
    visualize images with different compression ratios
    """
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            # add spectrum after compression
            ratio = compr_ratios[i*3+j]
            ax[i][j].imshow(img_idct[i*3+j], cmap ='gray')    
            ax[i][j].set_title('{}:{} {} compressed result'.format(ratio[0], ratio[1], algo), fontsize=8, y=0.98)   
    plt.suptitle("{} with different compression ratios".format(algo))
    plt.show()
    return 0

def plot_mse_ratio_curve(compr_ratios, img_idct, img_pca_all, img_svd_all, img_f32):
    """
    plot mse-ratio curve
    """
    mses_dct = []
    for i in range(len(img_idct)):
        mse = MSE(img_idct[i], img_f32)
        mses_dct.append(mse)
    
    mses_pca = []
    for i in range(len(img_pca_all)):
        mse = MSE(img_pca_all[i], img_f32)
        mses_pca.append(mse)

    mses_svd = []
    for i in range(len(img_svd_all)):
        mse = MSE(img_svd_all[i], img_f32)
        mses_svd.append(mse)

    # print("Ratios:")
    # print(compr_ratios)
    # print("MSE values:")
    # print(mses_dct)
    # print(mses_pca)
    # print(mses_svd)

    # plot mse-ratio curve
    x_dct = np.array(["{}:{}".format(compr_ratios[i][0], compr_ratios[i][1]) for i in range(len(compr_ratios))])
    y_dct = np.array(mses_dct)
    x_pca = np.array(["{}:{}".format(compr_ratios[i][0], compr_ratios[i][1]) for i in range(len(compr_ratios))])
    y_pca = np.array(mses_pca)
    x_svd = np.array(["{}:{}".format(compr_ratios[i][0], compr_ratios[i][1]) for i in range(len(compr_ratios))])
    y_svd = np.array(mses_svd)
    plt.subplot(1, 1, 1)
    plt.plot(x_dct, y_dct, color = 'green', label = 'DCT')
    plt.plot(x_pca, y_pca, color = 'red', label = 'PCA')
    plt.plot(x_svd, y_svd, color = 'blue', label = 'SVD')
    plt.title("MSE-ratio curve")
    plt.legend()
    plt.show()
    return 0


def plot_time_ratio_curve(compr_ratios):
    """
    plot run_time-ratio curve
    """
    # plot mse-ratio curve
    x_dct = np.array(["{}:{}".format(compr_ratios[i][0], compr_ratios[i][1]) for i in range(len(compr_ratios))])
    y_dct = np.array(dct_time)
    x_pca = np.array(["{}:{}".format(compr_ratios[i][0], compr_ratios[i][1]) for i in range(len(compr_ratios))])
    y_pca = np.array(pca_time)
    x_svd = np.array(["{}:{}".format(compr_ratios[i][0], compr_ratios[i][1]) for i in range(len(compr_ratios))])
    y_svd = np.array(svd_time)
    # print("dct time:") 
    # print(y_dct) 
    # print("pca time:")
    # print(y_pca)
    # print("svd time:")
    # print(y_svd)
    plt.subplot(1, 1, 1)
    plt.plot(x_dct, y_dct, color = 'green', label = 'DCT')
    plt.plot(x_pca, y_pca, color = 'red', label = 'PCA')
    plt.plot(x_svd, y_svd, color = 'blue', label = 'SVD')
    plt.title("Time-ratio curve")
    plt.legend()
    plt.show()
    return 0

if __name__ == '__main__':
    img_path = "../static/lenna.jpg"
    reso = (512, 512)
    img_u8 = cv2.imread(img_path, 0) # read in grayscale
    img_u8 = cv2.resize(img_u8, reso) 
    img_f32 = img_u8.astype(np.float)
    img_size = img_u8.shape # (width, height)

    # Task1: Compressing using DCT with different ratios
    compr_ratios, img_dct_log, img_idct = block_img_dct(img_f32, img_size)
    visualize_ratio_images(compr_ratios, img_dct_log, img_idct, "DCT") # 1.2

    # Task2: Compressing with other methods
    # Block-based PCA compression
    compr_ratios, img_pca_all = block_img_pca(img_f32)
    visualize_ratio_images(compr_ratios, img_dct_log, img_pca_all, "PCA") # 1.2

    # SVD compression
    compr_ratios, img_svd_all = img_svd(img_f32)
    visualize_ratio_images(compr_ratios, img_dct_log, img_svd_all, "SVD") # 1.2
    
    # plot curve comparions
    plot_mse_ratio_curve(compr_ratios, img_idct, img_pca_all, img_svd_all, img_f32) # 1.3
    plot_time_ratio_curve(compr_ratios) # 1.3

