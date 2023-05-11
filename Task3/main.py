import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def salt_pepper(image, salt, pepper):
    """
    adding salt-pepper noises to the image
    """
    height = image.shape[0]
    width = image.shape[1]
    sp_total_per = salt + pepper    #总噪声占比
    noise_image = image.copy()
    noise_num = int(sp_total_per * height * width)
    for i in range(noise_num):
        rows = np.random.randint(0, height-1)
        cols = np.random.randint(0, width-1)
        if(np.random.randint(0,100)<salt*100):
            noise_image[rows][cols] = 255
        else:
            noise_image[rows][cols] = 0
    return noise_image


def main():
    path = "../static/lenna.jpg"
    img = cv2.imread(path, 0)

    # implant salt-pepper noise
    noise = salt_pepper(img, 0.1, 0.1)

    # eliminating the noise using different filters
    res_median = cv2.medianBlur(noise, 5) # median filter 
    res_mean = cv2.blur(noise,(7,7)) # mean filter 
    res_Gaussian = cv2.GaussianBlur(noise,(7,7),0) # gaussian filter
    res_bilateral = cv2.bilateralFilter(noise,40,75,75) # bilateral filter

    res_dict = {
        0: res_median, 
        1: res_mean,
        2: res_Gaussian,
        3: res_bilateral
    }
    text_dict = {
        0: "Median", 
        1: "Mean",
        2: "Gaussian",
        3: "bilateral"
    }

    # visualize filtering results
    fig, ax = plt.subplots(3, 4, figsize=(10, 10))
    for i in range(3):
        for j in range(4):
            if (i==0): # original image
                ax[i][j].imshow(img, cmap ='gray')    
                ax[i][j].set_title("original image", fontsize=8, y=0.98)   
            elif (i==1): # noisy image 
                ax[i][j].imshow(noise, cmap ='gray')    
                ax[i][j].set_title("noisy image", fontsize=8, y=0.98)   
            elif (i==2): # recovered image
                ax[i][j].imshow(res_dict[j], cmap ='gray')    
                ax[i][j].set_title("{} filter recovered image".format(text_dict[j]), fontsize=8, y=0.98)   
    plt.suptitle("Recovery with different algorithms")
    plt.show()

    # visualizing MSEs
    mses = [np.mean(np.square(res_dict[i]-img)) for i in range(4)]
    plt.bar(text_dict.values(), mses)
    plt.title("MSEs for different recovery algorithms")
    plt.show()



main()