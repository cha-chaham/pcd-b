import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_c1 = cv.imread('bunga.jpg', 0)
plt.imshow(img_c1, "gray"), plt.title("Original Image")


img_c2 = np.fft.fft2(img_c1)
plt.imshow(np.log(1+np.abs(img_c2)), "gray"), plt.title("Spectrum")

img_c3 = np.fft.fftshift(img_c2)
plt.imshow(np.log(1+np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")

img_c4 = np.fft.fftshift(img_c3)
plt.imshow(np.log(1+np.abs(img_c4)), "gray"), plt.title("Decentralized")
plt.show()