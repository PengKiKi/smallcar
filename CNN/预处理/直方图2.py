# coding:utf-8
__author__ = 'Microcosm'

import cv2
import numpy as np
from matplotlib import pyplot as plt

file = 'f:\\lane.jpg'
img = cv2.imread(file, 0)
# OpenCV 函数
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist, color='r')
plt.xlim([0, 256])
plt.show()

# Numpy 函数
# ravel 是将矩阵拉成列向量
plt.hist(img.ravel(), 256, [0, 256])
plt.xlim([0, 256])
plt.show()

# 彩色图像直方图
img = cv2.imread(file, 1)
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()

# 设置掩膜，只统计掩膜区域的直方图
# 创造一个掩膜
img = cv2.imread(file, 0)
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img, img, mask=mask)

hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, "gray")
plt.subplot(222), plt.imshow(mask, "gray")
plt.subplot(223), plt.imshow(masked_img, "gray")
plt.subplot(224), plt.plot(hist_full, color='r'), plt.plot(hist_mask, color='g'), plt.xlim([0, 256])
plt.show()

# 直方图均衡化
img = cv2.imread(file, 0)
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))  # 将图像拼在一起

cv2.namedWindow("equ")
cv2.imshow("equ", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# CLAHE 有限对比适应性直方图均衡化,效果更好
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
res1 = np.hstack((equ, cl1))  # 将图像拼在一起

cv2.imshow("CLAHE", res1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2D 直方图
img = cv2.imread(file, 1)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
plt.imshow(hist, interpolation='nearest'), plt.xlabel("S"), plt.ylabel("H")
plt.show()