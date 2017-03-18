# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 单通道直方图测试

src = cv2.imread('f:\\lane.jpg')
cv2.imshow('src', src)

hist = cv2.calcHist([src], [0], None, [256], [0, 255])
plt.plot(hist)
plt.show()

cv2.waitKey()