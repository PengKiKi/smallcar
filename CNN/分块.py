# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 使用2g-r-b分离土壤与背景

src = cv2.imread('f:\\lane.jpg')
cv2.imshow('src', src)

# 转换为浮点数进行计算
fsrc = np.array(src, dtype=np.float32) / 255.0
(b, g, r) = cv2.split(fsrc)
gray = 2 * g - b - r

# 求取最大值和最小值
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
(thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
cv2.imshow('bin_img', bin_img)

bin_img_save = np.copy(bin_img)
(contoures, hierarchy) = cv2.findContours(bin_img_save, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 按面积排序
areas = np.zeros( len(contoures) )
idx = 0
for cont in contoures :
    areas[idx] = cv2.contourArea(cont)
    idx = idx + 1
areas_s = cv2.sortIdx(areas, cv2.SORT_DESCENDING | cv2.SORT_EVERY_COLUMN)

(b8, g8, r8) = cv2.split(src)

# 对每个区域进行处理
for idx in areas_s:
    if areas[idx] < 100:
        break

        # 绘制区域图像，通过将thickness设置为-1可以填充整个区域，否则只绘制边缘
    poly_img = np.zeros(bin_img.shape, dtype=np.uint8)
    cv2.drawContours(poly_img, contoures, idx, [255, 255, 255], -1)
    poly_img = poly_img & bin_img

    # 得到彩色的图像
    color_img = cv2.merge([b8 & poly_img, g8 & poly_img, r8 & poly_img])

    cv2.imshow('poly_img', color_img)






# 计算直方图
hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
plt.plot(hist)
plt.show()



cv2.waitKey()
