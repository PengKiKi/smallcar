import cv2
import numpy as np

# filter2D

src = cv2.imread('f:\\lane.jpg')
cv2.imshow('src', src)

kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

dst = cv2.filter2D(src, -1, kernel)
cv2.imshow('dst', dst)

cv2.waitKey()