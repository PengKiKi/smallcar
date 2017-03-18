# coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt


cap=cv2.VideoCapture('solidWhiteRight.mp4')
while True:

    ret,img=cap.read()

    img = cv2.GaussianBlur(img, (13, 13), 1, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 400, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
    result = img.copy()

# 经验参数
    minLineLength = 100
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(edges, (x1, y1), (x2, y2), (0, 255, 255), 5)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 5)
        cv2.imshow('Result', img)
        #cv2.imshow('Result', edges)


    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows()

