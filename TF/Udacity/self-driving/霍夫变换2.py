# coding=utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt

cap=cv2.VideoCapture('solidWhiteRight.mp4')

while True:
    ret, img = cap.read()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    rows, cols, ch = img.shape

    pts1 = np.float32([[329, 244], [994, 244], [225, 372], [1099, 372]])
    pts2 = np.float32([[300, 0], [600, 0], [300, 300], [600, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (900,800))

    plt.subplot(131), plt.imshow(img), plt.title('Input')
    plt.subplot(133), plt.imshow(dst), plt.title('Output')
    #plt.show()


    img1 = img.copy()
    img2 = img.copy()

    gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
    #retval, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    img1 = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(img1, 50, 150, apertureSize=3)
    lines2 = cv2.HoughLines(edges, 1, np.pi / 180, 118)
    result = img1.copy()

    # 经验参数
    minLineLength = 2
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 4)

    for line in lines2[0]:
        rho = line[0] #第一个元素是距离rho
        theta = line[1] #第二个元素是角度theta
        print(rho)
        print(theta)
        if (theta < (np.pi/4.)) or (theta > (3.*np.pi/4.0)):  #垂直直线
                 #该直线与第一行的交点
            pt1 = (int(rho/np.cos(theta)),0)
            #该直线与最后一行的焦点
            pt2 = (int((rho-edges.shape[0]*np.sin(theta))/np.cos(theta)),edges.shape[0])
            #绘制一条白线
            cv2.line( edges, pt1, pt2, (128),10)
        else: #水平直线
            # 该直线与第一列的交点
            pt1 = (0,int(rho/np.sin(theta)))
            #该直线与最后一列的交点
            pt2 = (edges.shape[1], int((rho-edges.shape[1]*np.cos(theta))/np.sin(theta)))
            #绘制一条直线
            cv2.line(edges, pt1, pt2, (128), 10)


    plt.subplot(221), plt.imshow(img), plt.title('Input')
    plt.subplot(222), plt.imshow(dst), plt.title('Output')
    plt.subplot(223), plt.imshow(edges), plt.title('canny')
    plt.subplot(224), plt.imshow(img2), plt.title('changed')

    plt.show()
    key = cv2.waitKey(10)
    if key == 27:
        break
    cv2.destroyAllWindows()
