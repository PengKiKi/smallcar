# coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




cap=cv2.VideoCapture('solidWhiteRight.mp4 ')
while True:

    ret,image=cap.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    gray = cv2.GaussianBlur(gray, (5, 5),0)
    ret2, gray = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    edges = cv2.Canny(gray, 250, 50, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
    result = image.copy()

    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    imshape = image.shape
    #vertices = np.array([[(50, imshape[0]-50), (425, 325), (525, 325), (900, imshape[0]-50)]], dtype=np.int32)
    vertices = np.array([[(390, 590), (560, 460), (780, 460), (1000, 590)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 70  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    color_edges = np.dstack((edges, edges, edges))
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    #lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    pts1 = np.float32([[468, 316], [500, 316], [145, 540], [827, 540]])
    pts2 = np.float32([[145, 316], [827, 316], [145, 540], [827, 540]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(lines_edges, M, (960,540))

    cv2.imshow("Result", lines_edges)
    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows()

