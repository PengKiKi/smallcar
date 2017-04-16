import cv2
import tensorflow as tf
import numpy as np
import urllib
import os
import io
from PIL import Image



url = r"http://192.168.0.122:10088/?action=snapshot"

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

def nothing(x):
    pass
cv2.namedWindow('Track Bar')
cv2.namedWindow('Track Bar2')

cv2.createTrackbar('test1', 'Track Bar', 0, 255, nothing)
cv2.createTrackbar('test2', 'Track Bar', 0, 255, nothing)
cv2.createTrackbar('test3', 'Track Bar', 0, 255, nothing)

cv2.createTrackbar('test4', 'Track Bar2', 9, 255, nothing)

while True:
    flag1 = cv2.getTrackbarPos('test1', 'Track Bar')
    flag2 = cv2.getTrackbarPos('test2', 'Track Bar')
    flag3 = cv2.getTrackbarPos('test3', 'Track Bar')

    flag4 = cv2.getTrackbarPos('test4', 'Track Bar2')

    fd = urllib.request.urlopen(url)
    image_file = io.BytesIO(fd.read())
    im = Image.open(image_file)

    img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 1)


    im = cv2.GaussianBlur(img, (flag4*2+1, flag4*2+1), flag2)
    #im = cv2.bilateralFilter(img, flag4, 75, 75)

    canny = cv2.Canny(im, flag1, 3*flag2)

    #img2 = img[200:300, 200:300]
    #cv2.rectangle(img2, (0, 0), (10, 10), (0, 255, 0), 2)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = np.float32(gray)

    '''
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        cv2.imshow('roi_color', roi_color)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            print ('(',ex, ey,')', ex + ew, ey + eh)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    #sift = cv2.xfeatures2d.SIFT_create()
    #(kps, descs) = sift.detectAndCompute(gray, None)
    #img = cv2.drawKeypoints(img, kps,img)
    '''

    cv2.imshow('Track Bar', canny)
    cv2.imshow('Track Bar2', im)

    cv2.imshow('myself',img)

    k = cv2.waitKey(1)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
        break