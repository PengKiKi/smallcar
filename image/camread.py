import cv2
import tensorflow as tf
import numpy as np
import urllib
import os
from io import BytesIO


url = r"http://192.168.0.122:10088/?action=snapshot"



while True:

    fileName = '1.jpg'
    save_path = r'D:\picture'
    path=urllib.request.urlretrieve(url, save_path + os.sep + fileName)
    #i = cv2.imshow('a', BytesIO(ht))

    im = cv2.imread(path[0],)
    cv2.imshow('image', im)
    k = cv2.waitKey(1)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
        break