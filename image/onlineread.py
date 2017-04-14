import cv2
import tensorflow as tf
import numpy as np
import urllib
import os
import io
from PIL import Image



url = r"http://192.168.0.122:10088/?action=snapshot"



while True:


    fd = urllib.request.urlopen(url)
    image_file = io.BytesIO(fd.read())
    im = Image.open(image_file)

    opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    cv2.imshow('a',opencvImage)

    k = cv2.waitKey(1)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
        break