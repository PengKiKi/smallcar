import cv2
import tensorflow as tf
import numpy as np
import urllib
import os
import io
from PIL import Image
import math
from skimage import data, exposure, img_as_float
from skimage import morphology


linescopy=np.array([[[0, 0, 0, 0, ]], [[0, 0, 0, 0, ]]])
def nothing(x):
    pass

cv2.namedWindow('Track Bar')

cv2.createTrackbar('test1', 'Track Bar', 1, 255, nothing)
cv2.createTrackbar('test2', 'Track Bar', 0, 255, nothing)
cv2.createTrackbar('test3', 'Track Bar', 0, 255, nothing)
cv2.createTrackbar('test4', 'Track Bar', 0, 255, nothing)

flag1 = cv2.getTrackbarPos('test1', 'Track Bar')
flag2 = cv2.getTrackbarPos('test2', 'Track Bar')
flag3 = cv2.getTrackbarPos('test3', 'Track Bar')
flag4 = cv2.getTrackbarPos('test4', 'Track Bar')


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""


    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
def line_selct(lines, langle, rangle, y, yshape):
    global  linescopy
    new = []
    new1 = []
    k1 = 0
    k2 = 0
    b1 = 0
    b2 = 0

    for i in lines:

        k = (i[0][3] - i[0][1]) / (i[0][0] - i[0][2])

        if k > math.tan(langle / 180 * math.pi):

            new.append(i)
        elif k < math.tan(rangle / 180 * math.pi):
            new1.append(i)

    len1 = len(new)
    len2 = len(new1)
    if len1 == 0 or len2 == 0:
        return linescopy

    for i in new:
        k = (i[0][1] - i[0][3]) / (i[0][0] - i[0][2])
        k1 = k1 + k
        b = i[0][1] - k * i[0][0]
        b1 = b + b1
    k1 = k1 / len1
    b1 = b1 / len1
    for i in new1:
        k = (i[0][1] - i[0][3]) / (i[0][0] - i[0][2])
        k2 = k2 + k
        b = i[0][1] - k * i[0][0]
        b2 = b2 + b
    print(len2)
    k2 = k2 / len2
    b2 = b2 / len2

    if k1 == np.inf or k2 == np.inf or b1 == np.inf or b2 == np.inf:
        return linescopy

    x11 = int((img.shape[1] - b1) / k1)
    x12 = int((y - b1) / k1)

    x21 = int((img.shape[1] - b2) / k2)
    x22 = int((y - b2) / k2)

    linescopy = np.array([[[x11, yshape, x12, y, ]], [[x21, yshape, x22, y, ]]])

    return linescopy
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    global flag1
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    if lines==None:
        lines = np.array([])
    result = line_selct(lines, 30, 180-30, 100, img.shape[1])
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, result)

    return line_img
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale


    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray=img

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray=img
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale

    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray=img

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]/180.0*np.pi) & (absgraddir <= thresh[1]/180.0*np.pi)] = 1

    # Return the binary image
    return binary_output


def mysobel(image):
    global flag1, flag2, flag3, flag4
    ksize = 5  # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(9, 140))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(9, 140))
    mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(0, 255))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(26, 65))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined = np.array(combined, dtype='uint8')



    return combined


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float64)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    h_channel = hsv[:, :, 0]

    h_binary = np.zeros_like(s_channel)
    h_binary[(s_channel >= 50) & (s_channel <= 100)] = 255


    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_sobel=mysobel(l_channel)
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.zeros_like(s_channel)
    color_binary [s_binary==sxbinary]=255
    #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binary = np.array(color_binary, dtype='uint8')
    return color_binary





def process_image(img):
    global flag1,flag2,flag3,flag4
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    temp = img

    img = gaussian_blur(img, kernel_size=3)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > 5) & (s_channel <= 80)] = 255

    #test=mysobel(img)
    img=pipeline(img,(30,36),(1,5))

    cv2.imshow('sobel and color', img)
    #img = grayscale(img)

    img = canny(img, low_threshold=0, high_threshold=204)

    cv2.imshow('canny', img)

    imshape = img.shape
    img = region_of_interest(img, vertices=np.array([[(0, 282), (20,100), (397,100), (515,364)]],dtype=np.int32))
    #result=img
    cv2.imshow('ROI', img)
    line_img = hough_lines(img, rho=1, theta=np.pi / 180, threshold=52, min_line_len=150, max_line_gap=205)

    cv2.imshow('line', line_img)
    img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    temp2=weighted_img(line_img, img, α=0.7, β=500., λ=.1)
    cv2.imshow('ROI2', temp2)

    result = weighted_img(line_img, temp, α=0.7, β=500., λ=.1)

    return result




while True:

    flag1 = cv2.getTrackbarPos('test1', 'Track Bar')
    flag2 = cv2.getTrackbarPos('test2', 'Track Bar')
    flag3 = cv2.getTrackbarPos('test3', 'Track Bar')
    flag4 = cv2.getTrackbarPos('test4', 'Track Bar')

    img = cv2.imread("Frame1.jpg")

    cv2.imshow('myself', img)
    img = exposure.adjust_gamma(img, 0.4)

    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:, :, flag4]
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= flag1) & (h_channel <= flag2)] = 255
    h_binary = np.array(h_binary, dtype='uint8')
    masked_img = cv2.bitwise_and(img,img,mask=h_binary)
    '''

    cv2.imshow('gamma', img)

    cv2.imshow('Track Bar', process_image(img))






    k = cv2.waitKey(1)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
        break