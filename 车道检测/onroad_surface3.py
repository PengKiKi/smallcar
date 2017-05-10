from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import urllib.request
import glob
import io
import json
from moviepy.editor import VideoFileClip
from PIL import Image

moveup=50

moveup2=20.
wider=0.
controld=1

flag1=0
flag2=0
flag3=0
flag4=0

def nothing(x):
    pass
if controld:
    cv2.namedWindow('Track Bar')
    cv2.resizeWindow('Track Bar', 300, 170)
    cv2.createTrackbar('moveup', 'Track Bar', 1, 255, nothing)
    cv2.createTrackbar('wide', 'Track Bar', 0, 255, nothing)
    cv2.createTrackbar('test3', 'Track Bar', 0, 255, nothing)
    cv2.createTrackbar('test4', 'Track Bar', 0, 255, nothing)
sizefactor=1
orgw=0
orgh=0
linescopy=[]

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
def gamma_trans(img, gamma):
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)
def prep(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        NpKernel = np.uint8(np.ones((3,3)))
        dilated = cv2.dilate(img, kernel)
        Nperoded = cv2.erode(dilated, NpKernel)
        return Nperoded



class LaneDetector:
    """
    My class for lane detetion.
    """
    is_distortion_saved = False
    font = cv2.FONT_HERSHEY_SIMPLEX

    N_frames = 25
    left_fit_deque = deque()
    right_fit_deque = deque()

    retry_count = 25
    retry_counter = 0


    def canny(self,img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)
    def gaussian_blur(self,img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    def line_selct(self,lines, langle, rangle, y, yshape):
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

    def draw_lines(self,img, lines, color=[255, 0, 0], thickness=10):
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
    def hough_lines(self,img, rho, theta, threshold, min_line_len, max_line_gap):
        global flag1
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """

        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        if lines==None:
            lines = np.array([])
        result = self.line_selct(lines, 30, 180-30, 0, img.shape[1])
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self.draw_lines(line_img, result)

        return line_img
    def weighted_img(self,img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)

    def abs_sobel_thresh(self,img, orient='x', sobel_kernel=3, thresh=(0, 255)):
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
    def mag_thresh(self,img, sobel_kernel=3, mag_thresh=(0, 255)):
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
    def dir_threshold(self,img, sobel_kernel=3, thresh=(0, np.pi/2)):
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

    def mysobel(self,image):
        global flag1, flag2, flag3, flag4
        ksize = 5  # Choose a larger odd number to smooth gradient measurements
        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(9, 140))
        grady = self.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
        mag_binary = self.mag_thresh(image, sobel_kernel=9, mag_thresh=(0, 255))
        dir_binary = self.dir_threshold(image, sobel_kernel=15, thresh=(0, 255))
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 255
        combined = np.array(combined, dtype='uint8')



        return combined


    def pipeline(self,img, s_thresh=(170, 255), sx_thresh=(20, 100)):
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
        scaled_sobel=self.mysobel(l_channel)
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




    def mask_lane_lines(self, img):
        global flag4,flag3,flag2




        img = self.gaussian_blur(img, kernel_size=3)

        #cv2.normalize(img,img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        '''
        hist_b = cv2.calcHist(img, [0], None, [256], [0, 256])
        hist_g = cv2.calcHist(img, [1], None, [256], [0, 256])
        hist_r = cv2.calcHist(img, [2], None, [256], [0, 256])
        '''

        img = gamma_trans(img, 0.5)
        cv2.imshow('equ before',img)



        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        img_whty = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        img_whty = img_whty[:, :, 1]
        img_whty[img_whty < 230] = 0
        mask_whty = cv2.inRange(img_whty, 230, 255)
        cv2.imshow('mask_whty',mask_whty)

        img_wht=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        img_wht=self.mysobel(img_wht)
        cv2.imshow('equ after',img_wht)
        '''
        NpKernel = np.uint8(np.zeros((5,5)))
        for i in range(5):
            NpKernel[2, i] = 1
            NpKernel[i, 2] = 1
        '''
        mask_whty=prep(mask_whty)
        img_wht=prep(img_wht)


        mask=np.zeros_like(img_wht)
        mask[((mask_whty==255) & (img_wht==255))]=255

        cv2.imshow('combine',mask)


        return mask

    '''
    def mask_lane_lines(self, img):

        global flag4,flag3,flag2

        img = np.copy(img)

        # Blur
        kernel = np.ones((5, 5), np.float32) / 25
        img = cv2.filter2D(img, -1, kernel)

        # YUV for histogram equalization
        cv2.imshow('equ before',img)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        img_wht = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


        # Compute white mask
        img_wht = img_wht[:, :, 1]
        #img_wht[img_wht < 250] = 0
        mask_wht = cv2.inRange(img_wht, flag4, 255)
        cv2.imshow('mask_wht',mask_wht)

        yuv[:, :, 0:1] = 0

        # Yellow mask
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(yuv, -1, kernel)
        sobelx = np.absolute(cv2.Sobel(yuv[:, :, 2], cv2.CV_64F, 1, 0, ksize=5))
        sobelx[sobelx < flag2] = 0
        sobelx[sobelx >= flag2] = 255

        cv2.imshow('sobelx',sobelx)
        #sobelx=self.mysobel(yuv[:, :, 2])

        #cv2.imshow('sobelx',(self.abs_sobel_thresh(yuv[:, :, 2],thresh=(0, flag4)))*255)
        # Merge mask results
        mask=np.zeros_like(mask_wht)
        #mask = mask_wht + sobelx

        mask[((mask_wht==255) & (sobelx==255))]=255
        #cv2.imshow('equ output',np.array(mask,dtype='uint8'))

        #img = self.canny(np.array(mask,dtype='uint8'), low_threshold=0, high_threshold=255)
        #line_img = self.hough_lines(img, rho=1, theta=np.pi / 180, threshold=15, min_line_len=flag3, max_line_gap=flag4)
        #img = np.dstack((img, img, img))
        #print(img)
        #cv2.imshow('equ output',cv2.addWeighted(line_img, 0.5, img, 1, 0))
        cv2.imshow('equ output',mask)


        return mask
    '''

    def findLinePoints(self, image):
        """
        Find lane lines points in a single image - slow.
        """
        #cv2.imshow('input', image)
        points = ([], [])
        lt=0
        rt=0
        shape = image.shape
        img = image.copy()
        # Prepare images for visualization
        red = np.zeros_like(img)
        green = np.zeros_like(img)
        blue = np.zeros_like(img)

        # Set center to width/2
        center = int(shape[1] / 2)

        # For each row starting from bottom

        self.leftcp=[]
        self.rightcp=[]

        for yy, line in list(enumerate(image))[::-1]:
            x_val_hist = []
            counter = 0
            for x in line:
                if x > 0:
                    x_val_hist.append(counter)
                counter = counter + 1
            if len(x_val_hist) > 0:
                cv2.circle(green, (int(center), yy), 1, (255, 255, 255))

                # Split to left/right line
                left = [(x, yy) for x in x_val_hist if x < center]
                right = [(x, yy) for x in x_val_hist if x >= center]

                if len(left)>0:
                    self.leftcp=left
                else:
                    left=self.leftcp
                if len(right)>0:
                    self.rightcp=right
                else:
                    right=self.rightcp

                if len(left) > 0:
                    # Compute average
                    #leftcp=left
                    l = np.mean(np.array(left), axis=0)
                    l = (l[0], l[1])
                    lt=l[0]
                    #center = l[0] + int(shape[1] * 0.2)
                    cv2.circle(red, (int(l[0]), int(l[1])), 1, (255, 255, 255))
                    # Add to points
                    points[0].append(l)

                if len(right) > 0:
                    # Compute average
                    r = np.mean(np.array(right), axis=0)
                    r = (r[0], r[1])
                    rt=r[0]
                    cv2.circle(blue, (int(r[0]), int(r[1])), 1, (255, 255, 255))
                    # Add to points
                    points[1].append(r)
                center = (rt+lt)/2



        if True:  # for debug
            img = cv2.resize(np.dstack((blue, green, red)), (shape[1], int(shape[0])), fx=0, fy=0)
            cv2.imshow('lines', img)
        return points

    def findLinePointsFast(self, image):
        """
        Find lane lines points in video frame - fast.
        Works starting from second frame.
        """
        points = ([], [])
        shape = image.shape
        # Prepare images for visualization

        img = image.copy()
        red = np.zeros_like(img)
        blue = np.zeros_like(img)

        # For every 10th row starting from bottom
        for y, lx, rx, line in list(zip(self.all_y, self.left_fitx, self.right_fitx, image))[::-10]:
            lxmin = int(lx - 20 - 0.2 * (img.shape[0] - y))
            lxmax = int(lx + 20 + 0.2 * (img.shape[0] - y))
            rxmin = int(rx - 20 - 0.2 * (img.shape[0] - y))
            rxmax = int(rx + 20 + 0.2 * (img.shape[0] - y))
            cv2.circle(red, (lxmin, int(y)), 1, (255, 255, 255))
            cv2.circle(red, (lxmax, int(y)), 1, (255, 255, 255))
            cv2.circle(red, (rxmin, int(y)), 1, (255, 255, 255))
            cv2.circle(red, (rxmax, int(y)), 1, (255, 255, 255))
            x_val_hist = []
            counter = 0
            for x in line:
                if x > 0:
                    x_val_hist.append(counter)
                counter += 1
            if len(x_val_hist) > 5:
                # split points to left/right
                left = [(x, y) for x in x_val_hist if x <= lxmax and x >= lxmin]
                right = [(x, y) for x in x_val_hist if x >= rxmin and x <= rxmax]
                l = None
                r = None
                # Compute means for left/right
                if len(left):
                    l = np.mean(np.array(left), axis=0)
                if len(right):
                    r = np.mean(np.array(right), axis=0)
                if l is None or r is None or r[0] - l[0] > 200:
                    if (not l is None) and l[0] > lxmin and l[0] < lxmax:
                        cv2.circle(blue, (int(l[0]), int(l[1])), 1, (255, 255, 255))
                        points[0].append(l)
                    if (not r is None) and r[0] > rxmin and r[0] < rxmax:
                        cv2.circle(blue, (int(r[0]), int(r[1])), 1, (255, 255, 255))
                        points[1].append(r)

        if len(points[0]) < 10 or len(points[1]) < 10:
            self.retry_counter += 1

        # Show roi for video frame
        img = cv2.resize(np.dstack((blue, red, red)), (shape[1], int(shape[0])), fx=0, fy=0)
        cv2.imshow('lines-video', img)
        return points

    def processVideo(self, fname):
        '''
        Process video data from file and save result to results dir
        '''
        print('Video:', fname)
        self.retry_counter = 0
        # Open video
        cap = cv2.VideoCapture(fname)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        if fname>=3:
            video_out = cv2.VideoWriter('results/' + fname.split('/')[-1], fourcc, 25, (1280, 720))

        if cap.isOpened():
            ret, img = cap.read()
            # Process first image as single image
            if not img is None:
                res = self.processSingleImage(img)
                cv2.imshow('Result', res)
                cv2.waitKey(1)

        while (cap.isOpened()):
            ret, img = cap.read()
            if img is None:
                break
            if self.retry_counter < self.retry_count:
                res = self.processNextImage(img)
            else:
                self.retry_counter = 0
                print('process as single image')
                res = self.processSingleImage(img)
            # write result
            if fname>=3:
                video_out.write(res)
            # Show result
            cv2.imshow('Result', res)
            cv2.waitKey(1)
        cap.release()
        if fname>=3:
            video_out.release()
        cv2.destroyAllWindows()

    def undistort(self, img):
        '''
        Undistort image
        '''
        return img
        # return cv2.undistort(img,calib['matrix'], calib['dist'], None,None)

    def unwarp(self, undist):
        '''
        Unwarp image
        '''
        global orgh,orgw,flag3,flag4

        warpfactor=3

        cv2.imshow('undist',undist)
        src = np.float32(self.roi_corners)

        offset = 138
        warped_size = (int((orgw + 2 * offset)/(warpfactor*2)), int(undist.shape[1]/warpfactor))

        dst = np.float32([
            [0, warped_size[1]],
            [0, 0],
            [0 + warped_size[0], 0],
            [0 + warped_size[0], warped_size[1]]])

        self.Mpersp = cv2.getPerspectiveTransform(src, dst)
        warped_orig = cv2.warpPerspective(undist, self.Mpersp, dsize=warped_size)

        return warped_orig

    def removeTopBottom(self, img):
        '''
        Remove (set to black) top and bottom lines of image
        '''
        res = img.copy()
        # remove car
        #res[-int(res.shape[0] - self.roi_corners[0][1]):-1, :, :] = 0

        #res[-int(res.shape[0] * 0.1):-1, :, :] = 0
        # remove sky
        res[0:int(self.roi_corners[1][1]), :, :] = 0

        return res

    def computeAndShow(self, img, warped):
        '''
        Compute parameters:
        - left line curvature
        - right line curvature
        - distance from center

        And render result
        '''
        # Define y-value where we want radius of curvature
        y_eval = np.max(self.all_y)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 60 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(np.array(self.tl_y) * ym_per_pix, np.array(self.tl_x) * xm_per_pix, 2)
        right_fit_cr = np.polyfit(np.array(self.tr_y) * ym_per_pix, np.array(self.tr_x) * xm_per_pix, 2)

        # Calculate the new radii of curvature
        self.left_curverad = ((1 + (
        2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        self.right_curverad = ((1 + (
        2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        # Distance from center
        img_center = warped.shape[1] / 2
        lane_center = (self.left_fitx[-1] + self.right_fitx[-1]) * 0.5
        diff = lane_center - img_center
        self.diffm = diff * xm_per_pix

        img = cv2.putText(img, 'Pengkiki', (10, 25), self.font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, 'Curvature left: %.1f m' % (self.left_curverad), (30, 60), self.font, 0.8,
                          (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, 'Curvature right: %.1f m' % (self.right_curverad), (30, 100), self.font, 0.8,
                          (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, 'Dist from center: %.2f m' % (self.diffm), (30, 140), self.font, 0.8, (255, 255, 255), 2,
                          cv2.LINE_AA)

    def fitAndShow(self, warped, undist, points):
        """
        Fit points and show result on image
        """
        if len(points[0]) >= 5 and len(points[1]) >= 5:
            leftx, lefty = zip(*points[0])
            rightx, righty = zip(*points[1])

            self.all_y = np.array(list(range(warped.shape[0])))

            self.tl_x = list(leftx)
            self.tl_y = list(lefty)
            self.tr_x = list(rightx)
            self.tr_y = list(righty)

            # convert to numpy
            self.tl_x = np.array(self.tl_x)
            self.tl_y = np.array(self.tl_y)
            self.tr_x = np.array(self.tr_x)
            self.tr_y = np.array(self.tr_y)

            # Fit a second order polynomial to each fake lane line
            left_fit = np.array(np.polyfit(self.tl_y, self.tl_x, 2))
            right_fit = np.array(np.polyfit(self.tr_y, self.tr_x, 2))

            self.left_fit_deque.append(left_fit)
            if len(self.left_fit_deque) >= self.N_frames:
                self.left_fit_deque.popleft()
            self.right_fit_deque.append(right_fit)
            if len(self.right_fit_deque) >= self.N_frames:
                self.right_fit_deque.popleft()

            self.left_fit = np.array([0, 0, 0])
            for v in self.left_fit_deque:
                self.left_fit = self.left_fit + v
            self.left_fit = self.left_fit / len(self.left_fit_deque)
            self.right_fit = np.array([0, 0, 0])
            for v in self.right_fit_deque:
                self.right_fit = self.right_fit + v
            self.right_fit = self.right_fit / len(self.right_fit_deque)

            self.left_fitx = np.array(
                self.left_fit[0] * self.all_y ** 2 + self.left_fit[1] * self.all_y + self.left_fit[2])
            self.right_fitx = np.array(
                self.right_fit[0] * self.all_y ** 2 + self.right_fit[1] * self.all_y + self.right_fit[2])

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.all_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.all_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = np.linalg.inv(self.Mpersp)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.2, 0)

        unwarped = 255 * cv2.warpPerspective(warped, Minv, (undist.shape[1], undist.shape[0])).astype(np.uint8)
        unwarped = np.dstack((np.zeros_like(unwarped), np.zeros_like(unwarped), unwarped))

        line_image = cv2.addWeighted(result, 1, unwarped, 1, 0)
        return result, line_image

    def processNextImage(self, img):
        '''
        Process image using information from previous one to speed up on video.
        '''
        undist = self.undistort(img)
        undist_masked = self.removeTopBottom(undist)
        warped = self.unwarp(undist_masked)
        warped = self.mask_lane_lines(warped)
        points = self.findLinePointsFast(warped)
        # cv2.imshow('warped',warped)
        result, _ = self.fitAndShow(warped, undist, points)
        self.computeAndShow(result, warped)

        return result

    def processSingleImage(self, img):
        global moveup,wider,flag4
        '''
        Process single image or first frame of video.
        '''
        self.retry_counter = 0
        self.left_fit_deque.clear()
        self.right_fit_deque.clear()
        self.moveup = 0

        undist = self.undistort(img)
        #cv2.imshow('undist_masked',undist)
        wider2=0.1
        roimove=wider/100
        roidownw=30/255


        self.roi_corners = [[(0.16-wider2-roimove-roidownw) * undist.shape[1], (1 - self.moveup) * undist.shape[0]],
                            [(0.45-wider2-roimove+0.04) * undist.shape[1], (0.63 - self.moveup) * undist.shape[0]],
                            [(0.55+wider2-roimove-0.03) * undist.shape[1], (0.63 - self.moveup) * undist.shape[0]],
                            [(0.84+wider2-roimove+roidownw) * undist.shape[1], (1 - self.moveup) * undist.shape[0]]]



        # remove top and bottom (sky, car)
        undist_masked = self.removeTopBottom(undist)
        #cv2.imshow('undist_masked',undist_masked)

        # Save before and after undistortion
        if not self.is_distortion_saved:
            dist_before_after = np.concatenate((img, undist), axis=1)
            dist_before_after = cv2.putText(dist_before_after, 'Distorted', (50, 50), self.font, 1, (255, 255, 255), 2,
                                            cv2.LINE_AA)
            dist_before_after = cv2.putText(dist_before_after, 'Undistorted', (50 + img.shape[1], 50), self.font, 1,
                                            (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imwrite('images/distortion.jpg', dist_before_after)
            self.is_distortion_saved = True

        warped = self.unwarp(undist_masked)
        #cv2.imshow('transformed',warped)
        warped = self.mask_lane_lines(warped)
        # cv2.imshow('binary',warped)
        points = self.findLinePoints(warped)
        # cv2.imshow('warped',warped)

        if len(points[0]) == 0 or len(points[1]) == 0:
            return img

        result, _ = self.fitAndShow(warped, undist, points)
        self.computeAndShow(result, warped)

        show_roi = True
        if show_roi:
            src = np.float32(self.roi_corners)

            pts = np.array(src, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result, [pts], True, (0, 0, 255))
            # cv2.imshow('roi',undist)

        return result


det = LaneDetector()

'''
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(det.processSingleImage)
white_clip.write_videofile(white_output, audio=False)
'''
cap = cv2.VideoCapture("File_000.mov")
#cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (0, 0))

while True:

    # img = cv2.imread(fimg)
    # res = det.processSingleImage(img)
    # cv2.imshow('Result', res)

    if controld:
        flag1 = cv2.getTrackbarPos('moveup', 'Track Bar')
        flag2 = cv2.getTrackbarPos('wide', 'Track Bar')
        flag3 = cv2.getTrackbarPos('test3', 'Track Bar')
        flag4 = cv2.getTrackbarPos('test4', 'Track Bar')
        moveup2=flag1
        wider= 0

        #flag3=138

    ret, img = cap.read()

    # img = cv2.resize(img, (960, 540))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.flip(img, 1)

    #res = det.processVideo(0)
    img = cv2.resize(img, (0, 0),fx=0.4,fy=0.4)
    img = cv2.transpose(img)
    img = cv2.flip(img, 1)
    high = img.shape[0]

    moveup2=35
    img=img[int(img.shape[0] * (30/100)):int(img.shape[0] * (1-moveup2/100)), :, :]

    #img[int(img.shape[0]-50):int(img.shape[0]), :, :]=0

    #print (img.shape)
    orgh,orgw,dump = img.shape

    #img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    res = det.processSingleImage(img)
    #res = cv2.resize(res, (0,0), fx=2, fy=2)
    # write frame

    #out.write(res)

    cv2.imshow('Result', res)

    k = cv2.waitKey(1)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
        break

cap.release()
