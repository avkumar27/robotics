#input the file path
import cv2
import numpy as np
import glob
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import Task2
class Spd(object):
    def __init__(self, filepath):
        self.read_Imagefile(filepath)


    def read_Imagefile(self, img_path):
        #img_path='/home/omkar/Desktop/Project2a/project_2a/images/task_2/*'
        img_path = '/home/avk/Desktop/project_2a/images/task_3_and_4/*'
        #Add termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        #Prepare the object points
        self.objectpoint = np.zeros((9*6, 3), np.float32)
        self.objectpoint[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        #Array for object point
        self.objectArray=[]

        #Array for image points - left image points
        self.imagePoint_Left=[]

        #Array for image points - right image points
        self.imgePoint_Right=[]

        #subpixel for right and left
        self.imagePoint_Left_subpix=[]
        self.imagePoint_Right_subpix=[]


        #Fetch the all images from right camera
        #/home/omkar/Desktop/Project2a/project_2a/images
        image_right = glob.glob(img_path+'right*.png')

        #Fetch the all images from left camera
        image_left = glob.glob(img_path+'left*.png')

        #sort the images
        image_right.sort()
        image_left.sort()
        parser = Task2.argparse.ArgumentParser()
        parser.add_argument('filepath', help='String Filepath')
        args = parser.parse_args()
        cal_data = Task2.StereoCalibrate(args.filepath)

        for p,fPath in enumerate(image_right):

            #Load the image
            element_image_left = cv2.imread(image_left[p])
            element_image_right = cv2.imread(image_right[p])

            #Chage color space to Gray -left
            gray_img_left = cv2.cvtColor(element_image_left, cv2.COLOR_BGR2GRAY)

            #Chage color space to Gray -right
            gray_img_right = cv2.cvtColor(element_image_right, cv2.COLOR_BGR2GRAY)

            #
            stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
            disparity = stereo.compute(gray_img_left, gray_img_right)
            plt.imshow(disparity, 'gray')
            plt.show()




        Q2 = np.float32([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 2, 0],  # Focal length multiplication obtained experimentally.
                         [0, 0, 0, 1]])  # Reproject points into 3D
        points_3D = cv2.reprojectImageTo3D(disparity, Q2)




parser = argparse.ArgumentParser()
parser.add_argument('filepath', help='String Filepath')
args = parser.parse_args()
cal_data = Spd(args.filepath)