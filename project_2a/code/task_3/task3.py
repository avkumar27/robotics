#input the file path
import cv2
import numpy as np
import glob
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import Task2
#import test.xml
import sys
class Spd(object):
    def __init__(self, filepath):
        self.read_Imagefile(filepath)


    def read_Imagefile(self, img_path):
        #img_path='/home/omkar/Desktop/Project2a/project_2a/images/task_2/*'
        img_path = '/home/avk/Desktop/project_2a/images/task_3_and_4/*'
        #img_path='../task_3_and_4/*'
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
        print("hi",cal_data,type(cal_data))


        R= np.genfromtxt('R.csv',delimiter=',')
        T= np.genfromtxt('T.csv',delimiter=',')
        mat_rat1=np.genfromtxt('M1.csv',delimiter=',')
        mat_rat2 = np.genfromtxt('M2.csv', delimiter=',')
        pose1= np.genfromtxt('pose1.csv', delimiter=',')
        pose2 = np.genfromtxt('pose2.csv', delimiter=',')
        E = np.genfromtxt('E.csv', delimiter=',')
        F = np.genfromtxt('F.csv', delimiter=',')







        for p,fPath in enumerate(image_right):

            #Load the image
            element_image_left = cv2.imread(image_left[p])
            element_image_right = cv2.imread(image_right[p])
            cv2.imshow('test', element_image_left)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

            #Chage color space to Gray -left
            gray_img_left = cv2.cvtColor(element_image_left, cv2.COLOR_BGR2GRAY)
            img_shape = gray_img_left.shape[::-1]

            #Chage color space to Gray -right
            gray_img_right = cv2.cvtColor(element_image_right, cv2.COLOR_BGR2GRAY)

            img1 = cv2.undistort(gray_img_left,mat_rat1, pose1)
            img2 = cv2.undistort(gray_img_right,mat_rat2, pose2)

            cv2.destroyAllWindows()

            # Feature detection
            # left_sample_img = cv2.imread('/home/avk/Desktop/project_2a/images/task_3_and_4/left_1.png')
            # right_sample_img = cv2.imread('/home/avk/Desktop/project_2a/images/task_3_and_4/right_1.png')
            # left_sample_img = cv2.imread('/left_1.png')
            # cv2.imshow('left_sample_image', left_sample_img)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
            # cv2.imshow('right_sample_image', right_sample_img)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            # Initiate STAR detector
            orb = cv2.ORB()
            orb2 = cv2.ORB()

            # find the keypoints with ORB
            kpl = orb.detect(img1, None)
            kpr = orb.detect(img1, None)

            # compute the descriptors with ORB
            kprs, desrs = orb.compute(gray_img_right, kpl)
            kpls, desls = orb.compute(gray_img_left, kpr)

            # draw only keypoints location,not size and orientation
            lkp = cv2.drawKeypoints(gray_img_left, kpl, color=(0, 255, 0), flags=0)
            plt.imshow(img2), plt.show()
            cv2.destroyAllWindows()

            rkp = cv2.drawKeypoints(gray_img_right, kpr, color=(0, 255, 0), flags=0)
            plt.imshow(img2), plt.show()
            cv2.destroyAllWindows()

            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors.
            matches = bf.match(des1, des2)

            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)

            # Draw first 10 matches.
            matchimg = cv2.drawMatches(gray_img_left, lkp, gray_img_right, rkp, matches[:10], flags=2)

            plt.imshow(matchimg), plt.show()

            # Find the tringulation points

            undistorted_left = cv2.undistortPoints(np.reshape(self.imagePoint_Left_subpix, (108, 1, 2)), mat_rat1, d1)

            undistorted_right = cv2.undistortPoints(np.reshape(self.imagePoint_Right_subpix, (108, 1, 2)), mat_rat2, d2)

            # Identity matrix
            identity_matrix = np.identity(3)
            rot_cam_1_cam_2 = np.dot(R, identity_matrix)
            id_transpose = np.transpose([[0, 0, 0]])
            final_transformation = np.dot(R, id_transpose) + T

            camera_pose_1 = np.asarray(
                [[identity_matrix[0][0], identity_matrix[0][1], identity_matrix[0][2], id_transpose[0][0]],
                 [identity_matrix[1][0], identity_matrix[1][1], identity_matrix[1][2], id_transpose[1][0]],
                 [identity_matrix[2][0], identity_matrix[2][1], identity_matrix[2][2], id_transpose[2][0]]])

            camera_pose_2 = np.asarray(
                [[rot_cam_1_cam_2[0][0], rot_cam_1_cam_2[0][1], rot_cam_1_cam_2[0][2], final_transformation[0][0]],
                 [rot_cam_1_cam_2[1][0], rot_cam_1_cam_2[1][1], rot_cam_1_cam_2[1][2], final_transformation[1][0]],
                 [rot_cam_1_cam_2[2][0], rot_cam_1_cam_2[2][1], rot_cam_1_cam_2[2][2], final_transformation[2][0]]])

            # print("-----------------------------------------------")
            # print(camera_pose_1)
            # print(camera_pose_2)
            # print("-----------------------------------------------")

            transpose_undistorted_left = np.transpose(np.reshape((undistorted_left), (108, 2)))
            transpose_undistorted_right = np.transpose(np.reshape((undistorted_right), (108, 2)))

            # Tringulate the points
            fourD_point_val = cv2.triangulatePoints(camera_pose_1, camera_pose_2, lkp, rkp)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Xs, Ys, Zs, c='r', marker='o')
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax.set_zlabel('X')
            plt.title('3D point cloud: Use pan axes button below to inspect')
            plt.show()














if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = Spd(args.filepath)
