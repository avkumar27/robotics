#input the file path
import cv2
import numpy as np
import glob
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
class StereoCalibrate(object):
    def __init__(self, filepath):
        self.read_Imagefile(filepath)


    def read_Imagefile(self, img_path):
        #img_path='/home/omkar/Desktop/Project2a/project_2a/images/task_2/*'
        img_path = '/home/avk/Desktop/project_2a/images/task_2/*'
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

        for p,fPath in enumerate(image_right):

            #Load the image
            element_image_left = cv2.imread(image_left[p])
            element_image_right = cv2.imread(image_right[p])

            #Chage color space to Gray -left
            gray_img_left = cv2.cvtColor(element_image_left, cv2.COLOR_BGR2GRAY)

            #Chage color space to Gray -right
            gray_img_right = cv2.cvtColor(element_image_right, cv2.COLOR_BGR2GRAY)

            #Find the chess board corners - left
            left_ret, left_corners = cv2.findChessboardCorners(gray_img_left, (9, 6), None)

            # Find the chess board corners - right
            right_ret, right_corners = cv2.findChessboardCorners(gray_img_right, (9, 6), None)

            #Append object points
            self.objectArray.append(self.objectpoint)

            if left_ret is True:
                rt= cv2.cornerSubPix(gray_img_left, left_corners, (11, 11),(-1, -1), self.criteria)

                self.imagePoint_Left.append(left_corners)
                self.imagePoint_Left_subpix.append(rt)

                #Draw the image with corner indication
                ret_l = cv2.drawChessboardCorners(element_image_left, (9, 6),left_corners, left_ret)

                cv2.imshow(image_left[p], element_image_left)

                cv2.waitKey(500)
                cv2.destroyAllWindows()


            if right_ret is True:
                rt = cv2.cornerSubPix(gray_img_right, right_corners, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgePoint_Right.append(right_corners)
                self.imagePoint_Right_subpix.append(rt)
                #Draw the right images
                ret_r= cv2.drawChessboardCorners(element_image_right, (9, 6),right_corners, right_ret)

                cv2.imshow(image_right[p], element_image_right)

                cv2.waitKey(500)
                cv2.destroyAllWindows()

            img_shape = gray_img_left.shape[::-1]

        #Calibrate camera by left images
        rt, self.mat1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objectArray, self.imagePoint_Left, img_shape, None, None)

        #Calibrate camera by right images
        rt, self.mat2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objectArray, self.imgePoint_Right, img_shape, None, None)

        constant_flag = 0

        constant_flag |= cv2.CALIB_FIX_INTRINSIC

        constant_flag |= cv2.CALIB_USE_INTRINSIC_GUESS

        constant_flag |= cv2.CALIB_FIX_FOCAL_LENGTH

        constant_flag |= cv2.CALIB_ZERO_TANGENT_DIST

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        ret, mat_rat1, d1, mat_rat2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objectArray, self.imagePoint_Left,
            self.imgePoint_Right, self.mat1, self.d1, self.mat2,
            self.d2, img_shape,
            criteria=stereocalib_criteria, flags=constant_flag)

        print('Intrinsic_mtx_1', mat_rat1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', mat_rat2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        p_camera_model_parameters=dict([('M1', mat_rat1), ('M2', mat_rat2), ('dist1', d1),('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])


        cv2.destroyAllWindows()

        #Find the tringulation points

        undistorted_left = cv2.undistortPoints(np.reshape(self.imagePoint_Left_subpix, (108, 1, 2)), mat_rat1, d1)

        undistorted_right = cv2.undistortPoints(np.reshape(self.imagePoint_Right_subpix, (108, 1, 2)), mat_rat2, d2)

        #Identity matrix
        identity_matrix = np.identity(3)
        rot_cam_1_cam_2 = np.dot(R, identity_matrix)
        id_transpose = np.transpose([[0, 0, 0]])
        final_transformation = np.dot(R, id_transpose) + T

        camera_pose_1 = np.asarray([[identity_matrix[0][0], identity_matrix[0][1], identity_matrix[0][2], id_transpose[0][0]],
                         [identity_matrix[1][0], identity_matrix[1][1], identity_matrix[1][2], id_transpose[1][0]],
                         [identity_matrix[2][0], identity_matrix[2][1], identity_matrix[2][2], id_transpose[2][0]]])


        camera_pose_2 = np.asarray([[rot_cam_1_cam_2[0][0], rot_cam_1_cam_2[0][1], rot_cam_1_cam_2[0][2], final_transformation[0][0]],
                         [rot_cam_1_cam_2[1][0], rot_cam_1_cam_2[1][1], rot_cam_1_cam_2[1][2], final_transformation[1][0]],
                         [rot_cam_1_cam_2[2][0], rot_cam_1_cam_2[2][1], rot_cam_1_cam_2[2][2], final_transformation[2][0]]])

       # print("-----------------------------------------------")
       # print(camera_pose_1)
        #print(camera_pose_2)
        #print("-----------------------------------------------")

        transpose_undistorted_left = np.transpose(np.reshape((undistorted_left), (108, 2)))
        transpose_undistorted_right = np.transpose(np.reshape((undistorted_right), (108, 2)))

        #Tringulate the points
        fourD_point_val = cv2.triangulatePoints(camera_pose_1, camera_pose_2, transpose_undistorted_left, transpose_undistorted_right)

        #define the rectification scale
        rec_scale = -1

        #use the sterio rectify
        rotation1, rotation2, pose1, pose2, Q, roi1, roi2 = cv2.stereoRectify(mat_rat1, d1, mat_rat2, d2, img_shape, R,
                                                          T, alpha=rec_scale)



        #Undistort and rectify the map
        map_left = cv2.initUndistortRectifyMap(mat_rat1, d1, rotation1, pose1, img_shape, cv2.CV_32FC1)
        map_right = cv2.initUndistortRectifyMap(mat_rat2, d2, rotation2, pose2, img_shape, cv2.CV_32FC1)

        #Show the images
        #left_sample_img = cv2.imread('/home/omkar/Desktop/Project2a/project_2a/images/task_2/left_1.png')
        #right_sample_img = cv2.imread('/home/omkar/Desktop/Project2a/project_2a/images/task_2/right_1.png')
        left_sample_img = cv2.imread('/home/avk/Desktop/project_2a/images/task_2/left_1.png')
        right_sample_img = cv2.imread('/home/avk/Desktop/project_2a/images/task_2/right_1.png')
        cv2.imshow('left_sample_image', left_sample_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.imshow('right_sample_image', right_sample_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        left_undistorted_img = cv2.undistort(left_sample_img, mat_rat1, d1)
        cv2.imshow('left_undistorted_image', left_undistorted_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        right_undistorted_img = cv2.undistort(right_sample_img, mat_rat2, d2)
        cv2.imshow('right_undistorted_image', right_undistorted_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


        #Remapping the images
        remap_left_image = cv2.remap(left_sample_img, map_left[0], map_left[1], cv2.INTER_LINEAR)
        remap_right_image = cv2.remap(right_sample_img, map_right[0], map_right[1], cv2.INTER_LINEAR)

        #Displaying the remapped images
        cv2.imshow('remapped left image :', remap_left_image)
        cv2.imwrite('remapped left image.jpg', remap_left_image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.imshow('remapped right image :', remap_right_image)
        cv2.imwrite('remapped right image.jpg', remap_right_image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # convert from homogeneous coordinates to 3D
        pts3D = fourD_point_val[:, :3] / np.repeat(fourD_point_val[:, 3], 3).reshape(-1, 3)

        # plot with matplotlib
        Ys = pts3D[:, 0]
        Zs = pts3D[:, 1]
        Xs = pts3D[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xs, Ys, Zs, c='r', marker='o')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_zlabel('X')
        plt.title('3D point cloud: Use pan axes button below to inspect')
        plt.show()


        #Saving the text in csv file
        R = np.asarray(R)
        np.savetxt('R.csv', R, delimiter=',', fmt='%f')
        T = np.asarray(T)
        np.savetxt('T.csv', T, delimiter=',', fmt='%f')
        E = np.asarray(E)
        np.savetxt('E.csv', E, delimiter=',', fmt='%f')
        F = np.asarray(F)
        np.savetxt('F.csv', F, delimiter=',', fmt='%f')
        P1 = np.asarray(pose1)
        np.savetxt('pose1.csv', pose1, delimiter=',', fmt='%f')
        P2 = np.asarray(pose2)
        np.savetxt('pose2.csv', pose2, delimiter=',', fmt='%f')


        return p_camera_model_parameters



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibrate(args.filepath)





