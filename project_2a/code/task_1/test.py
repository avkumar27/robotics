import cv2
import numpy as np
import glob

image_side ='left'
def load_Image(image_side):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    object_point = np.zeros((9*6, 3), np.float32)
    object_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    #img_location='/home/omkar/Desktop/Project2a/project_2a/images/task_1/*'+image_side+'*.png'
    #img_location_zero='/home/omkar/Desktop/Project2a/project_2a/images/task_1/'+image_side+'_0.png'
    img_location = '/home/avk/Desktop//project_2a/images/task_1/*' + image_side + '*.png'
    img_location_zero = '/home/avk/Desktop/project_2a/images/task_1/' + image_side + '_0.png'
    #/ home / avk / Downloads / project_2a(1) / project_2a
    images= glob.glob(img_location)
    for fname in images:
        img_element=cv2.imread(fname)
        gray = cv2.cvtColor(img_element, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(object_point)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img_element, (9, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)


    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret)
    print(mtx)
    cv_file = cv2.FileStorage("test.xml", cv2.FILE_STORAGE_WRITE)
   # matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # this corresponds to a key value pair, internally opencv takes your numpy
    # object and transforms it into a matrix just like you would do with <<
    # in c++
    cv_file.write("my_matrix", mtx)
    cv_file.write("my_matrix", ret)

    # note you *release* you don't close() a FileStorage object
    cv_file.release()
    img_zero = cv2.imread(img_location_zero)
    h, w = img_zero.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    #dst = cv2.undistort(img_zero, mtx, dist, None, newcameramtx)

    #Rectification
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img_zero, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('calibresult.png', dst)
    image = cv2.imread('calibresult.png')
    cv2.imshow('calibresult.png', image)
    cv2.waitKey(1000)




load_Image(image_side)