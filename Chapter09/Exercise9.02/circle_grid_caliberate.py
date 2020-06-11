import imutils
import cv2
import numpy as np
import glob
import random 
from random import randint, sample 
import ctypes
import time

stopping_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

testgridize=1400
itemsperrow=15
itemspercolumn=4
radius=10
distancebetweencircles=36


color = (255,255,255)
paddingx=300
paddingy=300
scalefactor=2
base_folder = r'C:\Users\opencv_book\\' 
image_model = np.zeros((testgridize, testgridize, 3), dtype = "uint8") 
objp = np.zeros((itemsperrow*itemspercolumn, 3), np.float32)
currpos=0
delta = int(testgridize/itemsperrow)


for _x in range(itemsperrow):
    for _y in range(itemspercolumn):
        if _x % 2  == 1:
            shiftx=int(distancebetweencircles/2)
        else:
            shiftx=0
        image_model = cv2.circle(image_model, (paddingx+ scalefactor*_x*distancebetweencircles,scalefactor*shiftx + paddingy+scalefactor*_y*distancebetweencircles), scalefactor*radius, color, -1)
        objp[currpos] = (_x*distancebetweencircles,shiftx + _y*distancebetweencircles*2,0)
        currpos+=1

image_model =  cv2.cvtColor(image_model, cv2.COLOR_BGR2GRAY)
image_model = cv2.bitwise_not(image_model)


objp  = np.asarray(objp)

parameters = cv2.SimpleBlobDetector_Params()

# Change thresholds
parameters.minThreshold = 10
parameters.maxThreshold = 200
# Filter by Area
parameters.filterByArea = True
parameters.minArea = 1500
# Filter by Circularity
parameters.filterByCircularity = True
parameters.minCircularity = 0.1

# Filter by Convexityss
parameters.filterByConvexity = True
parameters .minConvexity = 0.87


# Filter by Inertia
parameters.filterByInertia = True
parameters.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(parameters)


objpoints = []
imgpoints = []

images = glob.glob(base_folder + 'image\\'  + "*.jpg")

found = 0
for image in images:
     rotated_image = cv2.imread(image, cv2.COLOR_BGR2GRAY)
     rotated_image   = cv2.resize(rotated_image , image_model.shape, interpolation = cv2.INTER_AREA)
     keypoints = detector.detect(rotated_image)
     image_with_keypoints = cv2.drawKeypoints(rotated_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     image_with_keypoints_gray = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2GRAY)
     status, corners = cv2.findCirclesGrid(image_with_keypoints, (itemspercolumn,itemsperrow), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)
     cv2.imshow("Keypoints", image_with_keypoints)
     cv2.waitKey(0)
     if status == True:
         print('found ',len(keypoints),' items in ',image)
         objpoints.append(objp)  
# Certainly, every loop objp is the same, in 3D.
         refined_corners = cv2.cornerSubPix(image_with_keypoints_gray, corners, (itemsperrow,itemsperrow), (-1,-1), stopping_criteria )
# Refines the corner locations.
         imgpoints.append(refined_corners)
         # Draw and display the corners.
         im_with_keypoints = cv2.drawChessboardCorners(rotated_image, (itemspercolumn,itemsperrow), refined_corners, status)
         cv2.imshow('image', im_with_keypoints)
         cv2.waitKey(0)
         found += 1

cv2.destroyAllWindows()

caliberation_error, camera_matrix, distortion_coff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_model.shape[::-1],None,None)

captured_image = cv2.imread(base_folder + "sample.jpg", cv2.IMREAD_GRAYSCALE)
captured_image  = cv2.resize(captured_image, image_model.shape, interpolation = cv2.INTER_AREA)
cv2.imshow('Original Image', captured_image)


h,  w = captured_image.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,distortion_coff,(w,h),1,(w,h))
undistorted_image =  cv2.undistort(captured_image, camera_matrix, distortion_coff, None, newcameramtx)
cv2.imshow('Undistorted Image', undistorted_image)


x,y,w,h = roi
cropped_image = undistorted_image[y:y+h, x:x+w]
cv2.imshow('cropped Image', cropped_image)

mapx,mapy = cv2.initUndistortRectifyMap(camera_matrix,distortion_coff,None,newcameramtx,(w,h),5)
cropped_image = cv2.remap(captured_image,mapx,mapy,cv2.INTER_LINEAR)
cv2.imshow('cropped/ processed Image', cropped_image) 



