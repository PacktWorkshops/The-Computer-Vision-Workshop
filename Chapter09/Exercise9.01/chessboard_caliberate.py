

import numpy as np
import cv2
import glob
import os 
# termination criteria
stopping_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 45, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
ObjectPoints = [] # 3d point in real world space
ImagePoints = [] # 2d points in image plane.

basefolder= r'C:\Users\mbhattac\OneDrive - HERE Global B.V-\coding\opencv_book\Chapter09\Section1\images\\'

os.chdir(basefolder)

images = glob.glob(basefolder + "*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    status, inner_corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if status == True:
        ObjectPoints.append(objp)

        refined_corners = cv2.cornerSubPix(gray,inner_corners ,(13,13),(-1,-1),stopping_criteria )
        ImagePoints.append(refined_corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), refined_corners,status)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)
        cv2.imwrite('Image wth chess board pattern.jpg',img )


cv2.destroyAllWindows()


caliberation_error, camera_matrix, distortion_coff, rvecs, tvecs = cv2.calibrateCamera(ObjectPoints, ImagePoints, gray.shape[::-1],None,None)

# select a sample image & display it 
sample_image = cv2.imread(basefolder+ "left12.jpg")
cv2.imshow('Sample Image',sample_image )
cv2.imwrite('Sample Image.jpg',sample_image )
h,  w = sample_image.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,distortion_coff,(w,h),0.8,(w,h))

# undistort
undistorted_sample_image= cv2.undistort(sample_image, camera_matrix, distortion_coff, None, newcameramtx)
cv2.imshow('Undistorted Image',undistorted_sample_image )
cv2.imwrite('Undistorted Image.jpg',undistorted_sample_image )

# crop the image
x,y,w,h = roi
clipped_sample_image = undistorted_sample_image[y:y+h, x:x+w]
cv2.imshow('Clipped Image',clipped_sample_image )
cv2.imwrite('Clipped Image.jpg',clipped_sample_image )


# undistort it using another method 
mapx,mapy = cv2.initUndistortRectifyMap(camera_matrix,distortion_coff,None,newcameramtx,(w,h),5)
undistorted_new = cv2.remap(clipped_sample_image,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
undistorted_new = undistorted_new[y:y+h, x:x+w]
cv2.imshow('Second undistorted Image',undistorted_new  )
cv2.imwrite('Second undistorted Image.jpg',undistorted_new  )


