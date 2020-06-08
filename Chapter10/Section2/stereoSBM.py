import numpy as np
import cv2
from matplotlib import pyplot as plt


basefolder = r'C:\Users\opencv_book\stereo\\'
imgL = cv2.imread(basefolder  + 'IMG20200421082417.jpg')
imgR = cv2.imread(basefolder  + 'IMG20200421082426.jpg')


imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


blockSize = 40


stereo = cv2.StereoSGBM_create(minDisparity=1,
    numDisparities=16,
    blockSize=15,
    speckleWindowSize = 10,
    speckleRange = 32,
    disp12MaxDiff = 1,
    P1 = 8*3*blockSize**2,
    P2 = 32*3*blockSize**2)

depth = stereo.compute(imgL, imgR)


depth= cv2.normalize(depth, depth, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
depth=  cv2.resize(depth   , (1000,1000), interpolation = cv2.INTER_AREA)
cv2.imshow('disparity', depth)



 
