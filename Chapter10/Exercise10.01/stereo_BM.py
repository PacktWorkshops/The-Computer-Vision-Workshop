

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

stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setBlockSize(9)
stereoMatcher.setSpeckleRange(32)
stereoMatcher.setSpeckleWindowSize(80)

depth = stereoMatcher.compute(imgL, imgR)
depth   =  cv2.resize(depth   , (1000,1000), interpolation = cv2.INTER_AREA)
scale = 255
cv2.imshow('depth Map', depth / scale )

