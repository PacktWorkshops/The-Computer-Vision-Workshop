



import cv2
import numpy as np
from matplotlib import pyplot as plt

minHessian = 400
sift_detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)

basefolder = r'C:\Users\opencv_book\stereo\\'

imgL = cv2.imread(basefolder  + 'IMG20200421082417.jpg')
imgR = cv2.imread(basefolder  + 'IMG20200421082426.jpg')

imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


keypoints1, descriptors1 = sift_detector.detectAndCompute(imgL, None)
keypoints2, descriptors2 = sift_detector.detectAndCompute(imgR, None)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

ratio_thresh = 0.7
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

img_matches = np.empty((max(imgL.shape[0], imgR.shape[0]), imgL.shape[1]+imgR.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(imgL, keypoints1, imgR, keypoints2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

