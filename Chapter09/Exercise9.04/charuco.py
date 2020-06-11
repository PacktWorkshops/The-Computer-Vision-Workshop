
import numpy as np
import cv2, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

arucoParams = aruco.DetectorParameters_create()
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
markerLength = 1
markerSeparation = 0.5
targetwidth = 4
targetheight = 4

board = aruco.CharucoBoard_create(targetwidth, targetheight, markerLength, markerSeparation, aruco_dict)
imboard = board.draw((1000, 1000))
handle = plt.imshow(imboard,cmap='gray')
plt.axis('off')
plt.savefig(basefolder + "/charuco_markers.png")

camera = cv2.VideoCapture(0)

while True:
    status, img_charuco = camera.read()
    im_gray = cv2.cvtColor(img_charuco ,cv2.COLOR_RGB2GRAY)
    h,  w = im_gray.shape[:2]
    dst = cv2.undistort(im_gray, camera_matrix, distortion_coff, None, newcameramtx)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
    cv2.imshow("original", im_gray)
    
    if not corners :
        print ("pass")
    else:
        aruco.refineDetectedMarkers(im_gray, board, corners, ids, rejectedImgPoints)
        charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, im_gray, board)
        image_with_charuco_board = aruco.drawDetectedCornersCharuco(img_charuco, charucoCorners, charucoIds, (0,255,0))
        status , rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, distortion_coff)         
        if status != 0:
            img_aruco = aruco.drawAxis(image_with_charuco_board, newcameramtx, distortion_coff, rvec, tvec,20)
        else:
            print('no markers detected')
            
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.imwrite(base_folder + 'detected charuco pattern.jpg', img_charuco)
        break;
        
    cv2.imshow("World co-ordinate frame axes", img_charuco)


camera.release()
cv2.destroyAllWindows()
