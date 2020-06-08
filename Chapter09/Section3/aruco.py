




aruco_dict = aruco.Dictionary_get( aruco.DICT_6X6_50 )


import matplotlib.pyplot as plt
import matplotlib as mpl

markerLength = 1
markerSeparation = 0.5   
targetwidth = 3
targetheight = 3


board = aruco.GridBoard_create(targetwidth, targetheight, markerLength, markerSeparation, aruco_dict)

imboard = board.draw((1000, 1000))
handle = plt.imshow(imboard,cmap='gray')
plt.axis('off')

plt.savefig(base_folder + "/aruco_markers.png")
arucoParams = aruco.DetectorParameters_create()

camera = cv2.VideoCapture(0)

while True:
    status, detected_image = camera.read()
    img_aruco = detected_image 
    im_gray = cv2.cvtColor(detected_image ,cv2.COLOR_RGB2GRAY)
    h,  w = im_gray.shape[:2]
    dst = cv2.undistort(im_gray, camera_matrix, distortion_coff, None, newcameramtx)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
    cv2.imshow("original", im_gray)
    if corners == None:
        print ("pass")
    else:
        status, Rotation, Translation = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, distortion_coff) # For a board
        if status != 0:
            img_aruco = aruco.drawDetectedMarkers(detected_image, corners, ids, (0,255,0))
            img_aruco = aruco.drawAxis(img_aruco, newcameramtx, distortion_coff, Rotation, Translation, 10)    # axis length 100 can be changed according to your requirement

        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.imwrite('detected aruco pattern.jpg', img_aruco)
            break;
    cv2.imshow("World co-ordinate frame axes", img_aruco)


status, detected_image = camera.read()
im_gray = cv2.cvtColor(detected_image ,cv2.COLOR_RGB2GRAY)
h,  w = im_gray.shape[:2]
dst = cv2.undistort(im_gray, camera_matrix, distortion_coff, None, newcameramtx)
ifmarkers, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
status, Rotation, Translation = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, distortion_coff) 

img_aruco = aruco.drawDetectedMarkers(detected_image, ifmarkers, ids, (0,255,0))
img_aruco = aruco.drawAxis(img_aruco, newcameramtx, distortion_coff, Rotation, Translation, 10)    

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.imwrite(base_folder + 'detected aruco pattern.jpg', img_aruco)
    break;

camera.release()
cv2.destroyAllWindows()



