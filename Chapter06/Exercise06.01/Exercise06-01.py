# Import modules
import cv2
import numpy as np

# Read video
cap = cv2.VideoCapture("../data/lemon.mp4")

if cap.isOpened() == False:
    print("Error opening video")

cv2.namedWindow("Input Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Hue", cv2.WINDOW_NORMAL)

while True:
    # capture frame
    ret, frame = cap.read()

    if ret == False:
        break
    
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Obtain Hue channel
    h = hsv[:,:,0]

    # Apply thresholding
    hCopy = h.copy()
    h[hCopy>40] = 0
    h[hCopy<=40] = 1

    # Display frame
    cv2.imshow("Input Video", frame)
    cv2.imshow("Output Video", frame*h[:,:,np.newaxis])
    cv2.imshow("Hue", h*255)

    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
