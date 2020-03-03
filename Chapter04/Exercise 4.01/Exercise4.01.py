# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:21:18 2020

@author: sajid
"""

#importing the required libraries 
import cv2 

image = cv2.imread('tennis court.jpg') 
imageCopy= image.copy()
cv2.imshow( 'BGR image' , image )
cv2.waitKey(0) 
cv2.destroyAllWindows()

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
cv2.imshow( 'gray' , gray_image )
cv2.waitKey(0) 
cv2.destroyAllWindows()

ret,binary_im = cv2.threshold(gray_image,160, 255, cv2.THRESH_BINARY)
cv2.imshow( 'binary' , binary_im )
cv2.waitKey(0) 
cv2.destroyAllWindows()

#calculate the contours from binary image
im,contours,hierarchy = cv2.findContours(binary_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 

contours_to_plot= -1
plotting_color= (0,255,0)
thickness= 3
with_contours = cv2.drawContours(image,contours,contours_to_plot, plotting_color,thickness)  #To draw all contours, pass -1) and remaining arguments are color, thickness etc.
cv2.imshow( 'contours' , with_contours )
cv2.waitKey(0) 
cv2.destroyAllWindows()

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
    
cv2.imshow( 'contours' , image )
cv2.waitKey(0) 
cv2.destroyAllWindows()

#cv2.imshow( 'orig' , imageCopy)
#cv2.waitKey(0) 
#cv2.destroyAllWindows()

required_contour = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(required_contour)
img_copy2 = cv2.rectangle(imageCopy,(x,y),(x+w,y+h),(0,255,255),2)

cv2.imshow( 'largest contour' , img_copy2)
cv2.waitKey(0) 
cv2.destroyAllWindows()
