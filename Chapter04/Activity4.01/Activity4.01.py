import cv2 

## reading refernece image now
ref_image = cv2.imread('bananaref.jpeg') 
cv2.imshow( 'binary image' , ref_image )
cv2.waitKey(0) 
cv2.destroyAllWindows()

gray_image = cv2.cvtColor(ref_image,cv2.COLOR_BGR2GRAY) 
cv2.imshow( 'binary image' , gray_image )
cv2.waitKey(0) 
cv2.destroyAllWindows()

ret,binary_im = cv2.threshold(gray_image,245,255,cv2.THRESH_BINARY) 
binary_im= ~binary_im
cv2.imshow( 'binary image' , binary_im )
cv2.waitKey(0) 
cv2.destroyAllWindows()

im,ref_contour_list,hierarchy = cv2.findContours(binary_im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
with_contours = cv2.drawContours(ref_image,ref_contour_list,-1,(0,0,255),3) 
cv2.imshow( 'contours marked on RGB image' , with_contours )
cv2.waitKey(0) 
cv2.destroyAllWindows()

reference_contour = max(ref_contour_list, key = cv2.contourArea)
with_contours = cv2.drawContours(ref_image,reference_contour,-1,(255,0,0),3) 
cv2.imshow( 'largest contour marked on RGB image' , with_contours )
cv2.waitKey(0) 
cv2.destroyAllWindows()


image = cv2.imread('many fruits.png') 
imagecopy= image.copy()
cv2.imshow( 'Original image' , image )
cv2.waitKey(0) 
cv2.destroyAllWindows()

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
cv2.imshow( 'gray' , gray_image )
cv2.waitKey(0) 
cv2.destroyAllWindows()

ret,binary_im = cv2.threshold(gray_image,245,255,cv2.THRESH_BINARY) 
cv2.imshow( 'binary' , binary_im )
cv2.waitKey(0) 
cv2.destroyAllWindows()

binary_im= ~binary_im
cv2.imshow( 'binary' , binary_im )
cv2.waitKey(0) 
cv2.destroyAllWindows()
#im,allcontours,hierarchy = cv2.findContours(binary_im,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
#with_contours = cv2.drawContours(image,allcontours,-1,(0,255,0),3) 
#cv2.imshow( 'contours marked on RGB image' , with_contours )
#cv2.waitKey(0) 
#cv2.destroyAllWindows()


import numpy as np
kernel = np.ones((5, 5),np.uint8)

binary_im = cv2.morphologyEx(binary_im, cv2.MORPH_OPEN, kernel)
cv2.imshow( 'binary image after noise removal' , binary_im )
cv2.waitKey(0) 
cv2.destroyAllWindows()

#calculate the contours from binary image
im,contours,hierarchy = cv2.findContours(binary_im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

with_contours = cv2.drawContours(image,contours,-1,(0,0,255),3) 
cv2.imshow( 'contours marked on RGB image' , with_contours )
cv2.waitKey(0) 
cv2.destroyAllWindows()

dist_list= []
for cnt in contours:
    retval	=	cv2.matchShapes(cnt, reference_contour,cv2.CONTOURS_MATCH_I1,0)
    dist_list.append(retval)

sorted_list= dist_list.copy()
sorted_list.sort() # sorts the list from smallest to largest 
ind1_dist= dist_list.index(sorted_list[0])
ind2_dist= dist_list.index(sorted_list[1])

pineapple_cnts= []
pineapple_cnts.append(contours[ind1_dist])
pineapple_cnts.append(contours[ind2_dist])
    
with_contours = cv2.drawContours(image,pineapple_cnts,-1,(255,0,0),3) 
cv2.imshow( 'contours marked on RGB image' , with_contours )
cv2.waitKey(0) 
cv2.destroyAllWindows()

for cnt in pineapple_cnts:
    x,y,w,h = cv2.boundingRect(cnt)
    if h>w:
        cv2.rectangle(imagecopy,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow( 'Upright banana marked on RGB image' , imagecopy )
cv2.waitKey(0) 
cv2.destroyAllWindows()

