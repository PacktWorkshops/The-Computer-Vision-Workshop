import cv2 

image = cv2.imread('phrase_handwritten.png') 
imagecopy= image.copy()
cv2.imshow( 'Original image' , image )
cv2.waitKey(0) 
cv2.destroyAllWindows()

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 

ret,binary_im = cv2.threshold(gray_image,0,255,cv2.THRESH_OTSU) 
cv2.imshow( 'binary image' , binary_im )
cv2.waitKey(0) 
cv2.destroyAllWindows()

_,contours_list,_ = cv2.findContours(binary_im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
for cnt in contours_list:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 255),2) 
cv2.imshow( 'Contours marked on RGB image' , image )
cv2.waitKey(0) 
cv2.destroyAllWindows()

ref_gray = cv2.imread('typed_B.png', cv2.IMREAD_GRAYSCALE) 
ret, ref_binary = cv2.threshold(ref_gray,0,255,cv2.THRESH_OTSU) 
cv2.imshow( 'Reference image' , ref_binary )
cv2.waitKey(0) 
cv2.destroyAllWindows()

_,ref_contour_list,_ = cv2.findContours(ref_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
if len(ref_contour_list)==1:
    ref_contour= ref_contour_list[0]
else:
    import sys
    print('Reference image contains more than 1 contour. Please check!')
    sys.exit()
    
ctr= 0
dist_list= []
for cnt in contours_list:
    retval	=	cv2.matchShapes(cnt, ref_contour,cv2.CONTOURS_MATCH_I1,0)
    dist_list.append(retval)
    ctr= ctr+1

min_dist= min(dist_list)  
#print('The minimum distance of the reference contour with a contour in the main image is ' + str(min_dist))
ind_min_dist= dist_list.index(min_dist)

#with_contours = cv2.drawContours(image,contours_list, ind_min_dist, (0,255,255),3) 
#cv2.imshow( 'contours marked on RGB image' , with_contours )
#cv2.waitKey(0) 
#cv2.destroyAllWindows()

x,y,w,h = cv2.boundingRect(contours_list[ind_min_dist])
cv2.rectangle(imagecopy,(x,y),(x+w,y+h),(255, 0, 0),2) 
cv2.imshow( 'Detected B' , imagecopy )
cv2.waitKey(0) 
cv2.destroyAllWindows()