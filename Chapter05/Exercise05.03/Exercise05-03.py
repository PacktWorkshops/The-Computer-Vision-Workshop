"""
MIT License

Copyright (c) 2020 Packt Workshops

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2
import numpy as np

# Read image
img = cv2.imread("../data/person.jpg")

imgCopy = img.copy()

# Create a mask
mask = np.zeros(img.shape[:2], np.uint8)

# Temporary arrays
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# Select ROI
rect = cv2.selectROI(img)

# Draw rectangle
x,y,w,h = rect

cv2.rectangle(imgCopy, (x, y), (x+w, y+h), (0, 0, 255), 3)

cv2.imwrite("roi.png",imgCopy)

# Perform grabcut
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,9,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
cv2.imshow("Mask",mask*80)
cv2.imshow("Mask2",mask2*255)
cv2.imwrite("mask.png",mask*80)
cv2.imwrite("mask2.png",mask2*255)
cv2.waitKey(0)

img = img*mask2[:,:,np.newaxis]

cv2.imwrite("grabcut-result.png",img)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
