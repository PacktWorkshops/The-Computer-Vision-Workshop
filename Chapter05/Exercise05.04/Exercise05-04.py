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

# OpenCV Utility Class for Mouse Handling
class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.imshow(self.windowname, self.dests[0])
        cv2.imshow(self.windowname + ": mask", self.dests[1])

    # onMouse function for Mouse Handling
    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None
        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()

# Read image
img = cv2.imread("../data/grabcut.jpg")

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
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
cv2.imshow("Mask",mask*80)
cv2.imshow("Mask2",mask2*255)
cv2.imwrite("mask.png",mask*80)
cv2.imwrite("mask2.png",mask2*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = img*mask2[:,:,np.newaxis]

img_mask = img.copy()
mask2 = mask2*255
mask_copy = mask2.copy()

# Create sketch using OpenCV Utility Class: Sketcher
sketch = Sketcher('image', [img_mask, mask2], lambda : ((255,0,0), 255))

while True:
    ch = cv2.waitKey()
    # Quit
    if ch == 27:
        print("exiting...")
        cv2.imwrite("img_mask_grabcut.png",img_mask)
        cv2.imwrite("mask_grabcut.png",mask2)
        break
    # Reset
    elif ch == ord('r'):
        print("resetting...")
        img_mask = img.copy()
        mask2 = mask_copy.copy()
        sketch = Sketcher('image', [img_mask, mask2], lambda : ((255,0,0), 255))
        sketch.show()
    # Change to background
    elif ch == ord('b'):
        print("drawing background...")
        sketch = Sketcher('image', [img_mask, mask2], lambda : ((0,0,255), 0))
        sketch.show()
    # Change to foreground
    elif ch == ord('f'):
        print("drawing foreground...")
        sketch = Sketcher('image', [img_mask, mask2], lambda : ((255,0,0), 255))
        sketch.show()
    else:
        print("performing grabcut...")
        mask2 = mask2//255
        cv2.grabCut(img,mask2,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask2==2)|(mask2==0),0,1).astype('uint8')
        img_mask = img*mask2[:,:,np.newaxis]
        mask2 = mask2*255
        print("switching bank to foreground...")
        sketch = Sketcher('image', [img_mask, mask2], lambda : ((255,0,0), 255))
        sketch.show()

cv2.destroyAllWindows()
