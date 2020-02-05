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
inputImagePath = "../data/face.jpg"

haarCascadePath = "../data/haarcascade_frontalface_default.xml"
inputImage = cv2.imread(inputImagePath)

grayInputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

haarCascade = cv2.CascadeClassifier(haarCascadePath)

detectedFaces = haarCascade.detectMultiScale(grayInputImage, 1.2, 1)

for face in detectedFaces:
  print(face)

for face in detectedFaces:

  # Each face is a rectangle representing

  # the bounding box around the detected face

  x, y, w, h = face

  cv2.rectangle(inputImage, (x, y), (x+w, y+h), (0, 0, 255), 3)

cv2.imshow("Faces Detected", inputImage)

cv2.waitKey(0)

cv2.destroyAllWindows()
