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

inputImagePath = "../data/eyes.jpeg"

haarCascadePath = "../data/haarcascade_eye.xml"

def detectionUsingCascades(imageFile, cascadeFile):

    """This is a custom function which is responsible

    for carrying out object detection using cascade model.

    The function takes the cascade filename and the image

    filename as the input and returns the list of

    bounding boxes around the detected object instances."""

    # Step 1 – Load the image
    image = cv2.imread(imageFile)

    # Step 2 – Convert the image from BGR to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3 – Load the cascade
    haarCascade = cv2.CascadeClassifier(cascadeFile)

    # Step 4 – Perform multi-scale detection
    detectedObjects = haarCascade.detectMultiScale(gray, 1.2, 2)

    # Step 5 – Draw bounding boxes
    for bbox in detectedObjects:
        # Each bbox is a rectangle representing
        # the bounding box around the detected object
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)

    # Step 6 – Display the output

    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.imwrite("eyes-result.png",image)
    cv2.destroyAllWindows()

    # Step 7 – Return the bounding boxes
    return detectedObjects

eyeDetection = detectionUsingCascades("../data/eyes.jpeg",
                                      "../data/haarcascade_eye.xml")
