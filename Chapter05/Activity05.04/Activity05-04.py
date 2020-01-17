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

def emojiFilter(image, cascadeFile, emojiFile):

    """This is a custom function which is responsible

    for carrying out object detection using cascade model.

    The function takes the filenames of both cascades and the image

    filename as the input and returns the list of

    bounding boxes around the detected object instances."""

    # Step 1 - Read the emoji
    emoji = cv2.imread(emojiFile,-1)
    #cv2.imwrite("emoji_alpha.jpg",emoji[:,:,3])

    # Step 2 – Convert the image from BGR to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3 – Load the cascade
    haarCascade = cv2.CascadeClassifier(cascadeFile)

    # Step 4 – Perform multi-scale detection
    detectedObjects = haarCascade.detectMultiScale(gray, 1.2, 3)

    # If no objects detected, return None
    if len(detectedObjects) == 0:
        return None

    # Step 5 – Draw bounding boxes
    for bbox in detectedObjects:

        # Each bbox is a rectangle representing
        # the bounding box around the detected object
        x, y, w, h = bbox

        # Resize emoji to match size of face
        emoji_resized = cv2.resize(emoji, (w,h), interpolation = cv2.INTER_AREA)
        (image[y:y+h, x:x+w])[np.where(emoji_resized[:,:,3]!=0)] = (emoji_resized[:,:,:3])[np.where(emoji_resized[:,:,3]!=0)]

    # Step 6 – Return the revised image
    return image



# Start webcam
cap = cv2.VideoCapture(0)#"video.mp4")

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening webcam")


while True:
    # Capture frame
    ret, frame = cap.read()
    #cv2.imshow("Image", frame)
    #cv2.waitKey(25)
    emojiFilterResult = None

    # Check if frame read successfully
    if ret == True:
        #frame = cv2.resize(frame, (int(0.5*frame.shape[1]),int(0.5*frame.shape[0])), interpolation = cv2.INTER_AREA)
        # Emoji filter
        emojiFilterResult = emojiFilter(frame,
                                        "../data/haarcascade_frontalface_default.xml",
                                        "../data/emoji.png")
    else:
        break

    if emojiFilterResult is None:
        continue
    else:
        cv2.imshow("Emoji Filter", emojiFilterResult)
        k = cv2.waitKey(25)
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
