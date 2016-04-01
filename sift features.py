import cv2
import numpy as np
import cPickle

img = cv2.imread('car.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(img, keypoints = kp)

index = []

for point in kp:
    temp = (point.pt, point.size, point.angle, point.response, point.octave, 
            point.class_id) 
    index.append(temp)

## Put the keypoints into a File

f = open("keypoints.txt", "w")
f.write(cPickle.dumps(index))
f.close()
cv2.imwrite('sift_keypoints.jpg', img)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
