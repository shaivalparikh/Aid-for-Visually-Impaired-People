import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('imgL.jpg',0)  #queryimage # left image
img2 = cv2.imread('imgR.jpg',0) #trainimage # right image
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img1,img2)
plt.imshow(disparity,'gray')
plt.show()
plt.savefig('disparity.jpg')
