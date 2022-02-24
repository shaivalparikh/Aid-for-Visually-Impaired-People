import cv2,time

img = cv2.VideoCapture(0)

check,frame = img.read()

cv2.imshow('frame',frame)

cv2.imwrite('imgR.png',frame)
key = cv2.waitKey(0)

img.release()
cv2.destroyAllWindows()
