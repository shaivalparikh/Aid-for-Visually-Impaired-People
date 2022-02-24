import cv2, time

video0 = cv2.VideoCapture(0)
#video1 = cv2.VideoCapture(1)
#video2 = cv2.VideoCapture(2)

while True:
    check0,frame0 = video0.read()
 #   check1,frame1 = video1.read()
  #  check2,frame2 = video2.read()

    cv2.imshow('frame0',frame0)
   # cv2.imshow('frame1',frame1)
    #cv2.imshow('frame2',frame2)

    key= cv2.waitKey(1)
    if key == ord('q'):
        break

video0.release()
#video1.release()
#video2.release()
cv2.destroyAllWindows()
