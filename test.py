import numpy as np
import cv2 as cv
import os 

filename = 'chessboard.png'
img_name_list = [i for i in os.listdir('./img')]
img_list = [cv.imread(os.path.join('./img',i)) for i in img_name_list]
img = img_list[0]
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()