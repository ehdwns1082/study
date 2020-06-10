# 아스키 코드를 이용한 암호화 및 복호화 : https://it1234.tistory.com/54
# https://stackoverflow.com/questions/51218937/saving-a-3d-numpy-array-to-txt-file
# array to list
    #http://egloos.zum.com/qordidtn02/v/6281203

import cv2
import numpy as np
img = cv2.imread('bug.jpg')
y,x,k = img.shape
reshape = img.reshape(1,y*x*k)
reshape1 = reshape[0].tolist()
reshape1.reverse()
new = []
for i in range(y*x*k):
    new.append(chr(reshape1.pop()))
with open('C:/Users/ehdwn/pixel2askii.txt', 'w', -1, 'utf-8') as file:
    for i in range(y*x*k):
        file.write(new[i])

np.savetxt('C:/Users/ehdwn/pixel.txt', reshape, fmt = '%2d', delimiter=' ')
img2 = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA )
cv2.imshow('img_small', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


