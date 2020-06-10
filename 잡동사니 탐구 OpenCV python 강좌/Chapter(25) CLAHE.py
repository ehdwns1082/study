# https://blog.naver.com/PostList.nhn?blogId=samsjang&categoryNo=66&from=postList
'''
Contrast Limited Adaptive Histogram Equalization
=> Adaptive Histogram Equalization
    CLAHE 는 이미지를 일정한 크기를 가진 작은 블록(타일)으로 구분하고(OpenCV 에서는 디폴트가 8x8 타일),
    '블록별로 히스토그램 균일화를 시행'하여 이미지 전체에 대한 equalization 을 달성하는 메커니즘을 가지고 있다.
    이미지에 노이즈가 있는경우, 타일 단위의 히스토그램 균일화를 적용하면 노이즈가 커질 수도 있는데,
    CLAHE 알고리즘은 이러한 노이즈를 감쇠시키는 Contrast Limiting 이라는 기법을 이용한다.
    각 타일별로 히스토그램 균일화가 모두 마무리 되면, 타일간 경계부분은 bilinear interpolation 을 적용해 매끈하게 만든다.

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('plaster_cast.bmp', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)

# original img
hist = cv2.calcHist([img], [0], None, [256], [0,256])
plt.plot(hist)
plt.title('hist')
plt.show()

# normal equalization
equ = cv2.equalizeHist(img)
cv2.imshow('equ', equ)
hist_equ = cv2.calcHist([equ], [0], None, [256], [0,256])
plt.plot(hist_equ)
plt.title('hist_equ')
plt.show()

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img)
res = np.hstack((img, img_clahe))
cv2.imshow('clahe', res)


cv2.waitKey(0)
cv2.destroyAllWindows()


















