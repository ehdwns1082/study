# https://blog.naver.com/eunseong31/221328072307
# https://blog.naver.com/eunseong31/221328155180

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('landscape_noise.bmp', cv2.IMREAD_GRAYSCALE)
#cv2.imshow('original', img)

# numpy equalization
hist_npEq, bins = np.histogram(img.ravel(), 256, [0, 256])
cdf = hist_npEq.cumsum()
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img_npEq = cdf[img]
#cv2.imshow('Histogram Equalization using numpy', img_npEq)

# opencv equalization
equ = cv2.equalizeHist(img)
#cv2.imshow('equalizer_OpenCV', equ)


# 히스토그램
hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])  # original
hist2 = cv2.calcHist([img_npEq], [0], None, [256], [0, 256])  # numpy equalization
hist3 = cv2.calcHist([equ], [0], None, [256], [0, 256])  # opencv equalization

# matplotlib
plt.subplot(2,3,1), plt.imshow(img, 'gray'),
plt.title('original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2), plt.imshow(img_npEq, 'gray')
plt.title('numpy_eq'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,3), plt.imshow(equ, 'gray')
plt.title('OpenCV_eq'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,4), plt.plot(hist1)
plt.title('original')

plt.subplot(2,3,5), plt.plot(hist2)
plt.title('numpy_eq')

plt.subplot(2,3,6), plt.plot(hist3)
plt.title('OpenCV_eq')

plt.show()
'''
plt.subplot(row, col, idx)
=> 창 하나에 그림이나 그래프 여러개 두고 싶을 떄
    1st : 몇 행 짜리 서브플롯 배치를 만들건지
    2nd : 몇 열 짜리 서브플롯 배치를 만들건지
    3rt : 서브플롯 배치에서 몇 번쨰 인덱스로 배치할 건지
    
plt.xticks([]), plt.yticks([])
=> 그래프나 그림에서 x좌표 y좌표 숫자 없애고 싶을 때
'''

