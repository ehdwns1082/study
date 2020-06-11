# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220544731502&parentCategoryNo=&categoryNo=66&viewDate=&isShowPopularPosts=false&from=postList

# BGR 은 3채널 컬러이므로 히스토그램을 그리려면 3차원 좌표가 필요하다.
# HSV 는 Hue(색상) 와 Saturation(채도) 만 이용해서 컬러이미지의 특징적인 히스토그램을 그릴 수 있다.
# Hue 는 0~179, Saturation 은 0~255 의 값법위를 가진다.

import numpy as np
import cv2
import matplotlib.pyplot as plt

# hist2D
img = cv2.imread('image_landscape.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])
# [0,1]:채널값, [180,256]:BIN 개수, [0,180,0,256]:범위
cv2.imshow('hist2D', hist)

# matplotlib.pyplot.imshow()
plt.imshow(hist, interpolation='nearest')
plt.show()

hscale = 10

def onChange(x):
    global hscale
    hscale = x

def HSVmap():
    hsvmap = np.zeros((180,256,3), np.uint8)
    h, s = np.indices(hsvmap.shape[:2])
    hsvmap[:,:,0] = h
    hsvmap[:,:,1] = s
    hsvmap[:,:,2] = 255
    hsvmap = cv2.cvtColor(hsvmap, cv2.COLOR_HSV2BGR)

    return hsvmap

def hist2D():
    img = cv2.imread('image_landscape.jpg')
    hsvmap = HSVmap()
    cv2.namedWindow('hist2D', 0)
    cv2.createTrackbar('scale', 'hist2D', hscale, 32, onChange)

    while True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])

        hist = np.clip(hist*0.005*hscale,0,1)
        '''
        numpy.clip(a, min, max) 
        => 인자로 입력된 numpy 배열 a의 멤버값이 min 보다 작으면 min, max 보다 크면 max 로 하는 새로운 numpy 배열을 리턴한다.
            >>> a = [-2,-3,1,3,4,5,0,1]
            >>> np.clip(a,0,2)
            [0,0,1,2,2,2,0,1]
                
        '''
        hist = hsvmap*hist[:,:,np.newaxis]/255.0
        '''
        hsvmap(180,256,3) 과 hist(180,256) 의 차원이 다르므로, 차원이 작은 배열을 큰 차원의 배열로 변경해서 곱해야 한다. 
        hist[:,:,np.newaxis] 는 (180,256) 인 배열을 (180,256,1) 로 변경하여 곱셈 연산이 가능케 한다. 
        그 후 255.0 으로 나누어 픽셀 값 대부분을 0~1 사이의 값이 되도록 만든다.
        '''
        cv2.imshow('hist2D', hist)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

hist2D()


cv2.waitKey(0)
cv2.destroyAllWindows()















