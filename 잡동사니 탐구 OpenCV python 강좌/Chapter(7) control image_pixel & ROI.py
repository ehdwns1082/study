# https://m.blog.naver.com/samsjang/220502203203

import cv2
import numpy as np

# 이미지 픽셀 값 얻고 수정하기 => 어떤 이미지의 픽셀값을 얻으면 (B,G,R) 의 형태로 나타난다.

print('이미지 픽셀 값 얻고 수정하기')
img = cv2.imread('image_landscape.jpg') # 이미지를 읽을 때 Numpy array 로 받는다.
px = img[20,154] # img 의 (20,154) 픽셀의 BGR 값
print(px)
img[20,154] = [0,0,0] # (20,154) 픽셀을 검정색으로 수정함
print(px)
img[20,154] = [255,0,0] # (20,154) 픽셀을 빨간색으로 수정함
print(px)
'''
BGR 값 개별적으로 수정하기
    
    img[20,154,0]  => B 값
    img[20,154,1]  => G 값
    img[20,154,2]  => R 값
    * 위 코드는 개개의 픽셀 작업을 수행하는데 있어 성능상 문제가 생길 수 있다.
'''
B = img.item(20, 154, 0)
G = img.item(20, 154, 1)
R = img.item(20, 154, 2)
# Numpy array의 item() 함수는 개별적인 픽셀에 접근할 수 있지만 B,G,R 개별적으로 접근해야 한다.
BGR = [B, G, R]
print(BGR)
img.itemset((20, 154, 0), 100)
# (20,154) 위치의 픽셀의 B 값을 100으로 변경, itemset() 함수 역시 B,G,R 개별적인 값을 변경해야함
print(px)

print(' ')

# 이미지 속성 얻기
print('이미지 속성 얻기')
img = cv2.imread('image_landscape.jpg')
print(img.shape) # (height, width, 컬러 채널 수) 반환
print(img.size) # 이미지 사이즈(byte) 반환
print(img.dtype) # 이미지 데이터 타입 반환

print(' ')

# ROI 설정 => Region Of Image(이미지 영역)
    # ex) 이미지에서 눈을 찾는다고 할 때, 이미지 전체에서 얼굴을 먼저 찾을 후, 이 얼굴 영역에서 눈을 찾는 것이 성능적에서 효율적.
    # ROI 는 Numpy 인덱싱을 통해 얻을 수 있다.
print('ROI 설정')
img = cv2.imread('image_landscape.jpg')
cv2.imshow('original', img)
subimg = img[90:140, 20:190]
# 원본 이미지의 90~140 row(세로) 와 20~190 cols(가로) 를 ROI로 잡음
cv2.imshow('cutting', subimg)
img[0:50, 0:170] = subimg
# 원본 이미지상의 좌표 (0,50)~(0,170) 영역에 ROI로 잡은 부분 삽입
print(img.shape)
print(subimg.shape)
cv2.imshow('modified', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(' ')

# 이미지 채널 분할 및 합치기
print('이미지 채널 분할 및 합치기')
# 필요시 컬러이미지의 B,G,R 채널별로 픽셀값들을 따로 분리할 수 있다. 또 분리해 놓은 B,G,R 채널을 합쳐서 컬러 이미지로 변환할 수 도 있다.
img = cv2.imread('image_landscape.jpg')
b,g,r = cv2.split(img)
print(img[100,100])
print(b[100,100], g[100,100], r[100,100])

cv2.imshow('b_cv2.split()', b)
cv2.imshow('g_cv2.split()', g)
cv2.imshow('r_cv2.split()', r)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.split() 함수와는 반대로 cv2.merge()함수를 이용하면 B,G,R로 분리된 채널을 합쳐서 컬러 이미지로 만들 수 있다.
merged_img = cv2.merge((b,g,r))
cv2.imshow('cv2.split()_merged', merged_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.split() 함수는 성능면에서 효율적인 함수가 아니기 때문에 꼭 필요한 경우가 아니면 Numpy 인덱싱을 활용하는 것이 좋다.
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
cv2.imshow('b_Numpy indexing',b)
cv2.imshow('g_Numpy indexing',g)
cv2.imshow('r_Numpy indexing',r)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 만약 어떤 이미지의 모든 픽셀의 RED 값을 0으로 하고 싶으면 >>> img[:,:,2]=0 하면 된다
img[:,:,2] = 0
cv2.imshow('red = 0',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

