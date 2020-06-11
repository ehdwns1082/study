# https://blog.naver.com/samsjang/220503082434

import cv2
import numpy as np

# 이미지 더하기
img1 = cv2.imread('image_landscape.jpg')
img2 = cv2.imread('puppy.jpg.')
print(img1.shape)
print(img2.shape)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
add_img1 = img1 + img2
# img1 과 img2 는 동일한 크기의 동일한 데이터 타입이어야 함.
# 각 픽셀들을 더한 값이 255보다 크면 그 값을 256으로 나눈 나머지가 픽셀값이 됨.
# ex) 257일경우 256으로 나눈 나머지가 1이므로 픽셀값이 1이 된다.
add_img2 = cv2.add(img1, img2)
# 위의 Numpy array 연산과 다르게 더한값이 255보다 크면 255로 값이 정해진다.
cv2.imshow('img1+1mg2',add_img1)
cv2.imshow('add(img1,img2)', add_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 이미지 블렌딩
def onMouse(x):
    pass

def imgBlending(imgfile1, imgfile2):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)
    cv2.namedWindow('imgPane') # Pane : 유리창
    cv2.createTrackbar('Mixing', 'imgPane', 0, 100, onMouse)
    while True:
        mix = cv2.getTrackbarPos('Mixing', 'imgPane')  # 초기값 0
        img = cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100, 0)
        # g(x) = (1-a)f0(x) + af1(x) : a값이 0에서 1로 변해가면 f0 의 효과는 감소하고 f1의 효과는 커지게 된다.
        cv2.imshow('imgPane', img)
        print(mix)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

imgBlending('image_landscape.jpg', 'puppy.jpg')


# 이미지 비트연산
def bitOperation(hpos, vpos): # 로고가 놓일 위치 ; horizontal_position, vertical_position
    img1 = cv2.imread('racoon.png')
    img2 = cv2.imread('logo.jpg')
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    print(img1.shape)
    print(img2.shape)
    # 로고를 배치할 해당영역 지정하기
    rows, cols, channels = img2.shape
    roi = img1[vpos:vpos + rows, hpos:hpos + cols]
    # 로고 배경제거를 위한 마스크와 역마스크 생성하기
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    # 1st = 이진화 할 대상 이미지(gray scale 이어야 함)
    # 2nd = threshold
    # 3rd = threshold 이상값을 이 값으로 바꿈
    # 4th = threshold 기준으로 픽셀값을 binary하게 나눔(크면 3rd값으로, 작으면 0으로)
    # 로고 배경색이 black(gray scale 0)이므로 threshold 10 으로 잡고 나눠준다.
    mask_inv = cv2.bitwise_not(mask)
    # ROI 에서 로고에 해당하는 부분만 검정색으로 만들기
    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    # 로고 이미지에서 로고 부분만 추출하기
    img2_fg = cv2.bitwise_and(img2, img2, mask = mask)
    # 로고 이미지 배경을 cv2.add로 투명으로 만들고 ROI 에 로고 이미지 넣기
    dst = cv2.add(img1_bg, img2_fg)
    img1[vpos:vpos + rows, hpos:hpos + cols] = dst
    cv2.imshow('result', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

bitOperation(400,120)