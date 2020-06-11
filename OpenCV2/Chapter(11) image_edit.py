# https://blog.naver.com/samsjang/220504966397

import cv2
import numpy as np

# 이미지 리사이징
def resize():
    img = cv2.imread('musician.png')

    img2 = cv2.resize(img, None, fx=0.5, fy=1, interpolation=cv2.INTER_AREA)
    img3 = cv2.resize(img, None, fx=1, fy=0.5, interpolation=cv2.INTER_AREA)
    img4 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    '''
    cv2.resize(img, None, fx, fy, interpolation)
    1st : 리사이징을 할 이미지 원본
    2nd : dsize 를 나타내는 튜플 값으로(가로방향 픽셀수, 세로방향 픽셀수) 로 나타낸다.
    3rd : fx, fy 각각 가로방향, 세로방향으로 몇 배 할건지. 0.5로 지정하면 원래 크기의 0.5로 리사이징 함.
    4th : 리사이징을 수행할 떄 적용할 interpolation 방법(보간법)
        INTER_NEAREST : nearest-neighbor interpolation
        INTER_LINEAR : bilinear interpolation (디폴트 값)
        INTER_AREA : 픽셀 영역 관계를 이용한 resampling 방법으로 이미지 축소에 있어 선호되는 방법
                     이미지를 확대하는 경우에는 INTER_NEAREST 와 비슷한 효과를 보임.
        INTER_CUBIC : 4x4 픽셀에 적용되는 bicubic interpolation
        INTER_LANCZOS4 : 8x8 픽셀에 적용되는 Lanczos interploation   
    * interpolation 을 할 때 리사이징에 따라 다른 방법을 적용하는 것이 좋다.
      일반적으로 이미지를 축소하는 경우에는 cv2.INTER_AREA 를 사용하고 
      이미지를 확대하는 경우에는 cv2.INTER_CUBIC + cv2.INTER_LINEAR 를 사용한다.
    '''
    cv2.imshow('original', img)
    cv2.imshow('fx=0.5', img2)
    cv2.imshow('fy=0.5', img3)
    cv2.imshow('fx=0.5, fy=0.5', img4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
resize()

# 이미지 이동
def shift():
    img = cv2.imread('musician.png')
    h, w = img.shape[:2]

    M = np.float32([[1, 0, 100], [0, 1, 50]])
    # x 방향으로 100, y 방향으로 50 픽셀만큼 이동 변환을 나타내는 2x3 매트릭스 생성

    img2 = cv2.warpAffine(img, M, (w, h))
    '''
    affine space : 원점이 어딘지 모르는 벡터 공간 https://m.blog.naver.com/PostView.nhn?blogId=jang_hwan_im&logNo=221289668071&proxyReferer=https:%2F%2Fwww.google.com%2F
    cv2.warpAffine(img, matrix, (width, height))
    1st : 변환할 소스 이미지
    2nd : 2x3 변환 매트릭스
    3rd : 출력될 이미지 사이즈
    * 이동하고 남은 여백은 검정색으로 처리된다.
    * 이미지의 한 픽셀을 (tx, ty) 만큼 이동하는 행렬은 2x3 행렬인 M = [[1, 0, tx], [0, 1, ty]] 이다.
    '''
    cv2.imshow('shift image', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
shift()

# 이미지 회전
def rotate():
    img = cv2.imread('musician.png')
    h, w = img.shape[:2]

    M1 = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
    M2 = cv2.getRotationMatrix2D((w/2, h/2), 90, 1)

    '''
    cv2.getRotationMatrix2D((center-coordinate), angle, scale )
    회전 변환 매트릭스 M = [[cos(), -sin()], [sin(), cos()]]
    * OpenCV 는 이미지 회전 중심을 지정할 수 있다. a = scale*cos(), b = scale.sin()
      M = [[a, b, (1-a)*center.x - b*center.y], [-b, a, b*center.x + (1-a)*center.y]]
    복잡한 식이지만 이를 쉽게 함수로 만든게 cv2.getRotationMatrix2D() 함수이다.
    '''
    img2 = cv2.warpAffine(img, M1, (w, h))
    img3 = cv2.warpAffine(img, M2, (w, h))

    cv2.imshow('45-Rotated', img2)
    cv2.imshow('90-rotated', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
rotate()

# 이미지 원근 보정하기
def perspective():
    img = cv2.imread('print.jpg')
    h, w = img.shape[:2]
    print(w, h)

    pts1 = np.float32([[140, 0], [20, 250], [350, 100]])
    pts2 = np.float32([[60, 0], [40, 315], [300, 170]])
    # cv2.warpAffine() 함수를 쓰려면 변환되기전 3개의 좌표와 변환된 후 3개의 좌표를 알아야 한다.
    M = cv2.getAffineTransform(pts1, pts2)

    img2 = cv2.warpAffine(img, M, (w, h))

    cv2.imshow('original', img)
    cv2.imshow('Affine-Transform', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
perspective()


