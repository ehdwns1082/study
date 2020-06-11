'''
Step 1 : 노이즈 제거 (Noise reduction)
이미지에서 노이즈가 있으면 edge 를 제대로 찾는 것이 어려울 수 있다.
따라서 첫단계로 5x5 Gaussian filter 를 이용해 이미지의 노이즈를 쥴여준다.

Step 2 : Gradient 값이 높은 부분 찾기
Gaussian filter 로 노이즈가 제거된 이미지에 Sobel 커널을 x, y 방향으로 적용하여 각 방향의 Gradient 를 얻는다.
수평방향 Gradient : Gx, 수직방향 Gradient : Gy 라고 하면
Edge_Gradient(G) = sqrt( (Gx)^2 + (Gy)^2) )
Angle(theta) = arctan(Gy/Gx)
* Gradient 의 방향은 Edge 에 수직인 방향이다.

Step 3 : 최대값이 아닌 픽셀의 값을 0으로 만들기
2단계를 거친 후 edge 에 기여하지 않은 픽셀을 제거하기 위해 이미지 전체를 스캔한다.
이미지를 스캔하는 동안 gradient 방향으로 스캔구역에서 gradient 의 최댓값을 가진 픽셀을 찾는다.

Step 4 : Hyteresis Thresholding
4단계는 3단계를 거친 것들이 실제 edge 인지 아닌지 판단하는 단계이다.
먼저 threshold 를 minVal, masVal 2개 잡는다.
maxVal 보다 높은 부분은 확실한 edge 이고, minVal 보다 낮은 부분은 edge 가 아니라고 판단한다.
minVal, masVal 의 사이에 있는 값들은 이 픽셀들의 연결 구조를 보고 edge 인지 아닌지 판단하다.

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny():
    img = cv2.imread('musician.png', cv2.IMREAD_GRAYSCALE)

    edge1 = cv2.Canny(img, 50, 200)
    edge2 = cv2.Canny(img, 100, 200)
    edge3 = cv2.Canny(img, 170, 200)
    '''
    cv2.Canny(img, minVal, maxVal)
    1st : 원본 이미지 (Gray scale)
    2nd : minimum thresholding value
    3rd : maximum thresholding value
    '''


    cv2.imshow('original_Gray', img)
    cv2.imshow('Canny_Edge1', edge1)
    cv2.imshow('Canny_Edge2', edge2)
    cv2.imshow('Canny_Edge3', edge3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

canny()



















