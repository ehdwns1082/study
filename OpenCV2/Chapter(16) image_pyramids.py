# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220508552078&categoryNo=66&parentCategoryNo=0&viewDate=&currentPage=5&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=5

'''
* 이미지 피라미드
이미지 작업 시에 동일한 이미지임에도 여러 해상도의 다양한 크기를 가지고 작업을 할 떄가 있다.
예를 들면 사람의 얼굴과 같이 객체에서 특정한 무언가를 탐색할 떄가 이 경우이다.
우리가 확보한 얼굴 이미지가 있다고 해도, 서치한 동일 이미지가 크기가 달라서 인식을 못할 수가 있다.
이런식으로 활용하도록 다양한 해상도의 이미지 세트를 만든 것을 이미지 피라미드라고 한다.

1. 가우시안 피라미드 (Gaussian pyramids)
=> DownSampling 과 UpSampling 2가지가 있다.
   DownSampling 은 이미지의 짝수열, 짝수행에 해당하는 픽셀을 제거함으로써 이미지 해상도를 줄인다.
   UpSampling 은 이미지의 짝수열, 짝수행에 픽셀을 추가하여 해상도는 증가하지만 블러링 효과를 입힌듯한 이미지로 보인다.
   1) cv2.pyrDown(src)
   2) cv2.pyrUP(src)

2. 라플라시안 피라미드 (Laplacian Pyramids)
=> 가우시안 피라미드의 결과로 생성한다.
   원본 이미지를 다운샘플링 한다음에 업샘플링 하면 해상도가 같지만 블러링된 이미지가 나온다.
   원본 이미지에서 블러링된 이미지를 ' - ' 연산하면 라플라시안 피라미드의 최하위 단계가 완성된다.

'''

import cv2
import numpy as np

def Down_pyramid():
    img = cv2.imread('musician.png', cv2.IMREAD_GRAYSCALE)
    tmp = img.copy()

    win_titles = ['org', 'level1', 'level2', 'level3']
    g_down = []
    g_down.append(tmp)

    for i in range(3):
        tmp1 = cv2.pyrDown(tmp)
        g_down.append(tmp1)
        tmp = tmp1

    for i in range(4):
        cv2.imshow(win_titles[i], g_down[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

Down_pyramid()

def Up_pyramid():
    img = cv2.imread('musician.png', cv2.IMREAD_GRAYSCALE)
    tmp = img.copy()

    win_titles= ['org', 'level1', 'level2', 'level3']
    g_down = []
    g_up = []

    g_down.append(tmp)

    for i in range(3):
        tmp1 = cv2.pyrDown(tmp)
        g_down.append(tmp1)
        tmp = tmp1
    cv2.imshow('level3', tmp)

    for i in range(3):
        tmp = g_down[i+1]
        tmp1 = cv2.pyrUp(tmp)
        g_up.append(tmp1)

    for i in range(3):
        cv2.imshow(win_titles[i], g_up[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

Up_pyramid()

def Lapacian_pyramids():
    img = cv2.imread('musician.png', cv2.IMREAD_GRAYSCALE)
    tmp = img.copy()

    win_titles = ['org', 'level1', 'level2', 'level3']
    g_down = []
    g_up = []
    img_shape = []

    g_down.append(tmp)
    img_shape.append(tmp.shape)

    for i in range(3):
        tmp1 = cv2.pyrDown(tmp)
        g_down.append(tmp1)
        img_shape.append(tmp1.shape)
        tmp = tmp1

    for i in range(3):
        tmp = g_down[i+1]
        tmp1 = cv2.pyrUp(tmp)
        tmp = cv2.resize(tmp1, dsize=(img_shape[i][1], img_shape[i][0]), interpolation=cv2.INTER_CUBIC)
        # cv2.subtract 함수에 들어가는 이미지 크기를 동일하게 맞추는 작업

        g_up.append(tmp)

    for i in range(3):
        tmp = cv2.subtract(g_down[i], g_up[i]) # cv2.subtract() 함수의 인자들은 크기가 같아야 한다.

        cv2.imshow(win_titles[i], tmp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

Lapacian_pyramids()

