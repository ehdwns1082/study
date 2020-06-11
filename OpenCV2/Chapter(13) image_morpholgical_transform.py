# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220505815055&categoryNo=66&parentCategoryNo=0&viewDate=&currentPage=5&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=5
import cv2
import numpy as np

def morph1():
    img = cv2.imread('alp.bmp', cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((3,3), np.uint8)
    # 3x3 크기의 1로 채워진 매트릭스를 생성한다. erosion 밑 dilation 을 위한 kernel 로 사용될 예정
    erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(img, kernel, iterations=1)
    # 1st : 원본 이미지
    # 2nd : 커널
    # 3rd : 작업 반복 횟수

    cv2.imshow('original', img)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

morph1()

def morph2():
    img1 = cv2.imread('a.bmp', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('b.bmp', cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((5,5), np.uint8)

    opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
    '''
    cv2.morphologyEx(scr, operation, kernel)
    1st : 이미지 변형을 적용할 원본 이미지 
    2nd : 이미지 변형 오퍼레이션 종류
          cv2.MORPH_OPEN : opening 을 수행
          cv2.MORPH_CLOSE : closing 을 수행
          cv2.MORPH_GRADIENT : dilation 이미지와 erosion 이미지의 차이를 나타냄
          cv2.MORPH_TOPHAT : 원본 이미지와 opening 한 이미지의 차이를 나타냄
          cv2.MORPH_BLACKHAT : closing 한 이미지와 원본 이미지의 차이를 나타냄
    3rd : 적용할 커널 매트릭스
    '''
    cv2.imshow('a_original', img1)
    cv2.imshow('b_original', img2)
    cv2.imshow('a_opening', opening)
    cv2.imshow('b_closing', closing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

morph2()


def makeKernel():
    M1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # 직사각형 모양으로 5x5 크기의 커널 매트릭스 생성
    M2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # 타원 모양으로 5x5 크기의 커널 매트릭스 생성
    M3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    # 십자 모양으로 5x5 크기의 커널 매트릭스 생성
    print(' ')
    print('M1')
    print(M1)

    print(' ')
    print('M2')
    print(M2)

    print(' ')
    print('M3')
    print(M3)

makeKernel()