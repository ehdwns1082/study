# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220516697251&categoryNo=66&parentCategoryNo=0&viewDate=&currentPage=5&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=5

# OpenCV 에서 Contour 찾기는 검정색 배경에서 흰색 물체를 찾는 것이다!

import cv2
import numpy as np

def contour():
    img = cv2.imread('musician.png', cv2.IMREAD_GRAYSCALE)

    ret, thr = cv2.threshold(img, 127, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    1st : contour 찾기를 할 소스 이미지. thresholding을 통해 변환된 바이너리 이미지어야 함
    2nd : contour 추출방법, 2번째 리턴값인 hierarcy 의 값에 영향을 줌
        cv2.RETR_TREE: 두 번째 인자는 contour 추출 모드이며, 2번째 리턴값인 hierarchy의 값에 영향을 줌
        cv2.RETR_EXTERNAL: 이미지의 가장 바깥쪽의 contour만 추출
        cv2.RETR_LIST: contour 간 계층구조 상관관계를 고려하지 않고 contour를 추출
        cv2.RETR_CCOMP: 이미지에서 모든 contour를 추출한 후, 2단계 contour 계층 구조로 구성함. 1단계 계층에서는 외곽 경계 부분을, 2단계 계층에서는 구멍(hole)의 경계 부분을 나타내는 contour로 구성됨
        cv2.RETR_TREE: 이미지에서 모든 contour를 추출하고 Contour들간의 상관관계를 추출함
    3rd : contour 근사 방법
        cv2.CHAIN_APPROX_SIMPLE: 세 번째 인자는 contour 근사 방법임
        cv2.CHAIN_APPROX_NONE: contour를 구성하는 모든 점을 저장함. 
        cv2.CHAIN_APPROX_SIMPLE: contour의 수평, 수직, 대각선 방향의 점은 모두 버리고 끝 점만 남겨둠. 예를 들어 똑바로 세워진 직사각형의 경우, 4개 모서리점만 남기고 다 버림
        cv2.CHAIN_APPROX_TC89__1: Teh-Chin 연결 근사 알고리즘(Teh-Chin chain approximation algorithm)을 적용함

    '''
    cv2.drawContours(img, contours, -1, (255,0,255), 2)
    '''
    cv2.drawContours()
    1st : contour 대상 이미지
    2nd : 이미지에 그릴 contour. 이 값은 cv2.findContours() 함수의 2번째 리턴 값으로 리스트형 자료임. i번째 contour 의 첫 번째 픽셀 좌표는 contours[i][0]과 같이 접근 가능
    3rd : 이미지에 실제로 그릴 contour 인덱스 파라미터. 이 값이 음수이면 모든 contour 를 그림
    4th : contour 선의 BGR 색상값. 여기서는 Green으로 지정했음
    5th : contour 선의 두께
    '''
    cv2.imshow('thresh', thr)
    cv2.imshow('contour', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

contour()

