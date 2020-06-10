# https://m.blog.naver.com/samsjang/220501964041

import cv2
import numpy as np
# 행렬을 만들어 주는 zeros() 함수 사용을 위해서
from random import shuffle
# 무작위 색상 값을 추출하기 위해 shuffle() 함수를 이용해서 리스트를 무작위로 섞을 예정.
import math
# 마우스 왼쪽 버튼을 누른 지점과 왼쪽 버튼을 뗀 지점간 거리를 재는 데 필요한 sqrt() 함수를 사용하기 위해서

mode, drawing = True, False
# mode 변수는 직사각형, 원 그리기를 토글하기 위한 플래그
# drawing 변수는 마우스 왼쪽 버튼을 누르고 움직이면 도형을 그리고, 마우스 왼쪽 버튼을 떼고 움직일 때 도형이 안그려지도록 하는 플래그

ix, iy = -1, -1
# 마우스 왼쪽 버튼을 누른 지점으로 활용될 변수

b = [i for i in range(256)]
g = [i for i in range(256)]
r = [i for i in range(256)]
# BGR 색상값을 위해 사용될 0~255를 멤버로 하는 리스트 생성.
# list 안에 for 문을 포함하는 List comprehension 표현

def onMouse(event, x, y, flags, param):
# 마우스 이벤트를 처리할 콜백 함수로 cv2.setMouseCallback() 함수의 인자로 지정되어 호출된다.
# 1st : 마우스 이벤트
# 2nd : 마우스 이벤트가 일어난 x좌표
# 3rd : 마우스 이벤트가 일어난 y좌표
# 4th : 플래그 (여기서는 사용하지 않음)
# 5th : 사용자 데이터, 여기서는 img 로, 밑에 cv2.setMouseCallback() 로 전달될 예정

    # 그리기 세팅
    global ix, iy, drawing, mode, b, g, r
    if event == cv2.EVENT_LBUTTONDOWN: # 왼쪽 마우스 클릭 할 때 발생하는 이벤트
        drawing = True # 마우스 좌클시 drawing 변수를 True 로 설정하여 그리기 모드 on
        ix, iy = x, y # ix, iy 에 마우스 좌클릭이 발생한 좌표 대입
        shuffle(b), shuffle(g), shuffle(r) # 0~255 값을 가지는 리스트 b,g,r 을 무작위로 섞는다

    # 그리는 과정
    elif event == cv2.EVENT_MOUSEMOVE: # 마우스를 움직일 때 발생하는 이벤트
        if drawing: # drawing 이 True 일 경우 그리기 모드
            if mode: # mode 로 토글 조작, 초깃값 False
                pass # cv2.rectangle(param, (ix,iy), (x,y), (b[0], g[0], r[0], -1)) -> 사각형이 그려지는 경로를 보고 싶을 경우
            else: # mode 가 True 일 경우
                R = (ix-x)**2 + (iy-y)**2 # 원의 방정식
                R = int(math.sqrt(R)) # R -> R'^2
                cv2.circle(param, (ix,iy), R, (b[0],g[0],r[0]), -1) # 원이 그려지는 경로를 보여줌

    # 그리기 결과값 보여주기
    elif event == cv2.EVENT_LBUTTONUP: # 왼쪽 마우스 버튼을 뗐을 때 발생하는 이벤트
        drawing = False # 그리기 모드 off
        if mode:
            cv2.rectangle(param, (ix,iy), (x,y), (b[0],g[0],r[0]), -1)
        else:
            R = (ix-x)**2 + (iy-y)**2
            R = int(math.sqrt(R))
            cv2.circle(param, (ix,iy), R, (b[0],g[0],r[0]), -1)


def mouseBrush():
    global mode
    img = np.zeros((512,512,3), np.uint8)
    '''
    img = np.zeros((512,512,3), np.uint8)
    각종 도형을 그리기 위한 공간을 생성.
    numpy.zeros() 함수는 numpy 배열을 만들고 모든 값을 0으로 채우는 함수이다.
    여기서는 512 x 512 인 배열을 만드는데, 각 멤버가 (0, 0, 0)인 배열. 그리고, 채우는 데이터 타입은 uint8 이다.
    이를 이미지 차원에서 다시 설명하면, 512 x 512 크기의 검정색 이미지를 생성한 것과 같다.
    '''
    cv2.namedWindow('paint')
    cv2.setMouseCallback('paint', onMouse, param = img)
    # 'paint' 로 이름 붙힌 윈도우에서 발생하는 마우스 이벤트를 처리하기 위한 콜백 함수를 설정.
    # 콜백 함수는 onMouse() 이고, 콜백 함수로 전달할 사용자 데이터는 img 이다.

    while True:
        cv2.imshow('paint', img) # 실시간으로 이벤트 처리결과 보여줌
        k = cv2.waitKey(1) & 0xFF
        if k == 27 : # ESC 누르면 종료
            break
        elif k == ord('m'): # 키보드 'm' 누르면 모드 변환
            mode = not mode

    cv2.destroyAllWindows()

mouseBrush() # 함수 실행