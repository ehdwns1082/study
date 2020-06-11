
import cv2
import numpy as np

def onChange(x):
    pass
# 트랙바 이벤트를 처리할 콜백 함수.
# 여기서는 트랙바 이벤트가 발생할 때 처리할 일이 아무것도 없으므로 pass 이다.

def trackbar():
    img = np.zeros((200,512,3), np.uint8)
    cv2. namedWindow('color_palette')
    # 200x512 크기의 그림판을 생성하고 'color_palette' 라는 이름의 윈도우를 생성한다.
    cv2.createTrackbar('B', 'color_palette', 0, 255, onChange)
    cv2.createTrackbar('G', 'color_palette', 0, 255, onChange)
    cv2.createTrackbar('R', 'color_palette', 0, 255, onChange)
    # 'color_palette' 윈도우에 0~255까지 값으로 변경 가능한 트랙바 B,G,R을 생성한다.
    switch = '0: 0FF\n1: ON' # \n : 줄바꿈, switch = 문자열
    cv2.createTrackbar(switch, 'color_palette', 0, 1, onChange)
    # On/Off 스위치 역할을 할 트랙바 생성

    while True:
        cv2.imshow('color_palette', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        b = cv2.getTrackbarPos('B', 'color_palette')
        g = cv2.getTrackbarPos('G', 'color_palette')
        r = cv2.getTrackbarPos('R', 'color_palette')
        s = cv2.getTrackbarPos(switch, 'color_palette')
        # 트랙바의 현재값을 가져옴

        if s == 0: # 스위치가 off 이면
            img[:] = 0 # 그림판의 BGR 을 0,0,0 (검정) 으로 만듬
        else: # 스위치가 on 이면
            img[:] = [b,g,r] # 그림판의 BGR을 (b,g,r) (트랙바 값) 으로 만듬

    cv2.destroyAllWindows()

trackbar()
