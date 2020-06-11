# https://webnautes.tistory.com/1246
# 텐서플로우 색 인식 테스트 ( Tensorflow Color Recognition Test )
# webnautes.tistory.com/1256
import cv2
import numpy as np

hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

def mouse_callback(event, x, y, flags, param):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3

    if event == cv2.EVENT_LBUTTONDOWN:
        # 마우스로 누른 픽셀의 정보를 가져온다 !!!
        # 픽셀이 여러개 있는데 그 중에서 마우스 클릭한 픽셀의 좌표 [y, x]에 해당하는 픽셀의 [B G R] 값을 가져옴
        print('img_color[y, x] : ', img_color[y, x]) # => 이미지 상의 모든 픽셀은 2차원 좌표를 가진다. 왼쪽 위가 (0, 0)
        color = img_color[y, x] # => img_color 라는 이미지의 마우스로 눌린 픽셀(y, x)의 정보 [B G R]

        one_pixel = np.uint8([[color]]) # => [[[B G R]]]
        hsv = cv2.cvtColor(one_pixel, cv2.COLOR_BGR2HSV) # => [[[H S V]]]
        hsv = hsv[0][0] # => [H S V]

        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 픽셀값의 범위를 정합니다.
        if hsv[0] < 10: # hsv[0] => H
            print("case1")
            # array() 함수로 1x3 행렬 생성
            # user 가 편하게 인지할 수 있도록 검출하기 위해 범위지정 +-10' 를 해주는데, 하한값에 -10 해준 결과가 음수값이므로 +180 하여 양수로 변환(같은 값)
            # Saturation 과 Value 를 30으로 한 이유는 너무 어두워서 검은색에 가까운값, 색이 너무 옅어서 흰색에 가까운 경우를 없애기 위함.
            # case3 가 일반적인 경우이고 case1 과 case2는 예외의 경우이다 !!
            # 왜냐하면 범위값이 +-10' 인데 0=180' 근처에서 +-10' 연산시 음수가 나와 값이 없어지는 경우가 있기 때문.  cf) uint8 자료형은 0~255 까지의 값만 있음
            lower_blue1 = np.array([hsv[0]-10+180, 30, 30])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, 30, 30])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0], 30, 30])
            upper_blue3 = np.array([hsv[0]+10, 255, 255])
            #     print(i-10+180, 180, 0, i)
            #     print(i, i+10)

        elif hsv[0] > 170:
            print("case2")
            lower_blue1 = np.array([hsv[0], 30, 30])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, 30, 30])
            upper_blue2 = np.array([hsv[0]+10-180, 255, 255])
            lower_blue3 = np.array([hsv[0]-10, 30, 30])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, 180, 0, i+10-180)
            #     print(i-10, i)
        else:
            print("case3")
            lower_blue1 = np.array([hsv[0], 30, 30])
            upper_blue1 = np.array([hsv[0]+10, 255, 255])
            lower_blue2 = np.array([hsv[0]-10, 30, 30])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0]-10, 30, 30])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, i+10)
            #     print(i-10, i)

        print(hsv[0]) # => H
        print("@1", lower_blue1, "~", upper_blue1)
        print("@2", lower_blue2, "~", upper_blue2)
        print("@3", lower_blue3, "~", upper_blue3)


cv2.namedWindow('img_color')
cv2.setMouseCallback('img_color', mouse_callback)



while(True):
    img_color = cv2.imread('hue_color_table.png')
    height, width = img_color.shape[:2]
    img_color = cv2.resize(img_color, (width, height), interpolation=cv2.INTER_AREA)

    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv2.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask = img_mask1 | img_mask2 | img_mask3
    '''
    '|' 기호는 'Vertical Bar' 또는 'Pipe Symbol' 이라고 한다. Bitwise Operation 에서 or 와 같은 역할을 한다.
    # 예제 ) 0b => 2진수, 0o => 8진수, 0x => 16진수
    line 1) >>> a=4
    line 2) >>> bin(a)
    line 3) '0b100' # => 100(2)
    line 4) >>> b=5
    line 5) >>> bin(b)
    line 6) '0b101' # => 101(2)
    line 7) >>> a|b 
    line 8) 5 # => 101(2)
    line 9) >>> c=a|b
    line 10) >>> bin(c)
    line 11) '0b101' # => 101(2)
    '''

    print('img_mask : ',img_mask)

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)


    cv2.imshow('img_color', img_color)
    cv2.imshow('img_mask', img_mask)
    cv2.imshow('img_result', img_result)


    # ESC 키누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break


cv2.destroyAllWindows()