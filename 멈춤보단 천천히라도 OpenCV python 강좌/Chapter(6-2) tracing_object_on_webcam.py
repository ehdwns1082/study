import cv2 as cv
import numpy as np

hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

def nothing(x):
    pass
# 트랙바 쓰기 위한 더미함수

# 2. mouse callback (event)
def mouse_callback(event, x, y, flags, param):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3, threshold
    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환
    if event == cv.EVENT_LBUTTONDOWN:
        print('in mouse callback : if LBUTTONDOWN')

        color = img_color[y, x] # => [B G R]
        # img_color 변수는 밑의 VideoCapture() 함수의 값을 저장한다.
        print('color = img_color[y,x] : ', color)

        one_pixel = np.uint8([[color]]) # => [[[B G R]]]
        # cvtColor 함수에서 input 으로 받을 수 있도록 형태를 맞춰준다.
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV) # => [[[H S V]]]
        print('hsv : ', hsv)

        hsv = hsv[0][0] # => [H S V]
        print('hsv[0][0] : ', hsv)
        # 맨 앞의 필요한 값만 불러오기

        threshold = cv.getTrackbarPos('threshold', 'img_result')

        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
        if hsv[0] < 10:
            print("case1")
            # 변수 = np.array(Hue, Saturation, Value)
            lower_blue1 = np.array([hsv[0] - 10 + 180, threshold, threshold])
            # hue 값이 10보다 작으므로
            upper_blue1 = np.array([180, 255, 255])
            #
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0], threshold, threshold])
            upper_blue3 = np.array([hsv[0] + 10, 255, 255])
            #     print(i-10+180, 180, 0, i)
            #     print(i, i+10)

        elif hsv[0] > 170:
            print("case2")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0] + 10 - 180, 255, 255])
            lower_blue3 = np.array([hsv[0] - 10, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, 180, 0, i+10-180)
            #     print(i-10, i)
        else:
            print("case3")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([hsv[0] + 10, 255, 255])
            lower_blue2 = np.array([hsv[0] - 10, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0] - 10, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, i+10)
            #     print(i-10, i)

        print(hsv[0]) # => hue 값
        print("@1", lower_blue1, "~", upper_blue1)
        print("@2", lower_blue2, "~", upper_blue2)
        print("@3", lower_blue3, "~", upper_blue3)


cv.namedWindow('img_color')
cv.setMouseCallback('img_color', mouse_callback)

cv.namedWindow('img_result')
cv.createTrackbar('threshold', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('threshold', 'img_result', 30)

cap = cv.VideoCapture(0)

# 1. load webcam (main structure)
while (True):
    print('in lode webcam')
    ret, img_color = cap.read()
    print('ret : ', ret)
    print('img_color : ', img_color)
    height, width = img_color.shape[:2]
    # array.shape = 행렬의 행과 열의 갯수를 알려줌
    # ex) img_color 는 480x640 행렬이므로 img_color.shape = (480, 640), img_color.shape[0] = 480, img_color.shape[1] = 640
    # 따라서 height, width = img_color.shape[:2] 는 튜플 img_color.shape = (1,2) 의 값을 unpack 하여 각각 height 와 width 에 저장하는 코드이다.
    print('height :', height)
    print('width : ', width)
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)
    # cv2.resize(원본 이미지, 결과 이미지 크기, 보간법) '결과 이미지 크기' 항목은 (너비, 높이) 의 튜플 이다.
    print('resized img_color : ', img_color)
    '''
    보간법 종류
    cv2.INTER_NEAREST 이웃 보간법
    cv2.INTER_LINEAR 쌍 선형 보간법
    cv2.INTER_LINEAR_EXACT 비트 쌍 선형 보간법
    cv2.INTER_CUBIC	바이큐빅 보간법
    cv2.INTER_AREA 영역 보간법
    cv2.INTER_LANCZOS4 Lanczos 보간법
    '''

    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
    # 원본 영상을 HSV 영상으로 변환
    print('img_hsv : ', img_hsv)
    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    kernel = np.ones((11, 11), np.uint8)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv.bitwise_and(img_color, img_color, mask=img_mask)

    numOfLabels, img_label, stats, centroids = cv.connectedComponentsWithStats(img_mask)

    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])
        print(centerX, centerY)

        if area > 50:
            cv.circle(img_color, (centerX, centerY), 10, (0, 0, 255), 10)
            cv.rectangle(img_color, (x, y), (x + width, y + height), (0, 0, 255))

    cv.imshow('img_color', img_color)
    cv.imshow('img_mask', img_mask)
    cv.imshow('img_result', img_result)

    # ESC 키누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()