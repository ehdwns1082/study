import cv2

def nothing(x):
    pass
# 더미함수, 실질적으로 필요는 없는데 트랙바 만드려면 있어야 함
# 트랙바를 조정할 때 마다 실행되는 callback 함수를 정의해야 한다. 트랙바를 조절할 때마다(각각 새로운 input) 따로 실행할 명령이 없기 때문에 아무일도 하지 않는 더미 함수를 만듬
# https://webnautes.tistory.com/795

cv2.namedWindow('Binary')
# 트랙바 붙힐 윈도우 생성

cv2.createTrackbar('threshold', 'Binary', 0, 255, nothing)
# 트랙바를 이름이 Binary인 윈도우에 붙힘
# 1st = 트랙바 이름
# 2nd = 트랙바 붙힐 윈도우 이름
# 3rd = 트랙바로 조정할 최솟 값 (value)
# 4th = 트랙바로 조정할 최대 값 (count)
# 5th = 더미함수 (onChange)

cv2.setTrackbarPos('threshold', 'Binary', 100)
# 트랙바의 초기값을 100로 설정
# 1st = 트랙바 이름
# 2nd = 윈도우 이름
# 3rd = 초기값

img_CHolor = cv2.imread('ball_sample.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('Color', img_color)
# cv2.waitKey(0)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
# cv2.imshow('Gray', img_gray)
# cv2.waitKey(0)

while(True):
    # 트랙바 이동시 결과를 바로 확인할 수 있도록 루프 추가

    low = cv2.getTrackbarPos('threshold', 'Binary')
    # 트랙바의 현재값을 가져와 threshold 값으로 사용할 수 있게 함
    # 1st = 트랙바 이름
    # 2nd = 윈도우 이름

    ret, img_binary = cv2.threshold(img_gray, low, 255, cv2.THRESH_BINARY)
    # 1st = 이진화 할 대상 이미지(gray scale 이어야 함)
    # 2nd = threshold
    # 3rd = threshold 이상값을 이 값으로 바꿈
    # 4th = threshold 기준으로 픽셀값을 binary 하게 나눔(크면 3rd 값으로, 작으면 0으로)
    # 0 = black, 255 = white
    # cv2.threshold() 의 값이 길이가 2인 튜플로 반환되므로 이를 unpack 해주는 과정이 필요하다.
    # cv2.threshold() 에서 나온 2개의 값 중 앞의 값은 ret 에 뒤의 값은 img_binary 에 저장하는데, ret 에 저장된 값은 안쓰는 값이라 무시하는 듯.
    # ret 설명 https://leechamin.tistory.com/251


    cv2. imshow("Binary", img_binary)

    img_result = cv2.bitwise_and(img_color, img_color, mask = img_binary)
    # 원본 이미지와 binary 이미지를 and 연산 (이미지 비트연산)
    # 색이 없으면 검은색(0), 색이 있으면 흰색(1), and 연산 = 두 그림모두 흰색(1)인 부분만 흰색
    # 만약에 mask 영역이 반대라면 cv2.THRESH_BINARY 대신 cv2.THRESH_BINARY_INV(인버스) 쓰면 반전됨
    # https://copycoding.tistory.com/156
    # 이미지 연산 : https://opencv-python.readthedocs.io/en/latest/doc/07.imageArithmetic/imageArithmetic.html
    # 이미지 믹스 : https://m.blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220503082434&proxyReferer=https%3A%2F%2Fwww.google.com%2F

    cv2.imshow('Result', img_result)

    if cv2.waitKey(1)&0xff == 27:
        break
        # esc키를 누르면 루프에서 빠져나올 수 있음

cv2.destroyWindow()