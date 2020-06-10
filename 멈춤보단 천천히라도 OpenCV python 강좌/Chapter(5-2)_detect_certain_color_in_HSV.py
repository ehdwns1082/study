# 이미지에서 파란색 (hue 값 120) 검출하기
# 유연한 검출 위해 hue 값 120에 +-10 해줘서 범위 잡아준다.
# https://m.blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220504633218&proxyReferer=https:%2F%2Fwww.google.com%2F

import cv2

img_CHolor = cv2.imread('hueolor_table.png')
#height, width = img_color.shape[:2] -> 왜 쓴건지 ?

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
lower_blue = (120-10, 30, 30)
upper_blue = (120+10, 255, 255)
# 범위 설정 하여 hsv 이미지에서 원하는 색 이미지를 binary 이미지로 설정
# saturaion 과 value 에 30 과 255을 준 이유는 너무 어두워서 검은색에 가깝거나, 밝아서 흰색에 가까운 경우를 제외
# lower_blue 를 (120-10,0,0) 으로 주면 노이즈(완전 검정인 부분)가 껴서 30 정도를 threshold 값으로 잡아야 걸러지는 듯

img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
# cv2.inRange() 함수는 소스인 hsv 의 모든 값을 lower_blue, upper_blue 를 통해 지정한 범위에 있는지 체크한 후,
# 범위에 해당하는 부분은 그대로, 나머지 부분은 0(black) 으로 채워서 결과값을 반환한다.

img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask)
# binary 이미지를 mask로 사용하여 원본 이미지에서 범위 값에 해당하는 부분을 획득

cv2.imshow('img_color', img_color)
cv2.imshow('img_result', img_result)
cv2.waitKey(0)
cv2.destroyWindow()