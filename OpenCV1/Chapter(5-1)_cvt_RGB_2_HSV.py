"""
HSV = Hue_Saturation_Value
Hue(색상) : 0'(빨강) ~ 359' 스펙트럼별로 색 분류, 빨강이 파장 젤 길어서 0'
=> OpenCV에서는 0' ~ 179' 까지 범위를 가짐; 원래 Hue 값에 1/2 곱해야 함.
Saturation(채도) : 0%(무채색, 흰색) ~ 100%(가장 진함)
=> OpenCV 에서는 0 ~ 255까지 범위를 가짐.
Value(명도) : 0%(검은색) ~ 100%(빨간색)
=> OpenCV에서는 0 ~ 255까지 범위를 가짐.

OpenCV 에서는 픽셀이 RGB 순이 아니라 BGR 순서이다.
=> RGB 이미지를 처리할 때 R과 B 순서를 바꿔줘야 정상적으로 인식함.
OpenCV 에서는 픽셀 좌표가 (y,x) 이다. 아래로 갈수록 y증가, 오른쪽으로 갈수록 x증가
"""

import numpy as np
# Numpy(Numerical Python, 넘파이)는 C언어로 구현된 파이썬 라이브러리로써, 고성능의 수치계산을 위해 제작되었으며, 벡터 및 행렬 연산에 있어서 매우 편리한 기능을 제공한다.
# 또한 이는 데이터분석을 할 때 사용되는 라이브러리인 pandas와 matplotlib의 기반으로 사용되기도 합니다.
# numpy에서는 기본적으로 array라는 단위로 데이터를 관리하며 이에 대해 연산을 수행한다. array는 말그대로 행렬이라는 개념으로 생각하면 됨.
# 먼저 numpy를 사용하기 위해서는 아래와 같은 코드로 numpy를 import해야 합니다.
# https://doorbw.tistory.com/171

import cv2

color = [255, 0, 0] # BGR; 파란색
pixel = np.uint8([[color]])
# cvtColor 함수의 입력으로 사용할 수 있도록 한 픽셀로 구성된 이미지로 변환 (BGR 3개에 대한 값을 하나의 값으로)
# uint : 부호없는 정수, 자료형 뒤의 숫자는 bit 수를 의미한다. ex) uint8 -> 8bit의 부호 없는 정수
# cf) int8 은 2^8개의 정수 -128~127을 표현할 수 있으며, uint8 은 2^8개의 양수인 정수 0~255를 표현할 수 있다.
# https://kongdols-room.tistory.com/53
# NumPy 자료형 : https://eunguru.tistory.com/216
# Python 에서 NumPy import 하여 array 이용하기 : https://datascienceschool.net/view-notebook/35099ac4aea146c69cc4b3f50aec736f/

hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
# cvtColor 함수로 HSV 색공간으로 변환
hsv = hsv[0][0]
# hsv값을 출력하기 위해 픽셀값만 가져옴

print("bgr: ", color)
print("hsv: ", hsv)
