# https://webnautes.tistory.com/1257
# Morphological Operation
# OpenCV에서 제공하는 Erosion, Dilation, Opening, Closing 를 연산하는 함수들을 다룹니다.
# 보통 Binary Image 에서 흰색으로 표현된 오브젝트의 형태를 개선하기 위해 사용됩니다.

import cv2
import numpy as np

img1 = cv2.imread('morphology_test.png')
img2 = cv2. imread('morphology_test2.png')

kernel_1 = np.ones((3, 3), np.uint8)
kernel_2 = np.ones((11,11), np.uint8)
# (n,n) : 커널의 크기. object 의 테두리를 얼마나 깎을건지 설정, 클수록 많이 깎임

cv2.imshow("source1", img1)
cv2.imshow("source2", img2)

# 1. Erosion
# 바이너리 이미지에서 흰색 오브젝트의 외곽 픽셀을 0(검은색)으로 만듭니다.
# 노이즈(작은 흰색 물체)를 제거하거나 붙어 있는 오브젝트들을 분리하는데 사용할 수 있습니다.
erosion = cv2.erode(img1, kernel_1, iterations = 1)
# iterations = n : errosion 을 몇 번 작업할 건지 설정. 클수록 이미지가 많이 깎임
cv2.imshow("erosion", erosion)


# 2. Dilation
# Erosion과 반대로 동작합니다. 바이너리 이미지에서 흰색 오브젝트의 외곽 픽셀 주변에 1(흰색)으로 추가합니다.
# 노이즈(작은 흰색 오브젝트)를 없애기 위해 사용한 Erosion에 의해서 작아졌던 오브젝트를 원래대로 돌리거나 인접해 있는 오브젝트들을 하나로 만드는데 사용할 수 있습니다.
dilation = cv2.dilate(img1, kernel_1, iterations = 1)
# iteration = m : dialtion 을 몇 번 작업할 건지 설정. 클수록 이미지가 팽창함
cv2.imshow("dilation", dilation)

# 3. Opening
# Erosion 연산 다음에 Dilation 연산을 적용합니다.  이미지 상의 노이즈(작은 흰색 물체)를 제거하는데 사용합니다.
# 노이즈(작은 흰색 오브젝트)를 없애기 위해 사용한 Erosion에 의해서 작아졌던 오브젝트에 Dilation 를 적용하면  오브젝트가 원래 크기로 돌아오게 됩니다.
opening = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel_2)
cv2.imshow("opening", opening)

# 4. Closing
# Opening과 반대로 Dilation 연산을 먼저 적용한 후,  Erosion 연산을 적용합니다.
# 흰색 오브젝트에 있는 작은 검은색 구멍들을 메우는데 사용합니다.
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel_2)
cv2.imshow("closing", closing)

cv2.waitKey(0)
cv2.destroyAllWindows()
