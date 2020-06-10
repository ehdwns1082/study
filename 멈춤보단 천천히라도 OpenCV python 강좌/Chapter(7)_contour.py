# 컨투어 : https://webnautes.tistory.com/1270

import cv2 as cv

# 이미지 로드 & 바이너리 변환
img_color = cv.imread('contour_test1.png')
img_color2 = cv.imread('contour_test2.png')
img_colorh = cv.imread('contour_testh.jpg')
cv.imshow('orignal', img_color)
cv.imshow('orignal2', img_color2)
cv.imshow('orignalh', img_colorh)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
img_gray2 = cv.cvtColor(img_color2, cv.COLOR_BGR2GRAY)
img_grayh = cv.cvtColor(img_colorh, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
ret2, img_binary2 = cv.threshold(img_gray2, 127, 255, 0)
ret_h, img_binary_h = cv.threshold(img_grayh, 127,255, 0)
# 컨투어 하려면 먼저 binary 이미지로 만들어 줘야 한다.

# 컨투어 검출
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv.findContours(img_binary2, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
contours_h, hierarchy_h = cv.findContours(img_binary_h, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

'''
cv.findContours 함수가 길이가 2인 튜플을 반환한다.
contours 에는 검출된 컨투어, object 의 외곽선을 구성하는 점들의 좌표가 list 로 저장됨 (c++에서는 벡터로 저장됨)
hierarchy 에는 검출된 컨투어의 정보를 구조적으로 저장, 파이썬에서는 list 로 저장

cv.findContours(image, mode, method, [, contours[, hierarchy[, offset]]])

1st : image : 입력이미지, 흑백으로 binary 해야함

2nd : mode : Contour Retrieval Mode 를 의미. 검출된 엣지 정보를 계층 또는 list로 지정하는 방식을 지정
=> 컨투어를 찾은 결과를 어떻게 리턴할 지 결정 
=> RETR_TREE : 컨투어 내부에 다른 컨투어가 있을 경우 hierarchy 구조로 만들어 줌. pint(hierarchy) >>> [Next, Previous, First_Child, Parent] 로 표시 
=> RETR_LIST : 모든 컨투어가 같은 hierarchy level 을 같는다. print(hierarchy) >>> [Next, Previous, -1, -1]
=> RETR_EXTERNAL : 가장 외곽에 있는 컨투어만 리턴한다. 
=> RETR_CCOMP : 모든 컨투어를 2개의 hierarchy level 으로 재구성 한다. 외부에 있는 컨투어는 레벨 1이되고 오브젝트 내부에 있는 컨투어는 레벨 2가 된다. 

3rd : method : contour approximation method 를 의미. 컨투어를 구성하는 포인트 검출 방법 지정
=> cv.CHAIN_APPROX_NONE : 모양을 구성하는 모든 경계점 표시
=> cv.CHAIN_APPROX_SIMPLE : 컨투어를 구성하는 선 일부가 직선일 경우 그 시작점과 끝점만 저장, 메모리 절약 

4th : offset : 지정한 크기만큼 컨투어를 구성하는 포인트의 좌표를 이동하여 저장.
=> ROI(Region Of Interest) 를 사용하여 이미지 일부에서 컨투어를 추출한 후 전체 이미지에서의 좌표를 구할 떄 유용하다.
=> 마우스 클릭으로 ROI 영역 추출하기 : https://gaussian37.github.io/vision-opencv-roi-extraction/
'''

# 컨투어 그리기

# image_color : cv.RETR_LIST, cv.CHAIN_APPROX_NONE
cv.drawContours(img_color, contours, 0, (255, 0, 0), 3)  # blue/ thickness 3
cv.drawContours(img_color, contours, 1, (0, 255, 0), 7)  # green/ thickness 7

# image_color2 : cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
for cnt in contours2 :  # for 문을 하나 거칠 때마다 list 의 겹이 쌓인다.
    for p in cnt :
        cv.circle(img_color2, (p[0][0], p[0][1]), 3, (255,0,0), -1) # 컨투어에 포함된 모든 좌표마다 원을 그리는 코드

print("contours2 : ")
print(contours2)

# cv.circle(img, center, radius, color, thickness, lineType, shift)
# img : 원이 그려질 이미지
# center : 원의 중심 좌표 ( x, y )
# radius : 원의 반지름
# color : 원의 색( B, G, R )
# thickness : 선굵기(디폴트값 1), 음수면 내부를 다 칠함
# lineType : 디폴트값 cv.LINE_8(=8-connected line)
# shift : 디폴트값 0
# https://webnautes.tistory.com/1207

# image_color_hierarchy : cv.RETR_LIST, cv.CHAIN_APPROX_NONE
for cnt in contours_h:
    cv.drawContours(img_colorh, [cnt], 0 , (0,0,255),3)
print("hierarchy_h : ")
print(hierarchy_h)
'''
cv.drawContours(image, contours, contourIdx,color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]])
image : 컨투어를 그릴 대상 이미지 (원본 이미지)
contours : 이미지 위에 그릴 컨투어가 저장된 list (c++ 에서는벡터)
contourIdx : 이미지에 그릴 특정 컨투어의 index. 음수로 지정하면 모든 컨투어를 그림
color : 컨투어를 그릴 때 사용할 색상 지정. B,G,R 순서임.
thickness : 컨투어를 그릴 때 선의 굵기. 음수이면 내부를 채움.
'''
cv.imshow("result", img_color)
cv.imshow("result2", img_color2)
cv.imshow("result_h", img_colorh)



cv.waitKey(0)
cv.destroyAllWindows()