# https://webnautes.tistory.com/1270
# https://m.blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220516822775&proxyReferer=https:%2F%2Fwww.google.com%2F

import cv2 as cv
import numpy as np


# 영역 크기
print('영역크기')
img_color = cv.imread('contour_test1.png')
# cv.imshow('original', img_color)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:  # contour
    cv.drawContours(img_color, [cnt], 0, (255, 0, 0), 3)  # blue
cv.imshow("area", img_color)

for cnt in contours:  # area
    area = cv.contourArea(cnt)
    print('area : ', area)

print(' ')


# 근사화
print('근사화')
img_color_approx = cv.imread('square_approx.png')
cv.imshow('original_approx', img_color_approx)
img_gray_approx = cv.cvtColor(img_color_approx, cv.COLOR_BGR2GRAY)
ret_approx, img_binary_approx = cv.threshold(img_gray_approx, 127, 255, 0)
contours_approx, hierarchy_approx = cv.findContours(img_binary_approx, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

for cnt_approx in contours_approx:  # contour
    cv.drawContours(img_color_approx, [cnt_approx], 0, (255, 0, 0), 3)  # blue

for cnt_approx in contours_approx:  # approximation
    epsilon = 0.02 * cv.arcLength(cnt_approx, True)
    approx = cv.approxPolyDP(cnt_approx, epsilon, True)
    print('len(approx) : ', len(approx))
    cv.drawContours(img_color_approx,[approx],0,(0,255,255),5)  # yellow

cv.imshow("result_approx", img_color_approx)


# 무게중심
img_color_COM = cv.imread('contour_test1.png')
# cv.imshow('original_COM', img_color_COM)
img_gray_COM = cv.cvtColor(img_color_COM, cv.COLOR_BGR2GRAY)
ret_COM, img_binary_COM = cv.threshold(img_gray_COM, 127, 255, 0)
contours_COM, hierarchy_COM = cv.findContours(img_binary_COM, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

for cnt_COM in contours_COM:  # contour
    cv.drawContours(img_color_COM, [cnt_COM], 0, (255, 0, 0), 3)  # blue

for cnt_COM in contours_COM:  # center of mass
    M = cv.moments(cnt_COM)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv.circle(img_color_COM, (cx, cy), 10, (0,0,255), -1)  # red

cv.imshow("result_COM", img_color_COM)


# 경계 사각형(Bounding Rectangle) : object 를 둘러싸는 최소 직사각형
img_color_B = cv.imread('contour_test1.png')
img_gray_B = cv.cvtColor(img_color_B, cv.COLOR_BGR2GRAY)
ret_B, img_binary_B = cv.threshold(img_gray_B, 127, 255, 0)
contours_B, hierarchy_B = cv.findContours(img_binary_B, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

for cnt_B in contours_B:  # contour
    cv.drawContours(img_color_B, [cnt_B], 0, (255, 0, 0), 3)  # blue

for cnt_B in contours_B:  # normal bounding rectangle
    x, y, w, h = cv.boundingRect(cnt_B)
    cv.rectangle(img_color_B, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green

for cnt_B in contours_B:  # minimal bounding rectangle
    rect = cv.minAreaRect(cnt_B)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img_color_B,[box],0,(0,0,255),2)  # red

cv.imshow("result_B", img_color_B)


# Convex Hull : 컨투어를 모두 포함하는 볼록 다각형을 그린다 cf) convex <-> concave
img_color_CH = cv.imread('hand.png')
img_gray_CH = cv.cvtColor(img_color_CH, cv.COLOR_BGR2GRAY)
ret_CH, img_binary_CH = cv.threshold(img_gray_CH, 127, 255, 0)
contours_CH, hierarchy_CH = cv.findContours(img_binary_CH, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

for cnt_CH in contours_CH:  # contour
    cv.drawContours(img_color_CH, [cnt_CH], 0, (255, 0, 0), 3)  # blue

for cnt_CH in contours_CH:  # convex hull
    hull = cv.convexHull(cnt_CH)
    cv.drawContours(img_color_CH, [hull], 0, (255, 0, 255), 5)

cv.imshow("result_CH", img_color_CH)

print(' ')

# Convexity Defects : 볼록 다각형 내부의 오목한 부분 찾기
# https://m.blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220524551089&proxyReferer=https:%2F%2Fwww.google.com%2F
print('Convexity Defects')
img_color_CD = cv.imread('hand.png')
img_gray_CD = cv.cvtColor(img_color_CD, cv.COLOR_BGR2GRAY)
ret_CD, img_binary_CD = cv.threshold(img_gray_CD, 127, 255, 0)
contours_CD, hierarchy_CD = cv.findContours(img_binary_CD, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

for cnt_CD in contours_CD:  # contour
    cv.drawContours(img_color_CD, [cnt_CD], 0, (255, 0, 0), 3)  # blue

for cnt_CD in contours_CD:  # convex hull
    hull_CD = cv.convexHull(cnt_CD)
    cv.drawContours(img_color_CD, [hull_CD], 0, (255, 0, 255), 5)  # purple

for cnt_CD in contours_CD:  # convexity defects
    hull_CD = cv.convexHull(cnt_CD, returnPoints = False)
    defects = cv.convexityDefects(cnt_CD, hull_CD)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt_CD[s][0])
        end = tuple(cnt_CD[e][0])
        far = tuple(cnt_CD[f][0])
        print('d : ', d)
        if d > 500:
            cv.line(img_color_CD, start, end, [0, 255, 0], 5)  # green
            cv.circle(img_color_CD, far, 5, [0,0,255], -1)  # red

cv.imshow("result_CD", img_color_CD)


cv.waitKey(0)
cv.destroyAllWindows()