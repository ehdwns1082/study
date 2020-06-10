# https://blog.naver.com/samsjang/220504966397
# 심화
# 마우스 이벤트 : https://webnautes.tistory.com/1219
import cv2
import numpy as np
import math

img = cv2.imread('moi.jpg')
h, w = img.shape[:2]
x1, x2, x3, x4, y1, y2, y3, y4, i = 0, 0, 0, 0, 0, 0, 0, 0, 0
list = [x1, y1, x2, y2, x3, y3, x4, y4]
w1, h1 = 0,0
print('h, w :', h, ',', w)
print('왼쪽 모서리 위부터 시계방향으로 찍어주세요')

def mouse_callback(event, x, y, flags, param):
    global i, w1, h1
    if event == cv2.EVENT_LBUTTONDOWN:
        list[2 * i], list[2 * i + 1] = x, y
        print('input(pts1)', i+1,'번쨰 : ', list)
        i = i + 1
        w1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        h1 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
cv2.imshow('original', img)
cv2.setMouseCallback('original', mouse_callback)

while True:
    x1 = list[0]
    y1 = list[1]
    x2 = list[2]
    y2 = list[3]
    x3 = list[4]
    y3 = list[5]
    x4 = list[6]
    y4 = list[7]

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    if i == 4:

        pts2 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        # 변환되기 전 4개 좌표와 변환된 후 4개 좌표를 통해 변환 매트릭스를 구한다.
        img2 = cv2.warpPerspective(img, M, (w, h))
        # cv2. warpPerspective(img, M, (cols, rows))
        # cv2.warpAffine()함수와 인자가 같지만, M 은 3x3 행렬이어야 한다.
        # img 의 모든 점 (x, y) 는 변환후 dst(x, y) 가 된다.
        # dst(x,y) = src( (M11x + M12y + M13)/(M31x + M32y + M33), (M21x + M22y + M23)/(M31x + M32y + M33) )

        img2 = img2[0:int(h1), 0:int(w1)]
        img2 = cv2.resize(img2, None, fx=w/w1, fy=h/h1, interpolation=cv2.INTER_AREA)
        cv2.imshow('perspective', img2)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
print(w1,h1)
print(img2.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()




