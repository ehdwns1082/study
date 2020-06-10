# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220548160080&parentCategoryNo=&categoryNo=66&viewDate=&isShowPopularPosts=false&from=postList
# 이미지 히스토그램 배경투사
'''
먼저 이미지에서 우리가 원하는 객체에 해당하는 픽셀과 확률적으로 비슷한 픽셀들을 구한다.
얻어진 픽셀 부분을 다른 부분에 비해 좀 더 흰색에 가깝도록 만든 후, 원본 이미지와 동일한 크기의 새로운 이미지를 생성한다.
새롭게 생성한 이미지를 적절하게 thresholding 처리를 거치고, 처리된 이미지와 원본 이미지를 이용해 적절한 연산을 수행하면 배경만 추출되던가 혹은 배경을 제외한 부분을 추출할 수 있다.
1. 먼저 우리가 찾고자 하는 객체의 컬러 히스토그램(M)과 원본 이미지의 컬러 히스토그램(I)를 계산한다.
2. R=M/I 를 구한다. 이렇게 구한 R을 팔레트 값으로 하고 원본 이미지에서 우리가 원하는 대상과 확률적으로 일치하는 모든 픽셀을 이용해 새로운 이미지를 생성한다.
   해당하는 픽셀 (x,y) 는 B(x,y) = R[h(x,y),s(x,y)] 로 구한다. 여기서 h(x,y)는 픽셀 (x,y)에서 hue 이고, s(x,y)는 saturation 이다.
   이렇게 구한 픽셀집합 B(x,y)의 값을 1과 비교해서 작은 값을 취한다. B(x,y), = min(B(x,y),1)
3. 원형 convolution 을 적용한다. B = D*B (D는 원형 커널)
4. 3번 과정까지 처리하게 되면 픽셀값이 가장 밝은 부분이 우리가 원하는 대상이다. 적절한 값으로 thresholding 하여 흰색으로 만든다.
5. 4번 과정의 결과와 원본 이미지를 비트연산 하면 우리가 원하는 대상 또는 원하지 않는 대상만을 추출 할 수 있다.
'''
import numpy as np
import cv2

# 초깃값 세팅
ix, iy = -1, -1
mode = False
img1, img2 = None, None

def onMouse(event, x, y, flag, param):
# 원본 이미지에서 마우스로 사각형 영역을 지정하면 해당 영역과 비슷한 부분을 추출하여 화면에 보여준다.
    global ix, iy, mode, img1, img2  # 전역 변수(함수 밖에서도 쓰임)

    if event == cv2.EVENT_LBUTTONDOWN:
        mode = True  # 마우스 좌클 누른 시점에 mode 를 True 로 세팅한다.
        ix, iy = x, y  # ix, iy 에 마우스 좌클누른 시점의 좌표를 저장한다.
    elif event == cv2.EVENT_MOUSEMOVE:
        if mode:  # 마우스 커서 움직이는 동안에 모드가 True 라면 <=> 좌클한 후에 드래그 하다면
            img1 = img2.copy()  # img1 은 im2의 복사본이다.
            cv2.rectangle(img1, (ix,iy), (x,y), (0,0,255), 2)  # img1 에 ix,iy 에서 시작해서 x,y 로 끝나는 색상 (0,0,255), 굵기 2의 직사각형을 그린다. (과정 보여주기)
            cv2.imshow('original', img1)
    elif event == cv2.EVENT_LBUTTONUP:
        mode = False  # 마우스 좌클을 뗀 시점에서 mode 를 False 로 변경한다. <=> 이제부터는 직사각형을 그리지 않는다.
        if ix >= x or iy >= y:  # 좌상단에서 우하단으로 드래그 하지 않으면 함수 종류, 이거 없으면 오류남
            return
        cv2.rectangle(img1, (ix,iy), (x,y), (0,0,255), 2)   # img1 에 ix,iy 에서 시작해서 x,y 로 끝나는 색상 (0,0,255), 굵기 2의 직사각형을 그린다. (확정)
        roi = img1[iy:y, ix:x]  # img1 에서 직사각형 박스 안에 있는 부분을 roi 로 잡는다.
        backProjection(img2, roi)  # 밑에서 정의한 backProjection 함수로 img2 와 roi 를 인풋으로 넣는다.
    return  # 함수 종료 => 마우스로 드래그해서 roi 만들어주면 역할 끝!


def backProjection(img, roi):
# 마우스로 지정한 대상과 원본 이미지를 가지고 히스토그램 배경투사 알고리즘을 구현한 부분
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # roi 를 BGR 에서 HSV 로 바꾼후에 변수 hsv 에 저장한다
    hsvt = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 위에서 보낸 img1을 이 함수에서 img 로 받아서 BGR2HSV 한 후 변수 hsvt 에 저장한다.

    roihist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])  # roi 의 히스토그램을 그린다.
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)  # roi 의 히스토그램을 normalize 한다.
    '''
    cv2.normalize() : 인자로 입력된 numpy 배열을 정규화하는 함수이다.
    1st : 오리지널 배열
    2nd : 결과로 나올 배열
    3rd : 결과 numpy 배열의 최소값
    4th : 결과 numpy 배열의 최대값
    5th : cv2.NORM_MINMAX
        >>>  x = [ [0,1,2], [3,4,5] ]
        >>>  cv2.normalize(x,x,0,255,NORM_MINMAX)
        >>>  x = [ [0,51,102], [153,204,255] ]
        => 입력된 x의 최소값은 0 이고, 최대값은 5이다. 따라서 0->0 으로 5->255 로 조정한 후 나머지 1,2,3,4 의 값을 비율에 맞게 조정한다.
    '''

    dst = cv2.calcBackProject([hsvt], [0,1], roihist, [0,180,0,256], 1)  # roihist 에 속한 픽셀들이 hsvt(원본이미지)에 속할 확률을 계산, 확률이 높을 수록 하얀색으로 표현한 이미지
    '''
    cv2.calcBackProject() : 입력이미지와 같은 크기이지만 하나의 채널만 가지는 이미지를 생성한다. 
                            이 이미지의 픽셀은 특정 오브젝트에 속할 확률을 의미한다. 관심 오브젝트 영역에 속한 픽셀이 나머지 부분보다 더 흰색으로 표현된다.
    1st : 원본이미지, 인자는 반드시 []로 둘러쌓여야 함
    2nd : 채널 수, HSV 에서는 색상과 채도만 알면 되므로 0,1 
    3rd : 대상 부분의 컬러 히스토그램 
    4th : 픽셀값 범위
    5th : scale, 원본과 비율이 같을 떄는 1
    '''

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))  # 모폴로지 연산 수행할 커널 생성 => 노이즈 제거 위해서
    '''
    cv2.GetStructuringElement(MorphShapes shape, kernel size) : 커널 매트릭스 생성
    1st : MorphShapes 
          M1) Rect : 생성한 ksize 크기에 접하는 사각형
          M2) Cross : 생성한 ksize 크기에 접하는 십자형
          M3) Ellipse : 생성한 ksize 크기에 접하는 원형
    2nd : 커널 사이즈 
    
    cf) cv2.Erode() : 침식함수, 바이너리 이미지에서 흰색 오브젝트 외각을 검은색으로 변경
        => 이진화 이미지에서 작은 흰색노이즈를 제거하거나, 합쳐진 오브젝트를 분리하는데 사용가능
        cv2.Dilate() : 팽창함수, 바이너리 이미지에서 흰색 오브젝트 주변에 흰색을 추가
        => 이진화 이미지에서 침식으로 줄어든 오브젝트를 원복하거나, 인접한 오브젝트를 하나로 만드는데 사용가능
    '''

    cv2.filter2D(dst, -1, disc, dst)  # 위에서 확률 계산한 dst 를 2d convolution 필터링을 통해 노이즈 제거하는 과정
    '''
    cv2.filter2D(src, ddepth, kelnel, dst) 
    1st : input
    2nd : 이미지 깊이(자료형 크기), -1이면 입력과 동일
    3rd : 커널 매트릭스
    4th : output 
    
    cf) src : 원본이미지, dst : 출력이미지
    '''

    ret, thr = cv2.threshold(dst, 50, 255, 0)  # dst 에서 확실히 구분하기 위해 이진화 진행
    thr = cv2.merge((thr,thr,thr)) # 이미지 채널 합치기 <=> cv2.split  cf) https://webnautes.tistory.com/1241
    res = cv2.bitwise_and(img, thr)  # 원본이미지와 and 비트연산을 통해 마스크처리
    cv2.imshow('backproj', res)

def main():
    global img1, img2

    img1 = cv2.imread('musician.png')
    img2 = img1.copy()  # img2 는 img1 의 복사본이다.

    cv2.namedWindow('original'), cv2.namedWindow('backporj')
    cv2.setMouseCallback('original', onMouse, param=None)

    cv2.imshow('backporj', img2)

    while True:
        cv2.imshow('original', img1)

        k = cv2.waitKey(1)&0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

main()