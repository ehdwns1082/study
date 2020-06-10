# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220506567638&categoryNo=66&parentCategoryNo=0&viewDate=&currentPage=5&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=5

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gradient 를  이용한 edge 검출
'''
OpenCV 는 Sobel, Scharr, Laplacian 이 세가지 타입의 Gradient 필터(HPF)를 제공한다.

* Sobel 미분 : Gauss Smoothing 과 미분연산을 결합한 형태의 연산을 수행
=> 노이즈에 보다 강력한 저항성을 제공
=> 세로방향 또는 가로방향으로 연산 수행이 가능   
cv2.Sobel(src, ddepth, dx, dy, ksize)
    1st : Sobel 미분을 적용할 원본 이미지
    2nd : 결과 이미지 데이터 타입
          CV_8U : 이미지 픽셀값을 uint8 로 설정
          CV_16U : 이미지 픽셀값을 uint16 으로 설정
          CV_32F : 이미지 픽셀값을 float32로 설정
          CV_64F : 이미지 픽셀값을 float64로 설정
    3rd : x 방향 미분차수 ex) 1 이면 x 방향으로 1차미분 수행
    4th : y 방향 미분차수 ex) 0 이면 y 방향 그대로 
    5th : 확장 Sobel 커널의 크기. 1,3,5,7 중 하나의 값으로 설정, -1로 설정되면 3x3 Sobel 필터 대신 3x3 Scharr 필터를 적용 

* Laplacian 미분 : 아래의 편미분식으로 이미지의 라플라시안을 계산
=> delta src = (d^2/dx^2)src + (d^2/dy^2)src
=> 각 편미분은 Sobel 알고리즘을 적용하며, ksize = -1인 경우 아래의 커널 매트릭스가 활용됨.
=> kernel(3x3) = [ [0, 1, 0], [1, -4, 1], [0, 1, 0] ]
'''

def grad():
    img = cv2.imread('keyboard.png', cv2.IMREAD_GRAYSCALE)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    plt.subplot(2,2,1), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,2,2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,2,3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,2,4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()

grad()
'''
함수 실행 결과를 보면 Laplacian 은 x방향 편미분 결과와 y방향 편미분 결과를 더한 것이 결과가 된다. 
따라서 원본 이미지의 가로선과 세로선이 모두 나타나는 것을 볼 수 있다.

Sobel X 의 결과는 세로선이 두드러지게, Sobel Y 의 결과는 가로선이 두드러지게 나타남을 알 수 있다.

미분이란 어느 한 지점에서의 변화율을 구하는 것이다. 
x방향으로 미분한다는 것은 x방향을 따라 값의 변화가 있다면 변화율이 감지된다는 것이고, 곧 미분값이 존재한다는 것이다.
예를 들어 원본 이미지를 따라가다가 경계선이 나타나면 픽셀의 값이 변화되는 것이고, Sobel 알고리즘은 이 부분을 캐치하게 된다.
즉, Laplacian 과 Sobel 은 이미지에서 경계를 찾는데 유용하게 활용된다. 

주의할 점으로는 cv2.Sobel() 함수를 사용할 때, 인자로 입력되는 데이터타입은 결과물에 영향을 미치는 요소이므로 주의해야 한다.
Sobel 알고리즘은 이미지가 검정색에서 흰색으로 변화될 때 양수 값을 취하고, 흰색에서 검정색으로 변화될 떄 음수 값을 취한다.

만약 데이터 타입을 양수만 취급하는 np.uint8 또는 cv2.8U로 지정하면 흰색에서 검정색으로 변화될 때 취한 음수값을 모두 0으로 만들어 버린다.
다시 말해서, 흰색에서 검정색으로의 경계를 찾지 못하는 결과를 가져온다. 

'''

def test():
    img = cv2.imread('binary_square.bmp', cv2.IMREAD_GRAYSCALE)

    sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)

    tmp = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel64f = np.absolute(tmp)
    sobelx8u2 = np.uint8(sobel64f)

    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 2), plt.imshow(sobelx8u, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 3, 3), plt.imshow(sobelx8u2, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])

    plt.show()

test()