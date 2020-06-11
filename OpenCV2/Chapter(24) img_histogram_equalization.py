# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220542334453&parentCategoryNo=&categoryNo=66&viewDate=&isShowPopularPosts=false&from=postView
# Histogram equalization : https://en.wikipedia.org/wiki/Histogram_equalization

# 픽셀값이 특정 밝기에 집중적으로 분포하고 있으면 좋은 이미지가 아니다.
# 이러한 이미지를 어두은 픽셀부터 밝은 픽셀까지 골고루 분포하게 변경하면 더 나은 이미지가 된다.
# 얼굴인식 등을 할 떄에 동일한 조명조건을 만들어야하는데, 이 경우에도 쓸 수 있다.


import numpy as np
import cv2
import matplotlib.pyplot as plt



img = cv2.imread('landscape_noise.bmp', cv2.IMREAD_GRAYSCALE)

def histogram(img):
    cv2.imshow('picture for figure1', img)
    hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist1, color='r')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogram_equalization(img):
    global img2
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    cdf = hist.cumsum()
    # numpy.cumsum() 함수는 numpy 배열을 1차원 배열로 변경한 후,
    # 각 멤버값을 누적하여 더한 값을 멤버로 하는 numpy 1차원 배열을 생성한다.

    cdf_m = np.ma.masked_equal(cdf, 0)
    # numpy 1차원 배열인 cdf 에서 값이 0 인 부분은 모두 mask 처리하는 함수이다.
    # 만약 numpy 1차원 배열 a 가 [1,0,0,2] 라면, np.ma.masked_equal(a,0) 의 값은 [1,--,--,2] 가 된다.

    cdf_m = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # 이미지 히스토그램 균일화 방정식을 코드로 나타낸 것

    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # numpy 1차원 배열인 cdf_m 에서 mask 처리된 부분을 0으로 채운 후 numpy 1차원 배열로 리턴한다.

    img2 = cdf[img]

    cv2.imshow('original', img)
    cv2.imshow('Histogram Equalization using numpy', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def equalization_OpenCV(img):
    global equ
    equ = cv2.equalizeHist(img)
    cv2.imshow('equalizer_OpenCV', equ)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def stack(file1, file2, file3):
    file1 = cv2.resize(file1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    file2 = cv2.resize(file2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    file3 = cv2.resize(file3, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    f = np.hstack((file1, file2, file3))

    cv2.imshow('stack : original, usingNumpy, usingOpenCV ', f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


histogram(img)
histogram_equalization(img)
histogram(img2)
equalization_OpenCV(img)
histogram(equ)
stack(img, img2, equ)

