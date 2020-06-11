# https://blog.naver.com/samsjang/220504782549
# threshold 의 흔한 예로 반올림이 있다. 0.5라는 값을 기준으로 이보다 작으면 0으로, 크면 1로 한다.
'''
cv2.threshold(img, threshold_vlaue, value, flag)
1st : Grayscale 이미지
2nd : 픽셀 문턱값
3rd : 픽셀 문턱값보다 클 때 적용되는 최대값(적용되는 플래그에 따라 픽셀 문턱값보다 작을 때 적용되는 최대값)
4th : 문턱값 적용 방법 또는 스타일
    cv2.THRESH_BINARY: 픽셀 값이 threshold_value 보다 크면 value, 작으면 0으로 할당
    cv2.THRESH_BINARY_INV: 픽셀 값이 threshold_value 보다 크면 0, 작으면 value로 할당
    cv2.THRESH_TRUNC: 픽셀 값이 threshold_value 보다 크면 threshold_value, 작으면 픽셀 값 그대로 할당
    cv2.THRESH_TOERO: 픽셀 값이 threshold_value 보다 크면 픽셀 값 그대로, 작으면 0으로 할당
    cv2.THRESH_TOZERO_INV: 픽셀 값이 threshold_value 보다 크면 0, 작으면 픽셀 값 그대로 할당
'''

import cv2
import numpy as np

# Global Thresholding
img = cv2.imread('gray_gradation.jpg', cv2.IMREAD_GRAYSCALE)

ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thr2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thr3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thr4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thr5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow('original', img)
cv2.imshow('BINARY', thr1)
cv2.imshow('BINARY_INV', thr2)
cv2.imshow('TRUNIC', thr3)
cv2.imshow('TOZERO', thr4)
cv2.imshow('TOZERO_INV', thr5)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Adaptive Thresholding : 적절한 인자를 적용하여 노이즈를 최소화 하면서 정련된 이미지 추출
'''
cv2.adaptiveThreshold(img, value, adaptiveMethod, thresholdType, blocksize, C)
1st : Grayscale 이미지
2nd : adaptiveMethod에 의해 계산된 문턱값과 thresholdType에 의해 픽셀에 적용될 최대값
3rd : 사용할 Adaptive Thresholding 알고리즘
    cv2.ADAPTIVE_THRESH_MEAN_C: 적용할 픽셀 (x,y)를 중심으로 하는 blocksize x blocksize 안에 있는 픽셀값의 평균에서 C를 뺀 값을 문턱값으로 함
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 적용할 픽셀 (x,y)를 중심으로 하는 blocksize x blocksize안에 있는 Gaussian 윈도우 기반 가중치들의 합에서 C를 뺀 값을 문턱값으로 함
4th : 픽셀에 적용할 문턱값을 계산하기 위한 블럭 크기. 적용될 픽셀이 블럭의 중심이 됨. 따라서 blocksize는 홀수여야 함
5th : 보정 상수로, 이 값이 양수이면 계산된 adaptive 문턱값에서 빼고, 음수면 더해줌. 0이면 그대로..
'''
def thresholding():
    img =cv2.imread('print.jpg', cv2.IMREAD_GRAYSCALE)

    ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    thr2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thr3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11 ,2)

    titles = ['original', 'Global Thresholding(v=127)', 'Adaptive MEAN', 'Adaptive GAUSSIAN']
    images = [img, thr1, thr2, thr3]

    for i in range(4):
        cv2.imshow(titles[i], images[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

thresholding()

# Otsu's Binarization
'''
global thresholding 에서는 문턱값을 우리가 임의로 정했지만 적당한 값을 찾기가 어렵다. 
좋은 문턱값을 찾는 방법은 시행착오를 통해 최적값을 얻는 것이다. 
만약 이미지 히스토그램이 두개의 봉우리를 가지는 bimodal image 라고 하면 이 이미지에 대한 문턱값으로 둘 사이의 값을 취하면 최적의 결과를 얻을 수 있다.
Otsu Binarization 은 이미지 히스토그램 분석을 통해 중간값을 취하여 thresholding 한다.  
'''
import matplotlib.pyplot as plot # 그래프 그리는 라이브러리

def OtsusBinarization():
    img = cv2.imread('image_noise2.bmp', cv2.IMREAD_GRAYSCALE)

    # Global thresholding 사용
    ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Otsu Binarization
    ret, thr2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu Binarization 을 적용하는 함수는 따로 없고, cv2.threshold() 함수에 cv2.THRESH_OTSU 플래그 값을 thresholding 플래그에 더하고 문턱값으로 0을 전달해주면 된다.
    # 이렇게 하면 cv2.threshold() 함수는 적절한 문턱값을 계산한 후 이를 적용한 결과를 리턴한다.

    # GaussianBlur 적용 후 Otsu Binarization
    blur = cv2.GaussianBlur(img, (5,5), 0)
    ret, thr3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    titles = ['original noisy', 'Histogram', 'Global Thresholding',
              'Original noisy', 'Histogram', 'Otsu Thresholding',
              'Gaussian-filtered', 'Histogram', 'Otsu Thresholding']
    images = [img, 0, thr1, img, 0, thr2, blur, 0, thr3]

    for i in range(3):
        plot.subplot(3, 3, i*3+1), plot.imshow(images[i*3], 'gray')
        plot.title(titles[i*3]), plot.xticks([]), plot.yticks([])

        plot.subplot(3, 3, i*3+2), plot.hist(images[i*3].ravel(), 256)
        plot.title(titles[i*3]), plot.xticks([]), plot.yticks([])

        plot.subplot(3, 3, i*3+3), plot.imshow(images[i*3+2], 'gray')
        plot.title(titles[i*3+2]), plot.xticks([]), plot.yticks([])
    plot.show()

OtsusBinarization()


