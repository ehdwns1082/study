# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220565430664&parentCategoryNo=&categoryNo=66&viewDate=&isShowPopularPosts=false&from=postList

# 푸리에 변환
'''
푸리에 변환의 핵심 : 주기를 가진 함수는 주파수가 다른 삼각함수의 합으로 표현 가능하다.
                   주기는 무한대까지 가능하므로 '모든 함수는 주파수가 다른 삼각함수의 합으로 표현 가능하다.' 라는 말이 된다.
삼각함수 ex) x(t) = Asin(2pift)

주파수 개념을 이미지에 적용한다면?
픽셀값의 변화가 얼마나 빨리 진행되냐를 주파수로 생각한다.

이미지는 보통 2D 이므로 픽셀값의 변화도 x축과 y축 모두 고려해야 한다.
이미지에서 픽셀값의 변화가 큰 경우는 이미지의 경계부근이나 이미지와 관련없는 노이즈 부분에서 발생한다.
2D 이산 푸리에 변환(2D Discrete Fourier Transform:2D-DFT)을 이미지에 적용하면 이미지를 주파수 영역으로 변환해 준다.
DFT 를 계산하기 위해서는 고속 푸리에 변환(Fast Fourier Transform:FFT)을 이용한다.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

#Numpy를 활용한 푸리에 변환
def fourier_np():
    img = cv2.imread('musician.png', cv2.IMREAD_GRAYSCALE)

    f = np.fft.fft2(img)
    # 이미지의 2D DFT 를 계산

    fshift = np.fft.fftshift(f)
    '''
    np.fft.fft2(img) 를 수행하여 얻어진 푸리에 변환 결과는 주파수가 0인 컴포넌트를 좌상단에 위치시킨다.
    np.fft.fftshift(f) 는 주파수가 0인 부분을 정중앙에 위치시키고 재배열 해주는 함수이다.
    
    ex)
    >>> f = np.fft.fftfreq(10, 0.1)
    >>> f 
    [0. 1. 2. 3. 4. -5. -4. -3. -2. -1.]
    >>> fshift = np.fft.fftshift(f)
    >>> fshift
    [-5. -4. -3. -2. -1. 0. 1. 2. 3. 4.]
    원래 주파수 배열에서 왼쪽에 있던 0의 위치가 정중앙으로 바뀌었다.
    '''

    m_spectrum = 20*np.log(np.abs(fshift))
    # magnitude spectrum 을 구한다.

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(m_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.show()
fourier_np()


# OpenCV 를 활용한 푸리에 변환
def fourier_cv():
    img = cv2.imread('musician.png', cv2.IMREAD_GRAYSCALE)

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # cv2.dft() 함수는 이미지의 2d DFT 을 계산한다.
    # 이미지를 이함수의 인자로 입력할 때는 반드시 np.float32로 랩핑해야 한다.
    # 이 함수는 복소수 형태로 결과를 리턴하는 것이 numpy 를 이용할 떄와의 차이점이다.

    dft_shift = np.fft.fftshift(dft)

    m_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    # cv2.magnitude(x,y)는 2차원 벡터의 크기를 계산한다. 결과는 numpy 를 이용한 것과 동일.

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(m_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.show()
fourier_cv()