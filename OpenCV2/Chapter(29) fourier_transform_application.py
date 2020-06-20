# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220568857153&parentCategoryNo=&categoryNo=66&viewDate=&isShowPopularPosts=false&from=postList

'''
푸리에 변환은 이미지를 주파수 영역으로 전환하여 이미지 프로세싱 작업을 수행할 수 있게 해는 도구이고,
주파수 영역에서 작업이 끝나면 역푸리에 변환(Inversion Fourier Transform:IFT)을 수행하여 원래 이미지 영역으로 되돌려서 이미지 프로세싱 결과를 확인 할 수 있다.

LPF(Low Pass Filter) 는 낮은 주파수 대력만 통과시키는 필터이고, HPF(High Pass Filter) 는 높은 주파수 대역만 통과시키는 필터이다.
이미지에서 LPF 를 사용하면 낮은 주파수 대역만 남아있는 이미지가 되므로 blur 효과를 가진 이미지가 된다.
이미지에서 HPF 를 사용하면 높은 주파수 대역만 남아있는 이미지가 되므로 사물의 edge 나 노이즈 등만 남아 있는 이미지가 된다.

푸리에 변환을 통해 주파수 영역으로 옮긴 이미지로 주파수 작업을 수행하면 보다 다양한 필터링 작업을 수행할 수 있다.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

def fourier():
    img = cv2.imread('musician.png', cv2.IMREAD_GRAYSCALE)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)

    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('After HPT'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

    plt.show()

fourier()

