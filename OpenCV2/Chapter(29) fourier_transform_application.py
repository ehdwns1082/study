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
    # 이미지의 2D DFT 를 계산
    # 2D 이산 푸리에 변환을 이미지에 적용하면 이미지를 주파수 영역으로 변환한다.
    # np.fft.fft2() 를 수행하여 얻어진 푸리에 변환 결과는 주파수가 0인 컴포넌트(픽셀값의 변화가 없는 부분)를 좌상단에 위치시킨다.

    fshift = np.fft.fftshift(f)
    # 주파수가 0인 부분을 정중앙에 위치시키고 재배열 해준다.

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)

    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    # 주파수 영역의 이미지 정중앙의 60x60 크기 영역에 있는 값을 모두 0으로 만든다.

    f_ishift = np.fft.ifftshift(fshift)
    # 역쉬프트 함수 np.fft.ifftshift() 를 이용해 재배열된 주파수 값들의 위치를 되돌린다.

    img_back = np.fft.ifft2(f_ishift)
    # np.fft.ifft2D() 함수를 이용해 역푸리에 변환을 하여 원래 이미지 영역으로 전환한다.

    img_back = np.abs(img_back)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('After HPT'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

    plt.show()

fourier()

'''
이미지 필터링에 사용되는 커널 종류
- Averaging Filter : array[ [1,1,1], [1,1,1], [1,1,1] ]  (3x3 기준)
- Gaussian Filter : 
- Scharr Filter : array[ [-3,0,3], [-10,0,10], [-3,0,3] ]
- sobel_x Filter : array[ [-1,0,1], [-2,0,2], [-1,0,1] ] 
- soble_y Filter : array[ [-1,-2,-1], [0,0,0], [1,2,1] ]
- Laplacian Filter : array[ [0,1,0], [1,-4,1], [0,1,0] ]

'''

def check_kernel():

    mean_filter = np.ones((3,3))
    # simple averaging filter without scaling parameter

    x = cv2.getGaussianKernel(3,3)
    gaussian = x*x.T
    # creating a gaussian filter

    '''
    밑에부터 edge 검출 필터들
    '''
    scharr = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])

    laplacian = np.array([ [0,1,0], [1,-4,1], [0,1,0] ])

    sobel_x = np.array([ [-1,0,1], [-2,0,2], [-1,0,1] ])

    sobel_y = np.array([ [-1,-2,-1], [0,0,0], [1,2,1] ])



    filters = [mean_filter, gaussian, scharr, laplacian, sobel_x, sobel_y]
    filter_name = ['mean_filter', 'gaussian', 'scharr', 'laplacian', 'sobel_x', 'sobel_y']

    fft_filters = [np.fft.fft2(x) for x in filters]
    fft_shift = [np.fft.fftshift(y) for y in fft_filters]
    mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

    for i in range(6):
        plt.subplot(2,3,i+1), plt.imshow(mag_spectrum[i], cmap = 'gray')
        plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

    plt.show()

check_kernel()
'''
결과 그림에서 흰색 부분이 통과되는 주파수 영역이고, 검정색 부분은 필터링 되는 주파수 영역이다.
mean 필터와 gaussian 필터는 중앙 부분인 주파수가 낮은 영역만 통과시키는 LPF 이다.

laplacian 필터는 mean 필터와 정반대인 것을 알 수 있는데, 주파수가 높은 영역만 통과시키는 HPF 이다.

soble_x, sobel_y, scharr_x 의 주파수 영역 이미지를 보면 어떤 부분이 필터링 되고 패스되는지 알 수 있다.

'''