# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220537529081&parentCategoryNo=&categoryNo=66&viewDate=&isShowPopularPosts=false&from=postView

# 이미지 히스토그램은 가로축으로 픽셀값을, 세로축으로 이미지 픽셀수를 나타낸 좌표에 이미지 특성을 표시한 것인다.

import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram():
    img1 = cv2.imread('image_landscape.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('image_landscape.jpg')

    # OpenCV 함수를 이용해 히스토그램 구하기
    hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
    '''
    cv2.calcHist(img, channel, mask, histSize, range)
        이 함수는 이미지 히스토그램을 찾아서 numpy 배열로 리턴함
        1st : 히스토그램을 찾을 이미지. 인자는 반드시 []로 둘러 싸야 함.
        2nd : grayscale 이미지의 경우 [0]을 인자로 입력하며, 
              컬러 이미지의 경우 B,G,R 에 대한 히스토그램을 위해 각각 [0],[1],[2] 을 인자로 입력
        3rd : 이미지 전체에 대한 히스토그램을 구할 경우 None, 이미지의 특정 영역에 대한 히스토그램을 구할 경우 이 영역에 해당하는 mask 값을 입력 
        4th : BIN 개수. 인자는 []로 둘러싸야 함.
        5th : 픽셀값 범위. 보통 [0,256]    
    '''


    # numpy 를 이용해 히스토그램 구하기
    hist2, bins = np.histogram(img1.ravel(), 256, [0,256])
    # 1-D 히스토그램의 경우(1채널) numpy 가 빠름
    hist3 = np.bincount(img1.ravel(), minlength=256)
    '''
        numpy.histogram() 이 함수는 이미지에서 구한 히스토그램과 BIN 의 개수를 리턴한다.
        grayscale 의 경우 numpy.bincount() 함수를 이용하는게 numpy.histogram() 함수에 비해 10배 더 빠르다.
            
        But, 히스토그램을 구하기 위해 가장 성능이 좋은 함수는 cv2.calcHist() 함수이다! 
        
        cf) numpy.ravel() 함수는 numpy 배열을 1차원으로 바꿔주는 함수이다.
            >>>  import numpy as np
            >>>  x = np.array([0,1,2,3], [4,5,6,7])
            >>>  np.ravel(x)
            [0 1 2 3 4 5 6 7]
    '''

    # matplotlib 으로 히스토그램 그리기
    plt.hist(img1.ravel(), 256, [0,256])
    '''
    matplotlib 의 pyplot.hist() 함수를 이용해서 히스토그램을 구하지 않고 바로 그릴 수 있다.
    2번 쨰 인자는 BIN 의 개수이다. 
    '''

    # 컬러 이미지 히스토그램 그리기
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img2], [i], None, [256], [0,256])
        plt.plot(hist, color=col)
        plt.xlim([0,256])
    '''
    plt.hist() 함수는 grayscale 이미지에 대한 히스토그램을 그려주는 함수였지만, 
    컬러 이미지의 경우는 cv2.calcHist() 함수를 이용해 B,G,R 에 대한 히스토그램을 각각 구하고, 이를 plt.plot() 함수를 이용해 화면에 그려준다.
    plt.xilm() 함수를 이용해 가로축을 0~256 까지로 제한했다.
    
    '''
    cv2.imshow('img', img2)
    plt.show()

histogram()


