# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220505080672&categoryNo=66&parentCategoryNo=0&viewDate=&currentPage=5&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=5
# http://www.gisdeveloper.co.kr/?p=6520
# bilateral filter : http://egloos.zum.com/eyes33/v/6092269
# 커널 : https://blog.naver.com/roboholic84/220962562239
import numpy as np
import cv2

# 2D Convolution (Image Filtering)
'''
LPF(low-pass-filter) 는 이미지 '노이즈를 제거'하거나 이미지를 blur 처리하기 위해 사용된다.
HPF(high-pass-filter) 는 이미지에서 'edge 를 찾는데' 활용된다.
OpenCV 는 필터 커널을 이미지에 convolve 하여 적용하는 cv2.filter2D() 함수를 제공한다.
이미지의 픽셀 값을 해당 픽셀의 이웃과 평균하여 그 값을 취하도록 할 수 있는데, 이를 Averaging filter 라고 한다.
예를 들어 모든 원소가 1인 5x5 averaging filter 커널은 1/25*(5x5 matrix) 이다.
1. 픽셀을 중심으로 5x5 영역을 만듬
2. 이 영역의 모든 픽셀 값을 더함
3. 더한 값을 25로 나누고 이 값을 중심 픽셀 값으로 취함
즉, 이 커널이 적용된 averaging filter 는 5x5 영역 내의 모든 픽셀 값을의 평균값을 취한다.
'''
def average_filter():
    img = cv2.imread('musician.png')

    kernel = np.ones((5,5), np.float32)/25
    blur = cv2.filter2D(img, -1, kernel)
    # cv2.filter2D(imageNDArray, -1, kernelNDArray)

    cv2.imshow('original', img)
    cv2.imshow('blur', blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
average_filter()


# 이미지 블러링 (Image Bluring)
'''
이미지 블러링은 LPF 커널을 이미지에 적용하여 달성되며 노이즈 제거에 유용하다.
LPF 는 이미지의 노이즈나 모서리 등과 같은 고주파 부분을 제거할으로써 edge 가 무뎌지게 되고 결과적으로 이미지 블러링 효과가 도출된다.
OpenCV 에서는 4가지 블러링 기술을 제공한다.
1. Averaging
2. Gaussian Filtering : 가우스 노이즈를 제거하는데 효과적 (모서리를 무디게)
3. Median filtering : 소금-후추 뿌린 듯한 노이즈 제거에 효과적
4. Bilateral Filtering : edge 를 보존하고 표면의 질감 등을 제거해주는데 효과적 
   Gaussian 필터와 비교해 보면, Gaussian 은 하나의 픽셀을 기준으로 이 픽셀 주위에 있는 픽셀들의 값들에 의존적으로 계산을 수행하며,
   필터링 동안에 픽셀이 타겟 픽셀과 동일한 값을 가지고 있는건지, 픽셀이 모서리에 존재하는지 안하는지 이런부분은 체크를 하지 않는다.
   이런 이유로 Gaussian 필터를 적용하여 이미지 처리를 하면, edge 가 보존되지 않고 뭉개져 버린다.
   bilateral 필터는 Gaussian 필터를 이용하지만 픽셀의 intensity 차이를 고려한 또 하나의 Gaussian 필터를 적용한다.
   기존 Gaussian 필터는 필터링을 위해 공간적으로 이웃한 픽셀들만 확인하고 처리하는데 반해, 
   또 하나의 Gaussian 필터는 해당 필터와 비스무리한 intensity 를 가진 픽셀까지 고려해서 필터링을 수행하기 때문에 edge 등이 보존 될 수 있다.
   
'''

def onMouse(x):
    pass

def bluring():
    img = cv2.imread('musician.png')

    cv2.namedWindow('BlurPane')
    cv2.createTrackbar('BLUR_MODE', 'BlurPane', 0, 2, onMouse)
    cv2.createTrackbar('BLUR', 'BlurPane', 0, 5, onMouse)

    mode = cv2.getTrackbarPos('BLUR_MODE', 'BlurPane')
    val = cv2.getTrackbarPos('BLUR', 'BlurPane')

    while True:
        val = val *2 + 1

        try:
            if mode == 0:
                blur = cv2.blur(img, (val,val))  # LPF, 노이즈 제거
                # cv2.blur(img, (val,val))
                # 1st : 블러링 필터를 적용할 원본 이미지
                # 2nd : 필터 커널 사이즈, 두 값이 달라도 무관함
            elif mode == 1:
                blur = cv2.GaussianBlur(img, (val,val), 0)
                # cv2.GaussianBlur()  # 모서리를 무디게
                # 1st : 필터 적용할 원본 이미지
                # 2nd : Gaussian 블러 필터. (val1,val2)와 같이 두 값이 달라도 되지만, 모두 양의 홀수 이어야함.
                # 3rd : sigmaX 값 =0 으로 sigmaY 값은 자동적으로 0으로 설정되고 Gaussian 블러 필터만을 적용함
            elif mode == 2:
                blur = cv2.medianBlur(img, val)  # 소금-후추 노이즈 제거
                # cv2.medianBlur(img, val)
                # 1st : 필터 적용할 원본 이미지
                # 2nd : 커널 사이즈. (val x val) 크기의 박스내에 있는 모든 픽셀들의 median 값을 취해서 중앙에 있는 픽셀에 적용함.
            else:
                break

            cv2.imshow('BlurPane', blur)
        except:
            break

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        mode = cv2.getTrackbarPos('BLUR_MODE', 'BlurPane')
        val = cv2.getTrackbarPos('BLUR', 'BlurPane')

    cv2.destroyAllWindows()

bluring()

def bilateral():
    img = cv2.imread('musician.png')
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    '''
    cv2.bilateralFilter(scr, dst, d, sigmaColor, sigmaSpace)
    1st : 입력 이미지
    2nd : 출력 이미지
    3rd : 필터링을 수행할 지름. 정의 불가능한 경우 sigmaSpace 사용
    4th : 컬러공간의 시그마공간 정의, 클수록 이웃한 픽셀과 기준색상의 영향이 커진다.
    5th : 시그마 필터를 조정한다. 값이 클수록 긴밀하게 주변 픽셀에 영향을 미친다.
          d>0 이면 영향을 받지 않고, 그 외에는 d 값에 비례한다.
    '''
    cv2.imshow('original', img)
    cv2.imshow('bilateral_blur', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
bilateral()