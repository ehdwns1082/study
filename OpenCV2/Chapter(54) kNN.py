# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220673340574&categoryNo=66&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=1
# 난수 생성 : https://datascienceschool.net/view-notebook/8bf41f87a08b4c44b307799577736a28/
# ravel : https://rfriend.tistory.com/349

# k-Nearest Neighbours : 지도학습에 활용되는 가장 단순한 종류의 알고리즘
'''
1. 0~100 범위에서 2차원 좌표된 25개의 멤버를 랜덤하게 생성한다.
2. 각 멤버들을 랜덤하게 클래스0(빨간 삼각형), 클래스1(파랑 사각형)으로 구분한다.
3. 새로운 멤버(초록색 원)를 랜덤하게 생성하고 kNN 방법으로 k=3 일 때 이 멤버가 어느 클래스에 속하는지 알아본다.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def makeTraindata():
    traindata = np.random.randint(0,100,(25,2)).astype(np.float32)  # 0 ~ 100 사이의 숫자로 25x2 크기의 난수 매트릭스 생성. float32 로 타입 변경
    '''
    np.random.옵션()
    옵션1) rand : 0부터 1사이의 균인 분포 ex) np.random.rand(n) : (n,) 의 난수배열 생성 , np.random.rand(a,b) : a x b 매트릭스로 난수를 만듬 
    옵션2) randn : 가우시안 표준 정규 분포 ex) rand 와 사용법 같음
    옵션3) randint : 균일분포의 정수 난수 ex) np.random.randint(low, high=None, size=None) : high 를 입력하지 않으면 0 ~ low 사이의 숫자를, high 를 입력하면 low ~ high 숫자를 출력. size 는 난수배열의 크기(rand 와 형식 같음)
    '''
    resp = np.random.randint(0,2,(25,1)).astype(np.float32)
    return traindata, resp

def knn():
    traindata, resp = makeTraindata()

    red = traindata[resp.ravel()==0]  # ravel() 함수를 이용하여 (25,1) 배열 resp 를 (1,25) 배열로 만든후, 0 과 비교연산하여 Boolean 타입으로 만든다.
                                      # traindate[True] 일 경우 red 에 값이 저장되고, traindata[False] 일 경우 red 에 값이 저장되지 않는다.
                                      # 즉, red 라는 변수의 구성 인자의 수도 랜덤하게 세팅한 것.
    '''
    ravel out : 맺힌 것을 풀다.
    np.ravel(arr, order='C') : 다차원 배열을 1차원 배열로 만들어 줌
    1st : 대상 어레이                                                 #  x = np.arange(12).reshape(3,4)
    2nd : order='C' : C와 같은 순서로 인덱싱하여 평평하게 배열(디폴트) ex) np.ravel(x,order='C') => array([0,1,2,3,4,5,6,7,8,9,10,11])
          order='F' : Fortran 과 같은 순서로 인덱싱하여 평평하게 배열 ex) np.ravel(x,order='F') => array([0,4,8,1,5,9,2,6,10,3,7,11])
          order='K' : 메모리에서 발생하는 순서대로 인덱싱하여 평평하게 배열 ex) np.ravel(x,order='K') => array([0,1,2,3,4,5,6,7,8,9,10,11])
    '''

    blue = traindata[resp.ravel()==1]

    plt.scatter(red[:,0], red[:,1],80,'r','^' )  # matplotlib 이용하여 데이터를 시각자료로 표현
    plt.scatter(blue[:,0], blue[:,1], 80, 'b', 's')

    newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)  # 새로운 멤버를 랜덤하게 생성
    plt.scatter(newcomer[:,0], newcomer[:,1], 80, 'g', 'o')
    plt.show()




    

    knn = cv2.ml.KNearest_create()
    knn.train(traindata, cv2.ml.ROW_SAMPLE, resp)
    ret, result, neighbours, dist = knn.findNearest(newcomer, 3)

    print(result, neighbours)

    return

knn()

'''
[[0.]] : 초록색 원이 클래스 0(빨간 삼각형)에 속한다
[[1.]] : 초록색 원이 클래스 1(파란 사각형)에 속한다 
[[0.1.1.]] : 초록색 원 주위에 가까운 이웃멤버는 클래스 0인 멤버가 1개, 클래스 1인 멤버가 2개 있다는 의미
'''