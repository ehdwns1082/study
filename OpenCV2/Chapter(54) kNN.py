# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220673340574&categoryNo=66&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=1
# 난수 생성 : https://datascienceschool.net/view-notebook/8bf41f87a08b4c44b307799577736a28/

'''
1. 0~100 범위에서 2차원 좌표된 25개의 멤버를 랜덤하게 생성한다.
2. 각 멤버들을 랜덤하게 클래스0(빨간 삼각형), 클래스1(파랑 사각형)으로 구분한다.
3. 새로운 멤버(초록색 원)를 랜덤하게 생성하고 kNN 방법으로 k=3 일 때 이 멤버가 어느 클래스에 속하는지 알아본다.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def makeTraindata():
    traindata = np.random.randint(0,100,(25,2)).astype(np.float32)
    '''
    np.random.옵션()
    옵션1) rand : 0부터 1사이의 균인 분포 ex) np.random.rand(n) : 난수를 n개 생성함
    옵션2) randn : 가우시안 표준 정규 분포
    옵션3) randint : 균일분포의 정수 난수 
    '''
    resp = np.random.randint(0,2,(25,1)).astype(np.float32)
    return traindata, resp

def knn():
    traindata, resp = makeTraindata()

    red = traindata[resp.ravel()==0]
    blue = traindata[resp.ravel()==1]
    plt.scatter(red[:,0], red[:,1],80,'r','^' )
    plt.scatter(blue[:,0], blue[:,1], 80, 'b', 's')

    newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
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