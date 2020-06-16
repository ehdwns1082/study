'''
*  구성
1. 초기데이터 학습시키기
2. 실제 손글씨 인식하기
3. 인식실패시 재학습시키기

1. 초기 데이터 학습시키기
   digitimg.png 는 2000x1000 사이즈이며, 각 숫자는 가로로 100개 세로로 5개씩이다.

   1) 이미지에서 20x20 크기의 셀로 잘라서 총 5000개를 구성한다.
   2) 20x20 크기의 셀에 있는 픽셀값을 1차원으로 배열한다. 즉 1x400 배열이 됨.
   3) 각 숫자별로 400개의 배열이 있다고 하면, 총 배열(traindata)은 (5000,400) 크기가 됨.
   4) 0 에 해당하는 구역은 traindata[:500, :], 1 에 해당하는 구역은 traindata[500:1000, :], ~
      traindata 에 이런 숫자표시를 할 수 없으므로 (5000,1) 크기를 가진 배열에다 순서대로 0을 500개, 1을 500개, ~ 식으로 구성하면 된다.
   5) 학습한 데이터를 재사용하기 위해 4까지 학습한 내용을 파일로 저장한다.

2. 실제 손글씨 인식하기

   1) 학습한 내용이 저장된 파일을 읽는다.
   2) 손글씨로 적은 숫자 이미지를 인자로 받고, 이를 20x20 픽셀 크기로 변환한다.
   3) KNN 을 이용해 학습한 내용을 바탕으로 20x20 픽셀 크기로 반환한 이미지를 인식한다.

3. 재학습 시키기

   1) 실제 손글씨 숫자와 인식한 숫자가 다르면 이 손글씨에 대해 제대로 학습시킨다.
   2) 학습시킨 결과를 초기 데이터에 추가시킨다.
   3) 저장한 학습 데이터를 갱신하여 저장한다.

'''

import cv2
import numpy as np

# 인자로 입력된 손글씨 숫자 이미지 파일을 읽어 20x20 픽셀로 변환한 후 인식을 위해 (1,400) 크기의 numpy 배열로 리턴한다.
def resize20(digitimg):  # 실제 손글씨 이미지를 인자로 받음
    img = cv2.imread(digitimg)  # 이미지를 받은 후 img 에 저장
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # img 를 그레이 스케일로 변환 후 gray 에 저장
    cv2.imshow('img', gray)
    ret = cv2.resize(gray, (20,20), fx=1, fy=1, interpolation=cv2.INTER_AREA)  # gray 를 20x20 픽셀 사이즈로 변환후 ret 에 저장

    ret, thr = cv2.threshold(ret, 127, 255, cv2.THRESH_BINARY_INV)  # ret 을 바이너리 이미지로 만들어 thr 에 저장함
    cv2.imshow('ret', thr)

    return thr.reshape(-1, 400).astype(np.float32)  # thr 을 1x400 배열로 변환하여 리턴함
                                                    # astype : 데이터 타입을 바꿔줌
    '''
    numpy.reshape(array, newShape, order='C')
    1st : 대상 어레이
    2nd : 바뀔 모양(int or tuple of ints)  
          => 한쪽을 -1 로 설정하면 전체 갯수에 맞게 알아서 조정됨. ex) (100)->(2,50) 이면 (100)->(-1,2)->(50,2) 
    3rd : 옵션 : C 가 디폴트임
          => C : C 형식 Index 순서, 뒤쪽 차원부터 변경하고 앞쪽 차원을 변경
          => F : 포트란 형식 Index 순서, 앞쪽 차원부터 변경하고 뒤쪽 차원을 변경
    '''


# 초기 데이터 학습을 위한 함수. 학습한 내용은 'digits_for_ocr.txt' 에 저장한다.
def learningDigit():
    img = cv2.imread('digitimg.png')  # 학습데이터를 img 에 저장
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # img 를 그레이스케일 변환 후 gray 에 저장

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]  # gray(실제 손글씨 흑백버전) 배열을 50 개의 서브 어레이로 수직 방향으로 분할하여 row 에 저장
                                                                  # row 를 100개의 서브 어레이로 수평 방향으로 분할하여 cells 에 저장
    x = np.array(cells)

    train = x[:,:].reshape(-1,400).astype(np.float32)

    k = np.arange(10)
    train_labels = np.repeat(k,500)[:, np.newaxis]

    np.savez('digits_for_ocr.npz', train=train, train_labels=train_labels)
    print('데이터 저장')


# 학습한 내용이 저장된 파일을 열어 내용을 읽은 후 traindata 와 traindata_labels 를 리턴한다.
def loadLearningDigit(ocrdata):
    with np.load(ocrdata) as f:
        traindata = f['train']
        traindata_labels = f['train_labels']
    return traindata, traindata_labels


# 인자 test 는 우리가 인식할 손글씨 이미지를 resize20 으로 처리한 리턴값이다.
# KNN 을 이용해 가장 일치하는 결과를 도춯하고 리턴한다.
def OCR_for_Digits(test, traindata, traindata_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(traindata, cv2.ml.ROW_SAMPLE, traindata_labels)
    ret, result, neighbors, dist = knn.findNearest(test, k=5)

    return result


# 각 숫자 파일은 0.png~9.png 이다. 숫자 파일을 20x20 으로 변환한 이미지를 화면에 보여주고 이 숫자를 인식한 결과를 print 로 출력한다.
# 만약 인식한 숫자가 실제 숫자 이미지와 다르면 그에 해당하는 숫자를 키보드로 누르면 이 이미지에 대해 재학습 데이터를 만든다.
def main():
    #learningDigit()
    #'''
    ocrdata = 'digits_for_ocr.npz'
    traindata, traindata_labels = loadLearningDigit(ocrdata)
    digits = [str(x) + '.png' for x in range(10)]  # digits 에 손글씨 입력

    print(traindata.shape)
    print(traindata_labels.shape)

    savez = False
    for digit in digits:
        test = resize20(digit)
        result = OCR_for_Digits(test,traindata,traindata_labels)

        print(result)

        k = cv2.waitKey(0) & 0xFF
        if k > 47 and k < 58:
            savez = True
            traindata = np.append(traindata, test, axis=0)
            new_label = np.array(int(chr(k))).reshape(-1,1)
            traindata_labels = np.append(traindata_labels, new_label, axis=0)

        cv2.destroyAllWindows()
        if savez:
            np.savez('digits_for_ocr.npz', train=traindata, train_lables = traindata_labels)
        #'''
main()









