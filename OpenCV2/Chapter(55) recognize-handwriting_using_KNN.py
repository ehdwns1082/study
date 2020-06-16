import cv2
import numpy as np
'''
1. 초기데이터 학습시키기
2. 실제 손글씨 인식하기
3. 인식실패시 재학습시키기

traindata = digitimg.png 하고 하면
0 에 해당하는 구역은 traindata[:500, :] 이다.
1 에 해당하는 구역은 traindata[500:1000, :] 이다.

'''

# 인자로 입력된 손글씨 숫자 이미지 파일을 읽어 20x20 픽셀로 변환한 후 인식을 위해 (1,400) 크기의 numpy 배열로 리턴한다.
def resize20(digitimg):
    img = cv2.imread(digitimg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret = cv2.resize(gray, (20,20), fx=1, fy=1, interpolation=cv2.INTER_AREA)

    ret, thr = cv2.threshold(ret, 127, 255, cv2.THRESHOLD_INV)
    cv2.imshow('ret', thr)

    return thr.reshape(-1, 400).astype(np.float32) # astype : 데이터 타입을 바꿔줌


# 초기 데이터 학습을 위한 함수. 학습한 내용은 'digits_for_ocr.npz' 에 저장한다.
def learningDigit():
    img = cv2.imread('digitimg.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
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
    learningDigit()
    ocrdata = 'digits_for_ocr.npz'
    traindata, traindata_labels = loadLearningDigit(ocrdata)
    digits = ['imaes/' + str(x) + '.png' for x in range(10)]

    print(traindata.shape)
    print(traindata_labels.shape)

    savenpz = False
    for digit in digits:
        test = resize20(digit)
        result = OCR_for_Digits(test,traindata,traindata_labels)

        print(result)

        k = cv2.waitKey(0) & 0xFF
        if k > 47 and k < 58:
            savenpz = True
            traindata = np.append(traindata, test, axis=0)
            new_label = np.array(int(chr(k))).reshape(-1,1)
            traindata_labels = np.append(traindata_labels, new_label, axis=0)

        cv2.destroyAllWindows()
        if savenpz:
            np.savez('digits_for_ocr.npz', train=traindata, train_lables = traindata_labels)
main()









