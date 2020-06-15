import cv2
import numpy as np

def resize20(digitimg):
    img = cv2.imread(digitimg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret = cv2.resize(gray, (20,20), fx=1, fy=1, interpolation=cv2.INTER_AREA)

    ret, thr = cv2.threshold(ret, 127, 255, cv2.THRESHOLD_INV)
    cv2.imshow('ret', thr)

    return thr.reshape(-1, 400).astype(np.float32) # astype : 데이터 타입을 바꿔줌

def learningDigit():
    img = cv2.imread('digitimg.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]


    x = np.array(cells)



