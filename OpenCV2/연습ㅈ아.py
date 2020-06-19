import numpy as np

k = np.arange(10)  # k = [0 1 2 3 4 5 6 7 8 9], k.shape = (10,)
train_labels = np.repeat(k, 500)[:, np.newaxis]  # 길이 10짜리 배열 k를 500번 반복하여 5000개 cell 에 대한 label 배열을 만듬
print(train_labels.shape)
print(train_labels)

np.savetxt('digits_for_ocr_labels.txt', train_labels, fmt='%2d', delimiter=' ')
labels = 'digits_for_ocr_labels.txt'
traindata_labels = np.loadtxt(labels)

print(traindata_labels.shape)
print(traindata_labels)

new=traindata_labels[:, np.newaxis]
print(new.shape)
print(new)