import numpy as np

traindata = np.random.randint(0, 100, (25, 2)).astype(np.float32)
a = np.random.randint(0,2,(25,1)).astype(np.float32)
red = traindata[a.ravel() == 0]

print(traindata)
print(traindata.shape)
print(' ')
print(a)
print(a.shape)
print(' ')
print(red)
print(red.shape)

print(a.ravel() == 0)