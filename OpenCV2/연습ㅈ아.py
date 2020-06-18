import numpy as np

k = np.arange(10)  # k = [0 1 2 3 4 5 6 7 8 9], k.shape = (10,)
train_labels = np.repeat(k, 500)[:, np.newaxis]

print(train_labels.shape)