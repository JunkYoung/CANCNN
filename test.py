import numpy as np

a = np.load('DoS_result.npy', allow_pickle=True)

for i in range(len(a)):
    for j in range(len(a[0])):
        print(a[i][j])