import numpy as np


import ge_nn


n = ge_nn.Net()
#print(n)
size = 131072 # 128 * 1024
batchsize = 4
x = np.empty((batchsize, size, 4), dtype=np.bool)
print("x.shape")
print(x.shape)
y = n.forward(x)
print("y.shape")
print(y.shape)