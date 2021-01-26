import numpy as np
import matplotlib.pyplot as plt

data_set_x = np.loadtxt(
    fname='/Users/nemomac2/Desktop/data310/raw/data_out310.csv', #読み込むファイルのパスと名前
    dtype='float', #floatで読み込む
    delimiter=',', #csvなのでカンマで区切る
)
data_set_y = np.loadtxt(
    fname='/Users/nemomac2/Desktop/data310/raw/data_test_out310.csv', #読み込むファイルのパスと名前
    dtype='float', #floatで読み込む
    delimiter=',', #csvなのでカンマで区切る
)

print(len(data_set_x))
print(len(data_set_y))
print(len(data_set_x[0]))
print(len(data_set_y[0]))

for i in range(len(data_set_x)):
    for j in range(len(data_set_x[0])):
        plt.scatter(data_set_x[i][j], data_set_y[i][j])

plt.title('test')
plt.xlabel('test')
plt.ylabel('test')
plt.grid()
plt.savefig('test.png')
plt.clf()
