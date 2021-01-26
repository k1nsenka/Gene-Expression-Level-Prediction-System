import numpy as np
import matplotlib.pyplot as plt

data_set_x = np.loadtxt(
    fname='/home/abe/data/genome_data/data310/raw/smoothing_out310.csv', #読み込むファイルのパスと名前
    dtype='float', #floatで読み込む
    delimiter=',', #csvなのでカンマで区切る
)
data_set_y = np.loadtxt(
    fname='/home/abe/data/genome_data/data310/raw/smoothing_test_out310.csv', #読み込むファイルのパスと名前
    dtype='float', #floatで読み込む
    delimiter=',', #csvなのでカンマで区切る
)
'''
data_set_x = np.loadtxt(
    fname='/Users/nemomac2/Desktop/data310/raw/smoothing_out310.csv', #読み込むファイルのパスと名前
    dtype='float', #floatで読み込む
    delimiter=',', #csvなのでカンマで区切る
)
data_set_y = np.loadtxt(
    fname='/Users/nemomac2/Desktop/data310/raw/smoothing_test_out310.csv', #読み込むファイルのパスと名前
    dtype='float', #floatで読み込む
    delimiter=',', #csvなのでカンマで区切る
)
'''
print(len(data_set_x))
print(len(data_set_y))
print(len(data_set_x[0]))
print(len(data_set_y[0]))

for i in range(len(data_set_x)):
    print('data{}'.format(i))
    for j in range(len(data_set_x[0])):
        print('data{}, {}'.format(i, j))
        plt.scatter(data_set_x[i][j], data_set_y[i][j], c='black', alpha = 0.2)

plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.grid()
plt.savefig('smooth_scatter.png')
plt.clf()
