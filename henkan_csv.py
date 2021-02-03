import numpy as np
import matplotlib.pyplot as plt
import csv

'''
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

for i in range(len(data_set_x)):
    print('data{}'.format(i))
    for j in range(len(data_set_x[0])):
        print('data{}, {}'.format(i, j))
        with open('/Users/nemomac2/Desktop/data310/trans_data_out.csv', 'a') as f :
            writer = csv.writer(f)
            writer.writerow([data_set_x[i][j]])
        with open('/Users/nemomac2/Desktop/data310/trans_data_test_out.csv', 'a') as fc :
            writer = csv.writer(fc)
            writer.writerow([data_set_y[i][j]])
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

for i in range(len(data_set_x)):
    print('data{}'.format(i))
    for j in range(len(data_set_x[0])):
        print('data{}, {}'.format(i, j))
        with open('/Users/nemomac2/Desktop/data310/trans_smoothing_data_out.csv', 'a') as f :
            writer = csv.writer(f)
            writer.writerow([data_set_x[i][j]])
        with open('/Users/nemomac2/Desktop/data310/trans_data_smoothing_test_out.csv', 'a') as fc :
            writer = csv.writer(fc)
            writer.writerow([data_set_y[i][j]])