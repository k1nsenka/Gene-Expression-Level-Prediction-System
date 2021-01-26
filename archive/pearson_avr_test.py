import csv
import numpy as np
import torch
import math


test_score = []
row_f =[]
with open('./smoothing/smoothing_pearsonr.csv', 'r') as fp :
    reader = csv.reader(fp)
    for row in reader:
        for r in row:
            ele = float(r)
            if math.isnan(ele):
                ele = 0.0
            row_f.append(ele)
        test_score.append(row_f)


# [0.123, 1.23, 123.0]

#test_score = torch.tensor(test_score)

#print(test_score[0])
#print(len(test_score[0]))


avr_test_score = np.mean(test_score)
var_test_score = np.var(test_score)
print('test pearson score:{}, \n max:{} index:{}, \n min:{} index{}'.format(avr_test_score, np.max(test_score), np.argmax(test_score), np.min(test_score), np.argmin(test_score)))