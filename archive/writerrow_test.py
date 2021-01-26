import csv
import numpy as np

a = [[1, 1, 1], [2, 2, 2], [1, 1, 1], [2, 2, 2]]
print(list(range(3)))

'''
for i in range(3):
    x = []
    for j in range(4):
        x.append(a[j][i])
    print(x)
'''


'''
with open('test.csv', 'w') as f :
    writer = csv.writer(f)
    writer.writerows(a)

b = [[1, 1, 1], [2, 2, 2]]

with open('test.csv', 'a') as f :
    writer = csv.writer(f)
    writer.writerows(b)
'''