import matplotlib.pyplot as plt
import numpy as np
# 描画用サンプルデータ
#x= np.array([0,1,2,3,4])
y = np.array([2, 2, 3, 4, 5])
print(y)
#print(range(y))

plt.figure(figsize=(10,1))
plt.bar(range(len(y)), y)
plt.savefig('test.png')
plt.clf()