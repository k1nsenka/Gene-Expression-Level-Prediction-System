import numpy as np
import matplotlib.pyplot as plt
import h5py

#データ
data = h5py.File('/home/abe/data/genome_data/l131k_w128.h5')



out_data = data['test_out']
sample_raw = out_data[309, :, 955]
'''
sample_avr = []
sample_avr.append(sample_raw[0])
for i in range(1, 1024):
    sample_avr.append((sample_raw[i-1] + sample_raw[i]) / 2)
'''
thr = np.average(sample_raw)

window = 20 # 移動平均の範囲
w = np.ones(window)/window
#print(w)
sample_avr = np.convolve(sample_raw, w, mode='same')



#plt.figure(figsize=(10,1))
plt.plot(range(1024), sample_raw, label = 'raw_data', alpha = 0.4)
plt.plot(range(1024), sample_avr, label = 'filtered_data')
plt.title('raw data and filtered data')
plt.xlabel('location of the genome')
plt.hlines(thr, 0, 1023, linestyle='--', color='r',label='threshold')
plt.ylabel('amount of gene expression')
plt.legend()
#plt.subplots_adjust(top=0.95, bottom = 0.95)
plt.savefig('filtering.png')

