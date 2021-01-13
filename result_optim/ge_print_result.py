import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


for i in range(8):
    data = pd.read_csv('../train_result_optim/result{}.csv'.format(i), sep=" ")
    if i == 0:
        optimizer = 'Adam'
    elif i == 1:
        optimizer = 'SGD'
    elif i == 2:
        optimizer = 'RMSprop'
    elif i == 3:
        optimizer = 'Adadelta'
    elif i == 4:
        optimizer = 'AdamW'
    elif i == 5:
        optimizer = 'Adagrad'
    elif i == 6:
        optimizer = 'ASGD'
    elif i == 7:
        optimizer = 'Adamax'
    else :
        print('please input optimizer')
        sys.exit(1)
    #print(data.keys)
    #print(data['poissonLoss'])
    plt.ylim(0, 1)
    plt.plot(data['epoch'], data['poissonLoss'], label = optimizer)
plt.title('Losses for each optimization')
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('./result_graph/loss_log_optim.png')