import csv


kfold_loss = [1, 2, 3]

with open('kfold_loss.csv', 'w') as f :
    writer = csv.writer(f)
    writer.writerows(enumerate(kfold_loss))
#モデル評価