
with open('label.txt' ,'r') as f:
    for i in range(4229):
        line = f.readline()
        with open('label_index.txt', 'a') as wf:
            wf.write('{}'.format(i+1) + ' ' + line)

