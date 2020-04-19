import os
def split_data(datapath):

    dir = os.listdir(datapath)
    train_data =dir[:int(len(dir)*0.8)]
    valid_data = dir[int(len(dir)*0.8):int(len(dir)*0.9)]
    test_data = dir[int(len(dir)*0.9):]
    with open('train.txt','w') as f:
        for i in train_data:
            f.write(i)
            f.write('\n')
    with open('valid.txt','w') as f:
        for i in valid_data:
            f.write(i)
            f.write('\n')
    with open('test.txt','w') as f:
        for i in test_data:
            f.write(i)
            f.write('\n')
split_data('../CA')


        



