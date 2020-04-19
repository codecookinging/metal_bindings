from __future__ import print_function, division
import os
from torch.utils.data import Dataset
import math
import numpy as np
from torch.utils.data import DataLoader
class mydataset(Dataset):
    """Dataset
    Args:
        dataset_path (string): Path to the dataset folder.
        sample_list (string): Path relative to the dataset_path pointing to
                            the location of train-split/test-split file.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    def __init__(self, dataset_path, sample_list, transform=None):
        self.dataset_path = dataset_path
        self.sample_list = sample_list
        self.sample_paths = self._get_sample_paths()
        self.transform = transform

    def normalNum(self,num):
        num = int(num)
        num = 1 / (1 + math.exp(0 - num))
        return num

    def get_lable(self,str):
        with open(str, "r") as fa_out:
            lines = fa_out.readlines()
            position, strings = lines[0], lines[1:]
            # print(position)
            # print(strings)
            res = ''
            for s in strings:
                s = s.strip()
                res += s
            #print(res)
            label = [0]*len(res)
            # print(len(res))
            # print(res[383])
            # print(res[391])
            position = position[6:]
            temp = position.strip().split('_')
            for j in temp:
                key = j[0]
                value = int(j[1:]) - 1
                label[value] = 1
            return label
    def signPssM(self,path):
        if os.path.exists(path):
            fin = open(path)
            lines = fin.readlines()
            pssm = []
            count = 0
            for line in lines:
                # print(line)
                line = line.strip()
                if line != "":
                    if line[0].isdigit():
                        count += 1
                        tmp = line.split()
                        length = len(tmp)
                        tmp = tmp[2:22]
                        # print(tmp)
                        length = len(tmp)
                        for j in range(length):
                            tmp[j] = self.normalNum(tmp[j])
                        assert (len(tmp)) == 20
                        pssm.append(tmp)
            return pssm
        return 0

    def _get_sample_paths(self):
        sample_paths = []
        try:
            with open(self.sample_list,'r') as sample_file:
                for sample in sample_file:
                    sample_paths.append(sample.rstrip('\n'))
            return sample_paths
        except IOError:
            print('Split file not found, please make sure the path to dataset',
                  'and the path to the train-split/test-split file is correct')

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        input_key = 'input'
        label_key = 'label'
        input_path = os.path.join(self.dataset_path,self.sample_paths[idx],
                                  self.sample_paths[idx]+'.pssm')

        label_path = os.path.join(self.dataset_path,self.sample_paths[idx],
                                  'sequence.fa')

        input = self.signPssM(input_path)
        input = np.array(input) # list to array
        input = np.expand_dims(input,0) #拓展维度
        label = self.get_lable(label_path)
        label = np.array(label)
        label = np.expand_dims(label, 2)
         #The image is 16 bit so its normalized by dividing by 2**16

        sample = {input_key: input,
                  label_key: label}

        if self.transform:
            sample = self.transform(sample)

        return sample

# test_data = mydataset('../CA','test.txt')
# samp1 = test_data.__getitem__(2)
# print(samp1['label'].shape)
# dataloader_train = DataLoader(test_data, batch_size=1)
#
# for batched_sample in dataloader_train:
#     # convert the sample into proper data formats and send to GPU
#     input = batched_sample['input']

