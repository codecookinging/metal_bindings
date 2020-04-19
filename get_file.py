import os
class Solution:
    def __init__(self,file):
        self.file = file
        self.sequence = []#每个蛋白质的起始行 终止行
        self.len = 0  #蛋白质个数
    def read(self):
        with open(self.file,'r') as fr:
            num = 0 #
            lines = fr.readlines()
            print(lines)
            for line in lines:
                if line[0]=='>':
                    self.sequence.append(num)
                    self.len+=1
                num+=1
            self.sequence.append(num) #num 就是最后一行

    def create_folder(self):
        with open(self.file, 'r') as fr:
            lines = fr.readlines()
            for i in range(len(self.sequence)-1):
                a = lines[self.sequence[i]:self.sequence[i+1]]
                #print(a)
                #print(a[0][1:6])
                folder = os.path.join('MN', a[0][1:6])  # 文件夹路径
                os.makedirs(folder)
                #将a 写入这个文件夹下去
                file = os.path.join(folder,'sequence.fa')
                with open(file,'a+') as fout:
                    #print(a)
                    for line in a:
                        fout.write(line)
                         #print(line)

s = Solution('CA30')
a = s.read()
print(s.sequence)
s.create_folder()
# print(a)

