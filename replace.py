def repalce(str):
    new = ''
    for s in str:
        if s==' ':
            new+='_'
        if s!=' ':
            new+=s
    return new

str = 'A123 B356'

new = repalce(str)
import pandas as pd
import numpy as np
# data = pd.read_table('BioLip_2013-03-6.txt')
# print(data[0])
#a = numpy.loadtxt('odom.txt')
ZN = []
test_zn = []
CU = []
test_cu = []
FE2 = []
test_fe2 = []
FE3 = []
test_fe3 = []
Co=[]
test_co = []
Mn = []
test_mn = []
Ca = []
test_ca = []
Mg = []
test_mg = []
K = []
test_k = []
Na = []
test_na = []
test = []
with open('BioLip_2013-03-6.txt','r') as fr:
    lines = fr.readlines()
    #print(lines[0])
    print(type(lines[0]))
    #a = lines[0].strip()
    for line in lines:
        a = line.strip().split('\t') # 长度为20
        if a[4]=='ZN':
            a[8] = repalce(a[8])
            ZN.append([a[0],a[1],a[-1],a[8]])

with open('ZNnew.fa', "w") as fa_out:
    for i in range(len(ZN)):
            fa_out.write('>'+ZN[i][0]+ZN[i][1]+ZN[i][-1]+'\n')
            while len(ZN[i][-2]) > 60:  # 每行写60个碱基
                fa_out.write(ZN[i][-2][:60] + "\n")
                ZN[i][-2] = ZN[i][-2][60:]
            else:
                fa_out.write(ZN[i][-2]+'\n')



