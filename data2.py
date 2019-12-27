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
            #ZN.append([a[4],a[-1],a[8]])
            ZN.append(a)
        if a[4]=='CU':
            #CU.append([a[4],a[-1],a[8]])
            CU.append(a)
            #test.append(a)
        if a[4]=='FE2':
            #FE2.append([a[4],a[-1],a[8]])

            FE2.append(a)
        if a[4]=='FE3':
            #FE3.append([a[4],a[-1],a[8]])
            FE3.append(a)
        if a[4]=='CO':

            #Co.append([a[4],a[-1],a[8]])
            Co.append(a)
        if a[4]=='MN':
            Mn.append(a)
            #Mn.append([a[4],a[-1],a[8]])

        if a[4]=='CA':
            #Ca.append([a[4],a[-1],a[8]])

            Ca.append(a)
        if a[4]=='MG':
            Mg.append(a)
            #Mg.append([a[4],a[-1],a[8]])

        if a[4]=='K':
            K.append(a)
            #K.append([a[4],a[-1],a[8]])
        if a[4]=='NA':
            Na.append(a)
            #Na.append([a[4],a[-1],a[8]])
#print(CU)
from collections import defaultdict
res = []
def union(array):
    v= defaultdict(list)
    for arr in array:
        if arr[1] not in v:
            v[arr[1]] = arr[2]
        else:
            if arr[2]==v[arr[1]]:
                continue
            else:
                v[arr[1]]+=arr[2]
    for key,value in v.items():
        res.append([key,value])
    return res





# union(CU)
name = ['ZN','CU','FE2','FE3','CO','MN','CA','MG','K','NA']
np.savetxt('ZN',ZN,fmt="%.18f,%.18f",delimiter="\n")


