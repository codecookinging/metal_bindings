import pandas as pd
import numpy as np
# data = pd.read_table('BioLip_2013-03-6.txt')
# print(data[0])
#a = numpy.loadtxt('odom.txt')
def repalce(str):
    new = ''
    for s in str:
        if s==' ':
            new+='_'
        if s!=' ':
            new+=s
    return new
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
        if a[4]=='CU':
            a[8] = repalce(a[8])
            CU.append([a[0],a[1],a[-1],a[8]])
        if a[4]=='FE2':
            a[8] = repalce(a[8])
            FE2.append([a[0],a[1],a[-1],a[8]])
        if a[4]=='FE3':
            a[8] = repalce(a[8])
            FE3.append([a[0],a[1],a[-1],a[8]])
        if a[4]=='CO':
            a[8] = repalce(a[8])
            Co.append([a[0],a[1],a[-1],a[8]])
        if a[4]=='MN':
            a[8] = repalce(a[8])
            Mn.append([a[0],a[1],a[-1],a[8]])
        if a[4]=='CA':
            a[8] = repalce(a[8])
            Ca.append([a[0],a[1],a[-1],a[8]])
        if a[4]=='MG':
            a[8] = repalce(a[8])
            Mg.append([a[0],a[1],a[-1],a[8]])
        if a[4]=='K':
            a[8] = repalce(a[8])
            K.append([a[0],a[1],a[-1],a[8]])
        if a[4]=='NA':
            a[8] = repalce(a[8])
            Na.append([a[0],a[1],a[-1],a[8]])
with open('CU.fa', "w") as fa_out:
    for i in range(len(CU)):
            fa_out.write('>'+CU[i][0]+CU[i][1]+CU[i][-1]+'\n')
            while len(CU[i][-2]) > 60:  # 每行写60个碱基
                fa_out.write(CU[i][-2][:60] + "\n")
                CU[i][-2] = CU[i][-2][60:]
            else:
                fa_out.write(CU[i][-2]+'\n')
with open('CA.fa', "w") as fa_out:
    for i in range(len(Ca)):
            fa_out.write('>'+Ca[i][0]+Ca[i][1]+Ca[i][-1]+'\n')
            while len(Ca[i][-2]) > 60:  # 每行写60个碱基
                fa_out.write(Ca[i][-2][:60] + "\n")
                Ca[i][-2] = Ca[i][-2][60:]
            else:
                fa_out.write(Ca[i][-2]+'\n')

with open('ZN.fa', "w") as fa_out:
    for i in range(len(ZN)):
            fa_out.write('>'+ZN[i][0]+ZN[i][1]+ZN[i][-1]+'\n')
            while len(ZN[i][-2]) > 60:  # 每行写60个碱基
                fa_out.write(ZN[i][-2][:60] + "\n")
                ZN[i][-2] = ZN[i][-2][60:]
            else:
                fa_out.write(ZN[i][-2]+'\n')

with open('Co.fa', "w") as fa_out:
    for i in range(len(Co)):
            fa_out.write('>'+Co[i][0]+Co[i][1]+Co[i][-1]+'\n')
            while len(Co[i][-2]) > 60:  # 每行写60个碱基
                fa_out.write(Co[i][-2][:60] + "\n")
                Co[i][-2] = Co[i][-2][60:]
            else:
                fa_out.write(Co[i][-2]+'\n')

with open('FE2.fa', "w") as fa_out:
    for i in range(len(FE2)):
            fa_out.write('>'+FE2[i][0]+FE2[i][1]+FE2[i][-1]+'\n')
            while len(FE2[i][-2]) > 60:  # 每行写60个碱基
                fa_out.write(FE2[i][-2][:60] + "\n")
                FE2[i][-2] = FE2[i][-2][60:]
            else:
                fa_out.write(FE2[i][-2]+'\n')


# with open('FE3.fa', "w") as fa_out:
#     for i in range(len(FE3)):
#             fa_out.write('>'+FE3[i][0]+FE3[i][1]+'\n')
#             while len(FE3[i][-1]) > 60:  # 每行写60个碱基
#                 fa_out.write(FE3[i][-1][:60] + "\n")
#                 Co[i][-1] = FE3[i][-1][60:]
#             else:
#                 fa_out.write(FE3[i][-1]+'\n')

with open('MN.fa', "w") as fa_out:
    for i in range(len(Mn)):
            fa_out.write('>'+Mn[i][0]+Mn[i][1]+Mn[i][-1]+'\n')
            while len(Mn[i][-2]) > 60:  # 每行写60个碱基
                fa_out.write(Mn[i][-2][:60] + "\n")
                Mn[i][-2] = Mn[i][-2][60:]
            else:
                fa_out.write(Mn[i][-2]+'\n')

with open('MG.fa', "w") as fa_out:
    for i in range(len(Mg)):
            fa_out.write('>'+Mg[i][0]+Mg[i][1]+Mg[i][-1]+'\n')
            while len(Mg[i][-2]) > 60:  # 每行写60个碱基
                fa_out.write(Mg[i][-2][:60] + "\n")
                Mg[i][-2] = Mg[i][-2][60:]
            else:
                fa_out.write(Mg[i][-2]+'\n')

with open('K.fa', "w") as fa_out:
    for i in range(len(K)):
            fa_out.write('>'+K[i][0]+K[i][1]+K[i][-1]+'\n')
            while len(K[i][-2]) > 60:  # 每行写60个碱基
                fa_out.write(K[i][-2][:60] + "\n")
                K[i][-2] = K[i][-2][60:]
            else:
                fa_out.write(K[i][-2]+'\n')

with open('NA.fa', "w") as fa_out:
    for i in range(len(Na)):
            fa_out.write('>'+Na[i][0]+Na[i][1]+Na[i][-1]+'\n')
            while len(Na[i][-2]) > 60:  # 每行写60个碱基
                fa_out.write(Na[i][-2][:60] + "\n")
                Na[i][-2] = Na[i][-2][60:]
            else:
                fa_out.write(Na[i][-2]+'\n')




