import os
paths =os.listdir('CA')
for path in paths[:2]:
    t = 'process.sh'
    m = path
    cmd = "./%s %s " % (t,m)
    os.system(cmd)


