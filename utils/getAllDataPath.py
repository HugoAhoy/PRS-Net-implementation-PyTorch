import os

path = input()
alldir = os.listdir(path)
res = []

for i in alldir:
    fullPath = path+"\\"+i + "\\models"
    files = os.listdir(fullPath)
    for f in files:
        if f[-3:] == "obj":
            res.append(fullPath+"\\"+ f)

with open("allOBJPath.txt",'w') as f:
    for i in res:
        f.write(i+"\n")
