import os
import DataPreProc
import numpy as np

def OBJtoBinvox(datadir, filedir, index):
    binvoxcmd = "binvox -e -ri -d 32 {}\\{}.obj".format(datadir, index)
    # print(cpcmd)
    # print(binvoxcmd)
    os.system(binvoxcmd)

def OBJtoPCD(datadir, filedir, index):
    # ClosestSamplingCmd = "pcl_mesh_sampling.exe -no_vis_result {} {}\\{}_closest.pcd".format(filedir, datadir, index)
    SamplingCmd = "pcl_mesh_sampling.exe -n_samples 1000 -no_vis_result {} {}\\{}.pcd".format(filedir, datadir, index)
    # print(ClosestSamplingCmd)
    print(SamplingCmd)
    # os.system(ClosestSamplingCmd)
    os.system(SamplingCmd)

def getClosestGrid(datadir, index):
    # pcdPath = "{}\\{}_closest.pcd".format(datadir, index)
    pcdPath = "{}\\{}.pcd".format(datadir, index)
    closest = DataPreProc.calculateClosestGrid(pcdPath)
    np.save("{}\\{}.npy".format(datadir, index),closest)


allOBJPath = input("allOBJPath:")
datadir = input("data dir:")

i = 0
with open(allOBJPath, 'r') as f:
    for p in f.read().splitlines():
        os.chdir(datadir)
        os.mkdir(str(i))
        os.chdir(str(i))
        pwd = os.getcwd()
        # 拷贝obj
        cpcmd = "cp {} {}\\{}.obj".format(p, pwd, i)
        os.system(cpcmd)
        OBJtoBinvox(pwd, p, i)
        OBJtoPCD(pwd, p, i)
        getClosestGrid(pwd, i)
        i = i + 1
