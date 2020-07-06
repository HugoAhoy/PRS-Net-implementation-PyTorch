import utils.binvox_rw as brw
import torch
import numpy as np

def binvox2Tensor(filepath):
    with open(filepath, 'rb') as f:
        model = brw.read_as_3d_array(f)
    voxel = torch.zeros(32,32,32)
    scale = model.dims[0]/32
    for i in range(model.dims[0]):
        for j in range(model.dims[1]):
            for k in range(model.dims[2]):
                if model.data[i][j][k]:
                    voxel[int(i/scale)][int(j/scale)][int(k/scale)] = 1
    return voxel

def getPointCloud(filepath):
    maxPos = float('inf')
    minPos = float('-inf')
    with open(filepath, 'rb') as f:
        datastart = False
        # 开始读数据
        for line in f.read().splitlines():
            if not datastart:
                if line.split()[0] == "DATA":
                    datastart = True
            else:
                cloudPoint = []
                points = line.split()
                x = float(points[0])
                y = float(points[1])
                z = float(points[2])
                maxPos = max(maxPos, x, y, z)
                minPos = min(minPos, x, y, z)
                cloudPoint.append([x, y, z])
        cloudPoint = torch.tensor(cloudPoint)

        # 对标对齐
        cloudPoint = (cloudPoint - min)/max * 32

def calculateClosestGrid(filepath):
    cloudPoint = getPointCloud(filepath)
    ans = np.zeros((32*32*32, 3),dtype='int')
    for i in range(32):
        for j in range(32):
            for k in range(32):
                _, pos = torch.min(torch.norm(cloudPoint - torch.tensor([i, j, k],dtype = torch.float),dim = 1), dim = 0)
                x, y, z = cloudPoint[pos]
                ans[i*32*32+j*32 + k] = [int(x), int(y), int(z)]
    
    return ans 


if __name__ == "__main__":
    print(binvox2Tensor('./data/models-binvox-solid/room00.binvox'))
