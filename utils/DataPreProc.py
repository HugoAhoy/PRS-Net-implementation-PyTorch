import binvox_rw as brw
import torch
import numpy as np
import time

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
    maxPos = float('-inf')
    minPos = float('inf')
    with open(filepath, 'r') as f:
        datastart = False
        pointCloud = []
        # 开始读数据
        for line in f.read().splitlines():
            if not datastart:
                if line.split()[0] == "DATA":
                    datastart = True
            else:
                points = line.split()
                x = float(points[0])
                y = float(points[1])
                z = float(points[2])
                maxPos = max(maxPos, x, y, z)
                minPos = min(minPos, x, y, z)
                pointCloud.append([x, y, z])
        pointCloud = torch.tensor(pointCloud)
        # 对标对齐
        pointCloud = (pointCloud - minPos)/(maxPos - minPos) * 32
        return pointCloud

# return a numpy array
def calculateClosestGrid(filepath):
    # pointCloud = getPointCloud(filepath).cuda()
    pointCloud = getPointCloud(filepath)
    ans = np.zeros((32*32*32, 3))
    for i in range(32):
        for j in range(32):
            for k in range(32):
                # p = torch.tensor([i, j, k],dtype = torch.float).cuda()
                p = torch.tensor([i, j, k],dtype = torch.float)
                # print(pointCloud)
                subsect = pointCloud - p
                dis = torch.norm(subsect,dim = 1)
                _, pos = torch.min(dis, dim = 0)
                ans[i*32*32+j*32 + k] = [x, y, z]
                x, y, z = pointCloud[pos]
    return ans


if __name__ == "__main__":
    # print(binvox2Tensor('./data/models-binvox-solid/room00.binvox'))
    s = time.time()
    ans = calculateClosestGrid("D:\\GitRepository\\PRS-Net-implementation-PyTorch\\data\\datas\\0\\0.pcd")
    e = time.time()
    print(ans)
    print(e - s)
    # getPointCloud("D:\\GitRepository\\PRS-Net-implementation-PyTorch\\data\\datas\\0\\0_closest.pcd")
