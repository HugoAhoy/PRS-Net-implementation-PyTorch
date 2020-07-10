from torch.utils.data import Dataset
import DataPreProc
import numpy as np

class MyDataset(Dataset):
    def __init__(self, DataRoot, Train = True):
        datas = []
        self.filename = 'train.txt'
        if not Train:
            self.filename = 'validate.txt'
        with open(os.path.join(DataRoot, self.filename)) as f:
            for i in f.read().splitlines():
                datas.append(i)
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # todo
        voxel = DataPreProc.binvox2Tensor('./data/datas/{}.binvox'.format(index))
        pointCloud = DataPreProc.getPointCloud('./data/datas/{}.pcd'.format(index))
        closest = DataPreProc.calculateClosestGrid('./data/datas/{}.npy'.format(index))
        target = {
            'voxel':voxel,
            'points':pointCloud,
            'closest':closest
        }
        return target