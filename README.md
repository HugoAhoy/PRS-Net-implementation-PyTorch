# PRS-Net-implementation-PyTorch
This repo implements the PRS-Net proposed by Lin Gao, ICT
The access to the paper is: [PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models](https://arxiv.org/abs/1910.06511v5)

## Proj Structure
- ~root
    - DataPreProc.py (functions to get voxel, point cloud and closest grid for each point)
    - LossFunction.py (two loss functions proposed in the paper)
    - MyDataset.py
    -Network.py
    - train.py
- utils
    - binvox-rw.py (a file cloned from others' repo, which helps to read `.binvox` file to numpy array)