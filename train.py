import Network
import MyDataset
from torch.utils.data import Dataloader
import torch.optim as optim

batchsize = 4
epoch = 100
RegularWeight = 25
LR = 0.01
resolution = 32

def train():
    prsnet = PRSNet()
    # 设置gpu
    device = torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
    prsnet.to(device)

    optimizer = optim.Adam(prsnet.parameters(), lr = LR)
    LossSym = SymmetryDistanceLoss()
    LossReg = RegularizationLoss()

    dataset = MyDataset()
    dataloader = Dataloader(dataset, batch_size = batchsize, shuffle = True)
    bestLoss = float("inf")
    for e in range(1,epoch):
        totalLoss = 0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            voxel = torch.zeros(batchsize, resolution, resolution, resolution)
            for j in range(batchsize):
                voxel[j] = torch.tensor(data[j]['voxel'])
            output = prsnet(data['voxel'])

            loss = LossSym(output, data) + RegularWeight*LossReg(output, data)
            
            totalLoss = totalLoss + int(loss)
            
            loss.backward()
            optimizer.step()
            
        if totalLoss < bestLoss:
            torch.save(prsnet, 'net_best.pkl')

        if e % 10 == 0:
            torch.save(prsnet, 'net{}.pkl'.format(int(e/10)))
    
            
if __name__ == "__main__":
    train()