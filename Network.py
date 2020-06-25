import torch.nn as nn

class PRSNet(nn.Module):
    def __init__(self):
        super(PRSNet, self).__init__()    
        self.L1 = nn.Conv3d(in_channels = 4,
                            out_channels = 8,
                            kernel_size= 3,
                            stride = 1,
                            padding = 1)
        self.L2 = nn.Conv3d(in_channels = 4,
                            out_channels = 8,
                            kernel_size= 3,
                            stride = 1,
                            padding = 1)
        self.L3 = nn.Conv3d(in_channels = 4,
                            out_channels = 8,
                            kernel_size= 3,
                            stride = 1,
                            padding = 1)
        self.L4 = nn.Conv3d(in_channels = 4,
                            out_channels = 8,
                            kernel_size= 3,
                            stride = 1,
                            padding = 1)
        self.L5 = nn.Conv3d(in_channels = 4,
                            out_channels = 8,
                            kernel_size= 3,
                            stride = 1,
                            padding = 1)

        self.MP = nn.MaxPool3d(kernel_size = 2)

        self.FC11 = nn.Linear(in_features=64,out_features=32)
        self.FC21 = nn.Linear(in_features=64,out_features=32)
        self.FC31 = nn.Linear(in_features=64,out_features=32)

        self.FC12 = nn.Linear(in_features=32,out_features=16)
        self.FC22 = nn.Linear(in_features=32,out_features=16)
        self.FC32 = nn.Linear(in_features=32,out_features=16)

        self.FC13 = nn.Linear(in_features=16,out_features=4)
        self.FC23 = nn.Linear(in_features=16,out_features=4)
        self.FC33 = nn.Linear(in_features=16,out_features=4)

        self.AF = nn.LeakyReLU(negative_slope=0.01, inplace=False)


        def forward(self, input):
            out = self.AF(self.MP(self.L1(input)))
            out = self.AF(self.MP(self.L2(out)))
            out = self.AF(self.MP(self.L3(out)))
            out = self.AF(self.MP(self.L4(out)))
            out = self.AF(self.MP(self.L5(out)))
            out = view(1-,64)

            o1 = self.AF(self.FC11(out))
            o2 = self.AF(self.FC21(out))
            o3 = self.AF(self.FC31(out))

            o1 = self.AF(self.FC12(o1))
            o2 = self.AF(self.FC22(o2))
            o3 = self.AF(self.FC32(o3))

            o1 = self.AF(self.FC13(o1))
            o2 = self.AF(self.FC23(o2))
            o3 = self.AF(self.FC33(o3))

            return (o1,o2,o3)

class SymmetryDistanceLoss(nn.Module):
    def __init__(self):
        super(SymmetryDistanceLoss, self).__init__()
    
    def forward(self, output, target):
        # todo


class RegularizationLoss(nn.Module):
    def __init__(self):
        super(RegularizationLoss, self).__init__()
    
    def forward(self, output, target):
        # todo