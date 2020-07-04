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
        self.FC41 = nn.Linear(in_features=64,out_features=32)
        self.FC51 = nn.Linear(in_features=64,out_features=32)
        self.FC61 = nn.Linear(in_features=64,out_features=32)

        self.FC12 = nn.Linear(in_features=32,out_features=16)
        self.FC22 = nn.Linear(in_features=32,out_features=16)
        self.FC32 = nn.Linear(in_features=32,out_features=16)
        self.FC42 = nn.Linear(in_features=32,out_features=16)
        self.FC52 = nn.Linear(in_features=32,out_features=16)
        self.FC62 = nn.Linear(in_features=32,out_features=16)

        self.FC13 = nn.Linear(in_features=16,out_features=4)
        self.FC23 = nn.Linear(in_features=16,out_features=4)
        self.FC33 = nn.Linear(in_features=16,out_features=4)
        self.FC43 = nn.Linear(in_features=16,out_features=4)
        self.FC53 = nn.Linear(in_features=16,out_features=4)
        self.FC63 = nn.Linear(in_features=16,out_features=4)

        self.AF = nn.LeakyReLU(negative_slope=0.01, inplace=False)


        def forward(self, input):
            out = self.AF(self.MP(self.L1(input)))
            out = self.AF(self.MP(self.L2(out)))
            out = self.AF(self.MP(self.L3(out)))
            out = self.AF(self.MP(self.L4(out)))
            out = self.AF(self.MP(self.L5(out)))
            out = view(1-,64)

            # o1,o2,o3 are for symmetry plane
            # o4,o5,o6 are for rotation axis
            o1 = self.AF(self.FC11(out))
            o2 = self.AF(self.FC21(out))
            o3 = self.AF(self.FC31(out))
            o4 = self.AF(self.FC41(out))
            o5 = self.AF(self.FC51(out))
            o6 = self.AF(self.FC61(out))

            o1 = self.AF(self.FC12(o1))
            o2 = self.AF(self.FC22(o2))
            o3 = self.AF(self.FC32(o3))
            o4 = self.AF(self.FC42(o4))
            o5 = self.AF(self.FC52(o5))
            o6 = self.AF(self.FC62(o6))

            o1 = self.AF(self.FC13(o1))
            o2 = self.AF(self.FC23(o2))
            o3 = self.AF(self.FC33(o3))
            o1 = self.AF(self.FC43(o4))
            o2 = self.AF(self.FC53(o5))
            o3 = self.AF(self.FC63(o6))

            o1 = o1/torch.norm(o1, dim = 1)
            o2 = o2/torch.norm(o2, dim = 1)
            o3 = o3/torch.norm(o3, dim = 1)
            o4 = o4/torch.norm(o4, dim = 1)
            o5 = o5/torch.norm(o5, dim = 1)
            o6 = o6/torch.norm(o6, dim = 1)

            self.batchoutput = torch.zeros(o1.shape[0], 6, 4)
            self.reshapeOutput(o1, 1)
            self.reshapeOutput(o2, 2)
            self.reshapeOutput(o3, 3)
            self.reshapeOutput(o4, 4)
            self.reshapeOutput(o5, 5)
            self.reshapeOutput(o6, 6)

            return self.batchoutput
        # reshape output
        def reshapeOutput(self, output, pos)
            for i in range(self.batchoutput.shape[0]):
                self.batchoutput[i][pos] = output[i]

class SymmetryDistanceLoss(nn.Module):
    def __init__(self):
        super(SymmetryDistanceLoss, self).__init__()
    
    def forward(self, output, target):
        batchSize = output.shape[0]
        self.loss = torch.tensor(0)
        for batch in range(batchSize):
            self.Q = target[batch]['points']
            self.ClosestGrid = target[batch]['closest']
            self.batch = batch
            self.SymPoints = []
            for i in range(3):
                self.ReflectiveDistance(output[batch][i])

            for i in range(3,6):
                self.RotationDistance(output[batch][i])
            
            self.loss = self.loss + self.totalDis()
        
        return self.loss/batchSize

    def ReflectiveDistance(self, ReflectivePlane):
        # get normal vector of the plane
        nv = ReflectivePlane[0:3]
        d = ReflectivePlane[3]

        for k in range(len(Q)):
            q = Q[k]
            dis = (torch.dot(nv, torch.tensor(q))+d)/(torch.dot(nv, nv))
            q_sym = q - 2*(dis)*nv
            self.SymPoints.append(q_sym)

    def RotationDistance(self, RotationQuater):
        for k in range(len(Q)):
            q = Q[k]
            q_hat = torch.zeros(4)
            q_hat[1:] = q
            q_sym = self.QuaternionProduct(QuaternionProduct(RotationQuater,q_hat), self.QuaternionInverse(RotationQuater))[1:]
            self.SymPoints.append(q_sym)

    def totalDis(self):
        totalDis = 0
        for i in range(self.SymPoints):
            x, y, z = SymPoints[i]
            if x < 0:
                x = 0
            elif x > 32:
                x = 32
            
            if y < 0:
                y = 0
            elif y > 32:
                y = 32
            if z < 0:
                z = 0
            elif z > 32:
                z = 32

            CP = self.ClosestGrid[int(x)][int(y)][int(z)]
            totalDis = totalDis + torch.norm(SymPoints[i] - CP)

        return totalDis

    # 四元数乘法
    # q1q2 = (s1s2 - v1·v2) +s1v2 + s2v1 + v1Xv2
    def QuaternionProduct(self, Qa, Qb):
        Qres = torch.zeros(4)
        Qres[0] = Qa[0]*Qb[0] - torch.dot(Qa[1:], Qb[1:])
        Qres[1:] = Qa[0]*Qb[1:] + Qb[0]*Qa[1:] + torch.cross(Qa[1:], Qb[1:])

        return Qres


    def QuaternionInverse(self, Quaternion):
        Qi = torch.zeros(4)
        Qi[1:] = -Quaternion[1:]
        Qi[0] = Quaternion[0]
        Qi = Qi/ torch.norm(Quaternion)
        return Qi

class RegularizationLoss(nn.Module):
    def __init__(self):
        super(RegularizationLoss, self).__init__()
    
    def forward(self, output):
        batchSize = output.shape[0]
        self.loss = torch.tensor(0)
        for batch in range(batchSize):
            M1 = torch.zeros(3,3)
            M2 = torch.zeros(3,3)
            for i in range(3):
                M1[i] = output[batch][i]

            for i in range(3, 6):
                M2[i-3] = output[batch][i]
            I = torch.eye(3)
            A = torch.mm(M1, torch.t(M1)) - I
            B = torch.mm(M2, torch.t(M2)) - I

            self.loss = self.loss + torch.norm(A, p='fro')**2 + torch.norm(B, p = 'fro')**2

        return self.loss/batchSize
