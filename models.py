#imports
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#imports from https://github.com/hedixia/HeavyBallNODE
from HeavyBallNODE.source import *
from HeavyBallNODE import basehelper, misc
import HeavyBallNODE.base as hb

#base models
class RobNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inlayer = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 5)
        self.fc5 = nn.Linear(5, 5)
        self.fc6 = nn.Linear(5, 5)
        self.outlayer = nn.Linear(50, 3)

    def forward(self, h, x):
        x = self.inlayer(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.fc4(x)
        x = F.gelu(x)
        x = self.fc5(x)
        x = F.gelu(x)
        x = self.fc6(x)
        x = F.gelu(x)
        x = self.outlayer(x)
        x = F.gelu(x)
        output = x
        return output

class PolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inlayer = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.outlayer = nn.Linear(10, 20)

    def forward(self, h, x):
        x = self.inlayer(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.outlayer(x)
        x = F.gelu(x)
        output = x
        return output

#heavy ball NODEs
class RobHB(nn.Module):
    def __init__(self):
        super(RobHB, self).__init__()
        self.cell = hb.HeavyBallNODE(RobNet())
        self.outlayer = nn.Linear(50, 3)

    def forward(self, t, x):
        out = self.outlayer(x)
        return out

class PolHB(nn.Module):
    def __init__(self):
        super(PolHB, self).__init__()
        self.cell = hb.HeavyBallNODE(PolNet())
        self.outlayer = nn.Linear(20, 10)

    def forward(self, t, x):
        out = self.outlayer(x)
        return out

#scaled models
#robscale = torch.amax(robdat, axis=1)[rob_slow] - torch.amin(robdat, axis=1)[rob_slow]
#polscale = torch.amax(poldat, axis=1)[pol_slow] - torch.amin(poldat, axis=1)[pol_slow]
