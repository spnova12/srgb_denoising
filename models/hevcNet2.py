import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transforms import *
from models.subNets import *


class Generator_one2many_RDB_no_tanh2(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_no_tanh2, self).__init__()

        self.layer1 = nn.Conv2d(input_channel, 64,  kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64,  kernel_size=3, stride=1, padding=1)

        #self.layer4 = ResidualBlocks(64, 15)
        self.layer4 = RDB_Blocks(64, 16)

        self.layer7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

