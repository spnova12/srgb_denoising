import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transforms import *
from models.subNets import *
import math


class hevcnet_espcn(nn.Module):
    def __init__(self, input_channel):
        super(hevcnet_espcn, self).__init__()

        self.layer1 = nn.Conv2d(input_channel, 64,  kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64,  kernel_size=4, stride=2, padding=1)

        self.layer4 = RDB_Blocks(64, 16)

        #self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(64, 4 * 64, kernel_size=3, stride=1, padding=1)
        self.layer7 = nn.PixelShuffle(2)

        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.layer5(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

