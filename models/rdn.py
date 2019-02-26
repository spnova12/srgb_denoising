import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transforms import *
from models.subNets import *


class RDN(nn.Module):
    def __init__(self, input_channel):
        super(RDN, self).__init__()

        self.layer1 = nn.Conv2d(input_channel, 64,  kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(64, 64,  kernel_size=4, stride=2, padding=1)

        # rdb 블럭들.
        self.rdb1 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb2 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb3 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb4 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb5 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb6 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb7 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb8 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb9 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb10 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb11 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb12 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb13 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb14 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb15 = RDB(nChannels=64, nDenselayer=8, growthRate=64)
        self.rdb16 = RDB(nChannels=64, nDenselayer=8, growthRate=64)

        self.layer3 = nn.Conv2d(16*64, 64, kernel_size=1, padding=0, bias=False)

        self.layer4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out_layer1 = self.layer1(x)
        out_layer2 = self.layer2(out_layer1)

        out_rdb1 = self.rdb1(out_layer2)
        out_rdb2 = self.rdb2(out_rdb1)
        out_rdb3 = self.rdb3(out_rdb2)
        out_rdb4 = self.rdb4(out_rdb3)
        out_rdb5 = self.rdb5(out_rdb4)
        out_rdb6 = self.rdb6(out_rdb5)
        out_rdb7 = self.rdb7(out_rdb6)
        out_rdb8 = self.rdb8(out_rdb7)
        out_rdb9 = self.rdb9(out_rdb8)
        out_rdb10 = self.rdb10(out_rdb9)
        out_rdb11 = self.rdb11(out_rdb10)
        out_rdb12 = self.rdb12(out_rdb11)
        out_rdb13 = self.rdb13(out_rdb12)
        out_rdb14 = self.rdb14(out_rdb13)
        out_rdb15 = self.rdb15(out_rdb14)
        out_rdb16 = self.rdb16(out_rdb15)

        concat = torch.cat(
            (
                out_rdb1,
                out_rdb2,
                out_rdb3,
                out_rdb4,
                out_rdb5,
                out_rdb6,
                out_rdb7,
                out_rdb8,
                out_rdb9,
                out_rdb10,
                out_rdb11,
                out_rdb12,
                out_rdb13,
                out_rdb14,
                out_rdb15,
                out_rdb16
            ), 1)

        out = self.layer3(concat)
        out = self.layer4(out)
        out = out + out_layer1

        out = self.layer5(out)
        out = out + x

        # global residual 구조
        return out
