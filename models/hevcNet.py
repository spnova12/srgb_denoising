import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transforms import *
from models.subNets import *
from models.cbam import *



class Generator_one2many_RDB_no_tanh(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_no_tanh, self).__init__()

        self.layer1 = nn.Conv2d(input_channel, 64,  kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64,  kernel_size=4, stride=2, padding=1)

        #self.layer4 = ResidualBlocks(64, 15)
        self.layer4 = RDB_Blocks(64, 16)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
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

class Generator_one2many_RDB_cbam(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_cbam, self).__init__()

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        # self.layer4 = ResidualBlocks(64, 15)
        self.layer4 = RDB_Blocks(64, 16)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

        self.cbam = CBAM(64, 16)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.cbam(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_RDB_cbam_ver2(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_cbam_ver2, self).__init__()

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        # self.layer4 = ResidualBlocks(64, 15)
        self.layer4 = RDB_Blocks(64, 16)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

        self.cbam = CBAM(64, 16)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.cbam(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.layer7(out)
        out = self.cbam(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_RDB_cbam_ver3(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_cbam_ver3, self).__init__()

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        # self.layer4 = ResidualBlocks(64, 15)
        self.layer4 = RDB_Blocks(64, 16)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

        self.cbam = CBAM(64, 16)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.cbam(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.cbam(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x
    # input x
    # torch.Size([1, 3, 298, 530])
    # layer1
    # torch.Size([1, 64, 298, 530])
    # layer2
    # torch.Size([1, 64, 298, 530])
    # layer3
    # torch.Size([1, 64, 149, 265])
    # layer4
    # torch.Size([1, 64, 149, 265])
    # layer7
    # torch.Size([1, 64, 298, 530])
    # layer8
    # torch.Size([1, 64, 298, 530])
    # layer9
    # torch.Size([1, 3, 298, 530])