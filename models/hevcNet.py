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

class Generator_one2many_RDB_cbam_ver4(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_cbam_ver4, self).__init__()

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
        out = self.layer3(out)

        out = self.layer4(out)

        out = self.layer7(out)
        out = self.cbam(out)

        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_RDB_cbam_ver5(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_cbam_ver5, self).__init__()

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

        out = self.layer3(out)
        out = self.layer4(out)
        out = self.cbam(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_RDB_cbam_ver6(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_cbam_ver6, self).__init__()

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
        out = self.layer3(out)
        out = self.cbam(out)

        out = self.layer4(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_RDB_cbam_ver7(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_cbam_ver7, self).__init__()

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
        out = self.layer3(out)

        out = self.cbam(out)
        out = self.layer4(out)

        out = self.layer7(out)
        out = self.cbam(out)

        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_RDB_cbam_ver8(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_cbam_ver8, self).__init__()

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

        out = self.layer3(out)
        out = self.cbam(out)

        out = self.layer4(out)
        out = self.cbam(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_RDB_cbam_ver9(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_cbam_ver9, self).__init__()

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
        out = self.cbam(out)

        out = self.layer4(out)
        out = self.cbam(out)

        out = self.layer7(out)
        out = self.cbam(out)

        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_rir(nn.Module):
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_rir, self).__init__()

        self.numforrg = numforrg          #num of rdb units in one residual group
        self.numofrdb = numofrdb        #num of all rdb units

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        # self.layer4 = ResidualBlocks(64, 15)
        self.RG = RG_Blocks(64, self.numforrg)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

        self.cbam = CBAM(64, 16)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        for i in range(self.numofrdb//self.numforrg):
            inputofrg = out
            outofrg = self.RG(inputofrg)
            out = inputofrg + outofrg

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        return out + x

class Generator_one2many_gd(nn.Module):
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_gd, self).__init__()

        self.numforrg = numforrg          #num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        # self.layer4 = ResidualBlocks(64, 15)
        self.rdb = RDB(64, nDenselayer=8, growthRate=64)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

        self.onebyone = nn.Conv2d(64*self.numofrdb, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        outputsofrdbs = []
        for i in range(self.numofrdb):
            inputofrdb = out
            outputofrdb = self.rdb(inputofrdb)
            out = outputofrdb
            outputsofrdbs.append(out)

        firstoutput = outputsofrdbs[0]
        for idx in range(len(outputsofrdbs)):
            if idx > 0:
                out = torch.cat((firstoutput, outputsofrdbs[idx]), 1)

        out = self.onebyone(out)

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