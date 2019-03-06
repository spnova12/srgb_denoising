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

        RGlist = []
        for i in range(self.numofrdb // self.numforrg):
            RGlist.append(RG_Blocks(64, self.numforrg))

        self.RGlist = nn.Sequential(*RGlist)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        for RG in self.RGlist:
            inputofrg = out
            outofrg = RG(inputofrg)
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

        RDBList = []
        for i in range(self.numofrdb):
            RDBList.append(RDB(64, nDenselayer=8, growthRate=64))

        self.RDBlist = nn.Sequential(*RDBList)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

        self.onebyone = nn.Conv2d(64*self.numofrdb, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        outputsofrdbs = []
        for RDB in self.RDBlist:
            inputofrdb = out
            outputofrdb = RDB(inputofrdb)
            out = outputofrdb
            outputsofrdbs.append(out)

        firstoutput = outputsofrdbs[0]
        for idx in range(len(outputsofrdbs)):
            if idx > 0:
                out = torch.cat((firstoutput, outputsofrdbs[idx]), 1)
                firstoutput = out

        out = self.onebyone(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_rir_gd_mix(nn.Module):     #rir gd
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_rir_gd_mix, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        RGlist = []
        for i in range(self.numofrdb // self.numforrg):
            RGlist.append(RG_Blocks(64, self.numforrg))

        self.RGlist = nn.Sequential(*RGlist)


        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

        self.onebyone = nn.Conv2d(64 * self.numofrdb // self.numforrg, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        outputsofrdbs = []
        for RG in self.RGlist:
            inputofrdb = out
            outputofrdb = RG(inputofrdb)
            out = inputofrdb + outputofrdb
            outputsofrdbs.append(out)

        firstoutput = outputsofrdbs[0]
        for idx in range(len(outputsofrdbs)):
            if idx > 0:
                out = torch.cat((firstoutput, outputsofrdbs[idx]), 1)
                firstoutput = out

        out = self.onebyone(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_gd_rir(nn.Module):    #gd rir
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_gd_rir, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.rdb1 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb2 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb3 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb4 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone1 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb5 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb6 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb7 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb8 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone2 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb9 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb10 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb11 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb12 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone3 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb13 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb14 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb15 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb16 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone4 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out1 = self.rdb1(out)
        out2 = self.rdb2(out1)
        out3 = self.rdb3(out2)
        out4 = self.rdb4(out3)
        concated = torch.cat((out1, out2, out3, out4), 1)
        out5 = out + self.oneone1(concated)

        out6 = self.rdb5(out5)
        out7 = self.rdb6(out6)
        out8 = self.rdb7(out7)
        out9 = self.rdb8(out8)
        concated = torch.cat((out6, out7, out8, out9), 1)
        out10 = out5 + self.oneone2(concated)

        out11 = self.rdb9(out10)
        out12 = self.rdb10(out11)
        out13 = self.rdb11(out12)
        out14 = self.rdb12(out13)
        concated = torch.cat((out11, out12, out13, out14), 1)
        out15 = out10 + self.oneone3(concated)

        out16 = self.rdb13(out15)
        out17 = self.rdb14(out16)
        out18 = self.rdb15(out17)
        out19 = self.rdb16(out18)
        concated = torch.cat((out16, out17, out18, out19), 1)
        out20 = out15 + self.oneone4(concated)

        out = self.layer7(out20)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_gd_rir_recursive(nn.Module):  # gd rir recursive
    def __init__(self, input_channel, numforrg, numofrdb, timevalue=3):
        super(Generator_one2many_gd_rir_recursive, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units
        self.timevalue = timevalue # num of repeat

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.rdb1 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb2 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb3 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb4 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone1 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb5 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb6 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb7 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb8 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone2 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb9 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb10 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb11 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb12 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone3 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb13 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb14 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb15 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb16 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone4 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        for i in range(self.timevalue):
            input = out
            out1 = self.rdb1(input)
            out2 = self.rdb2(out1)
            out3 = self.rdb3(out2)
            out4 = self.rdb4(out3)
            concated = torch.cat((out1, out2, out3, out4), 1)
            out = input + self.oneone1(concated)

        for i in range(self.timevalue):
            input = out
            out6 = self.rdb5(input)
            out7 = self.rdb6(out6)
            out8 = self.rdb7(out7)
            out9 = self.rdb8(out8)
            concated = torch.cat((out6, out7, out8, out9), 1)
            out = input + self.oneone2(concated)


        for i in range(self.timevalue):
            input = out
            out11 = self.rdb9(input)
            out12 = self.rdb10(out11)
            out13 = self.rdb11(out12)
            out14 = self.rdb12(out13)
            concated = torch.cat((out11, out12, out13, out14), 1)
            out = input + self.oneone3(concated)

        for i in range(self.timevalue):
            input = out
            out16 = self.rdb13(input)
            out17 = self.rdb14(out16)
            out18 = self.rdb15(out17)
            out19 = self.rdb16(out18)
            concated = torch.cat((out16, out17, out18, out19), 1)
            out = input + self.oneone4(concated)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_gd_rir_no_relu(nn.Module):
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_gd_rir_no_relu, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        # self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.rdb1 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb2 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb3 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb4 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone1 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb5 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb6 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb7 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb8 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone2 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb9 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb10 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb11 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb12 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone3 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb13 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb14 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb15 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb16 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone4 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        # self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = self.layer3(out)

        out1 = self.rdb1(out)
        out2 = self.rdb2(out1)
        out3 = self.rdb3(out2)
        out4 = self.rdb4(out3)
        concated = torch.cat((out1, out2, out3, out4), 1)
        out5 = out + self.oneone1(concated)

        out6 = self.rdb5(out5)
        out7 = self.rdb6(out6)
        out8 = self.rdb7(out7)
        out9 = self.rdb8(out8)
        concated = torch.cat((out6, out7, out8, out9), 1)
        out10 = out5 + self.oneone2(concated)

        out11 = self.rdb9(out10)
        out12 = self.rdb10(out11)
        out13 = self.rdb11(out12)
        out14 = self.rdb12(out13)
        concated = torch.cat((out11, out12, out13, out14), 1)
        out15 = out10 + self.oneone3(concated)

        out16 = self.rdb13(out15)
        out17 = self.rdb14(out16)
        out18 = self.rdb15(out17)
        out19 = self.rdb16(out18)
        concated = torch.cat((out16, out17, out18, out19), 1)
        out20 = out15 + self.oneone4(concated)

        out = self.layer7(out20)
        # out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_gd_rir_cbam3_recursive_relu(nn.Module):
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_gd_rir_cbam3_recursive_relu, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.rdb1 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb2 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb3 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb4 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone1 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb5 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb6 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb7 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb8 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone2 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb9 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb10 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb11 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb12 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone3 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb13 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb14 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb15 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb16 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone4 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(64, 16)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.cbam(out)
        out = self.layer3(out)

        out1 = self.rdb1(out)
        out2 = self.rdb2(out1)
        out3 = self.rdb3(out2)
        out4 = self.rdb4(out3)
        concated = torch.cat((out1, out2, out3, out4), 1)
        out5 = out + self.oneone1(concated)

        out6 = self.rdb5(out5)
        out7 = self.rdb6(out6)
        out8 = self.rdb7(out7)
        out9 = self.rdb8(out8)
        concated = torch.cat((out6, out7, out8, out9), 1)
        out10 = out5 + self.oneone2(concated)

        out11 = self.rdb9(out10)
        out12 = self.rdb10(out11)
        out13 = self.rdb11(out12)
        out14 = self.rdb12(out13)
        concated = torch.cat((out11, out12, out13, out14), 1)
        out15 = out10 + self.oneone3(concated)

        out16 = self.rdb13(out15)
        out17 = self.rdb14(out16)
        out18 = self.rdb15(out17)
        out19 = self.rdb16(out18)
        concated = torch.cat((out16, out17, out18, out19), 1)
        out20 = out15 + self.oneone4(concated)

        out = self.cbam(out20)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x


class Generator_one2many_gd_rir_cbam3_recursive_no_relu(nn.Module):
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_gd_rir_cbam3_recursive_no_relu, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        # self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.rdb1 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb2 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb3 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb4 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone1 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb5 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb6 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb7 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb8 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone2 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb9 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb10 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb11 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb12 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone3 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb13 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb14 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb15 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb16 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone4 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        # self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(64, 16)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = self.cbam(out)
        out = self.layer3(out)

        out1 = self.rdb1(out)
        out2 = self.rdb2(out1)
        out3 = self.rdb3(out2)
        out4 = self.rdb4(out3)
        concated = torch.cat((out1, out2, out3, out4), 1)
        out5 = out + self.oneone1(concated)

        out6 = self.rdb5(out5)
        out7 = self.rdb6(out6)
        out8 = self.rdb7(out7)
        out9 = self.rdb8(out8)
        concated = torch.cat((out6, out7, out8, out9), 1)
        out10 = out5 + self.oneone2(concated)

        out11 = self.rdb9(out10)
        out12 = self.rdb10(out11)
        out13 = self.rdb11(out12)
        out14 = self.rdb12(out13)
        concated = torch.cat((out11, out12, out13, out14), 1)
        out15 = out10 + self.oneone3(concated)

        out16 = self.rdb13(out15)
        out17 = self.rdb14(out16)
        out18 = self.rdb15(out17)
        out19 = self.rdb16(out18)
        concated = torch.cat((out16, out17, out18, out19), 1)
        out20 = out15 + self.oneone4(concated)

        out = self.cbam(out20)
        out = self.layer7(out)
        # out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_gd_rir_cbam3_non_recursive_relu(nn.Module):
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_gd_rir_cbam3_non_recursive_relu, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units
        self.nospatial = True

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.rdb1 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb2 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb3 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb4 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone1 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb5 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb6 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb7 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb8 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone2 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb9 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb10 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb11 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb12 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone3 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb13 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb14 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb15 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb16 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone4 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(64, 16, no_spatial=self.nospatial)
        self.cbam2 = CBAM(64, 16, no_spatial=self.nospatial)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.cbam1(out)
        out = self.layer3(out)

        out1 = self.rdb1(out)
        out2 = self.rdb2(out1)
        out3 = self.rdb3(out2)
        out4 = self.rdb4(out3)
        concated = torch.cat((out1, out2, out3, out4), 1)
        out5 = out + self.oneone1(concated)

        out6 = self.rdb5(out5)
        out7 = self.rdb6(out6)
        out8 = self.rdb7(out7)
        out9 = self.rdb8(out8)
        concated = torch.cat((out6, out7, out8, out9), 1)
        out10 = out5 + self.oneone2(concated)

        out11 = self.rdb9(out10)
        out12 = self.rdb10(out11)
        out13 = self.rdb11(out12)
        out14 = self.rdb12(out13)
        concated = torch.cat((out11, out12, out13, out14), 1)
        out15 = out10 + self.oneone3(concated)

        out16 = self.rdb13(out15)
        out17 = self.rdb14(out16)
        out18 = self.rdb15(out17)
        out19 = self.rdb16(out18)
        concated = torch.cat((out16, out17, out18, out19), 1)
        out20 = out15 + self.oneone4(concated)

        out = self.cbam2(out20)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_gd_rir_cbam3_non_recursive_no_relu(nn.Module):
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_gd_rir_cbam3_non_recursive_no_relu, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units
        self.nospatial = True

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        # self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.rdb1 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb2 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb3 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb4 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone1 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb5 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb6 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb7 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb8 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone2 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb9 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb10 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb11 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb12 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone3 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb13 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb14 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb15 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb16 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone4 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        # self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(64, 16, no_spatial=self.nospatial)
        self.cbam2 = CBAM(64, 16, no_spatial=self.nospatial)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = self.cbam1(out)
        out = self.layer3(out)

        out1 = self.rdb1(out)
        out2 = self.rdb2(out1)
        out3 = self.rdb3(out2)
        out4 = self.rdb4(out3)
        concated = torch.cat((out1, out2, out3, out4), 1)
        out5 = out + self.oneone1(concated)

        out6 = self.rdb5(out5)
        out7 = self.rdb6(out6)
        out8 = self.rdb7(out7)
        out9 = self.rdb8(out8)
        concated = torch.cat((out6, out7, out8, out9), 1)
        out10 = out5 + self.oneone2(concated)

        out11 = self.rdb9(out10)
        out12 = self.rdb10(out11)
        out13 = self.rdb11(out12)
        out14 = self.rdb12(out13)
        concated = torch.cat((out11, out12, out13, out14), 1)
        out15 = out10 + self.oneone3(concated)

        out16 = self.rdb13(out15)
        out17 = self.rdb14(out16)
        out18 = self.rdb15(out17)
        out19 = self.rdb16(out18)
        concated = torch.cat((out16, out17, out18, out19), 1)
        out20 = out15 + self.oneone4(concated)

        out = self.cbam2(out20)
        out = self.layer7(out)
        # out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_gd_rir_cbam4_recursive_relu(nn.Module):
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_gd_rir_cbam4_recursive_relu, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.rdb1 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb2 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb3 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb4 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone1 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb5 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb6 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb7 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb8 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone2 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb9 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb10 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb11 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb12 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone3 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb13 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb14 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb15 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb16 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone4 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(64, 16)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out1 = self.rdb1(out)
        out2 = self.rdb2(out1)
        out3 = self.rdb3(out2)
        out4 = self.rdb4(out3)
        concated = torch.cat((out1, out2, out3, out4), 1)
        out5 = out + self.oneone1(concated)

        out6 = self.rdb5(out5)
        out7 = self.rdb6(out6)
        out8 = self.rdb7(out7)
        out9 = self.rdb8(out8)
        concated = torch.cat((out6, out7, out8, out9), 1)
        out10 = out5 + self.oneone2(concated)

        out11 = self.rdb9(out10)
        out12 = self.rdb10(out11)
        out13 = self.rdb11(out12)
        out14 = self.rdb12(out13)
        concated = torch.cat((out11, out12, out13, out14), 1)
        out15 = out10 + self.oneone3(concated)

        out16 = self.rdb13(out15)
        out17 = self.rdb14(out16)
        out18 = self.rdb15(out17)
        out19 = self.rdb16(out18)
        concated = torch.cat((out16, out17, out18, out19), 1)
        out20 = out15 + self.oneone4(concated)

        out = self.layer7(out20)
        out = self.cbam(out)

        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x

class Generator_one2many_gd_rir_cbam4_recursive_no_relu(nn.Module):
    def __init__(self, input_channel, numforrg, numofrdb):
        super(Generator_one2many_gd_rir_cbam4_recursive_no_relu, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units

        self.layer1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        # self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.rdb1 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb2 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb3 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb4 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone1 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb5 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb6 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb7 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb8 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone2 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb9 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb10 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb11 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb12 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone3 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.rdb13 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb14 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb15 = RDB(64, nDenselayer=8, growthRate=64)
        self.rdb16 = RDB(64, nDenselayer=8, growthRate=64)
        self.oneone4 = nn.Conv2d(64 * self.numforrg, 64, kernel_size=1, stride=1, padding=0)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        # self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(64, 16)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = self.layer3(out)

        out1 = self.rdb1(out)
        out2 = self.rdb2(out1)
        out3 = self.rdb3(out2)
        out4 = self.rdb4(out3)
        concated = torch.cat((out1, out2, out3, out4), 1)
        out5 = out + self.oneone1(concated)

        out6 = self.rdb5(out5)
        out7 = self.rdb6(out6)
        out8 = self.rdb7(out7)
        out9 = self.rdb8(out8)
        concated = torch.cat((out6, out7, out8, out9), 1)
        out10 = out5 + self.oneone2(concated)

        out11 = self.rdb9(out10)
        out12 = self.rdb10(out11)
        out13 = self.rdb11(out12)
        out14 = self.rdb12(out13)
        concated = torch.cat((out11, out12, out13, out14), 1)
        out15 = out10 + self.oneone3(concated)

        out16 = self.rdb13(out15)
        out17 = self.rdb14(out16)
        out18 = self.rdb15(out17)
        out19 = self.rdb16(out18)
        concated = torch.cat((out16, out17, out18, out19), 1)
        out20 = out15 + self.oneone4(concated)

        out = self.layer7(out20)
        out = self.cbam(out)

        # out = self.layer8(out)
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