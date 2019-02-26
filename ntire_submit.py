import numpy as np
import matplotlib.pyplot as plt
import cv2
from models import *
import os
import torch

from torch.utils.data import DataLoader
import scipy.io as sio
import time

from models import rdn
from utils import *


def show_gray_image_with_pixelvalues(imgpath):
    #gray image
    img = cv2.imread(imgpath)

    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = 'gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

def show_color_image_with_pixelvalues(imgpath):
    #color image
    rgb_img = cv2.imread(imgpath)
    channels = ['red channel', 'green channel', 'blue channel']

    rgb_img = rgb_img.transpose((2, 1, 0))      #CxWxH 로 바꿈
    print('rgb_img', rgb_img.shape)

    fig = plt.figure(figsize=(36, 36))
    for idx in np.arange(rgb_img.shape[0]):
        ax = fig.add_subplot(1, 3, idx + 1)
        img = rgb_img[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(channels[idx])
        width, height = img.shape
        thresh = img.max()/2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y], 2) if img[x][y] != 0 else 0
                ax.annotate(str(val), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if img[x][y] < thresh else 'black')



if __name__ == '__main__':
    # x = np.arange(6).reshape((2, 3))
    # print(x)
    # y = np.transpose(x, (1, 0))
    # print(y)

    if False:
        print('*****************open image test**********************')
        imgpath = '/home/lab/works/datasets/ssd2/NTIREchallengeDB/RawRGB/GroundTruth/0001_GT_RAW_010.png'
        #show_color_image_with_pixelvalues(imgpath)

        rgbimg = cv2.imread(imgpath)
        print(rgbimg.shape)
        rgbimg = cv2.resize(rgbimg, None, fx=0.25, fy=0.25)
        # cv2.imshow('image', rgbimg)
        # cv2.waitKey(0)
    if True:
        print('********************ValidationNoisyBlocksSrgb******************')

        mat_dir = '/home/lab/works/datasets/ssd2/ntire/validation/ValidationNoisyBlocksSrgb.mat'
        mat_file = sio.loadmat(mat_dir)

        # print(mat_file)

        for dictval in mat_file:
            print(dictval)

        valnoisyblock = mat_file['ValidationNoisyBlocksSrgb']

        print(valnoisyblock)
        print('valnoisyblock.shape', valnoisyblock.shape)

        # imgslist = []

        transform = transforms.Compose([
            # Color0_255to0_1(),
            transforms.ToTensor()
        ])

        revtransform = transforms.Compose([
            transforms.ToImage(),
            # Color0_1to0_255()
        ])

        expname = 'exp014_rdb_updown'
        modelpath = f'/home/lab/works/users/cjr/ntire_project/srgb_denoising/exp/{expname}/checkpoint.pkl'
        input_channel = 3
        device = torch.device('cuda:0')

        netG = rdn.RDN(input_channel).to(device)

        resultNP = np.ones(valnoisyblock.shape)
        print('resultNP.shape', resultNP.shape)

        submitpath = f'/home/lab/works/users/cjr/benchmarks/NTIRE/sRGB/submit/{expname}'
        make_dirs(submitpath)
        imgsavepath = submitpath + '/imgs'
        make_dirs(imgsavepath)
        resultpath = submitpath + '/result'
        make_dirs(resultpath)

        checkpoint = torch.load(modelpath)

        netG.load_state_dict(checkpoint['state_dict'])

        with torch.no_grad():

            netG.eval()
            starttime = time.time()
            for imgidx in range(valnoisyblock.shape[0]):
                for patchidx in range(valnoisyblock.shape[1]):
                    img = valnoisyblock[imgidx][patchidx]   # img shape (256, 256, 3)
                    # print('img shape', img)

                    # cv2.imwrite(f'{imgsavepath}/{imgidx}_{patchidx}_input.png', img)

                    input = transform(img)
                    input = input.float().unsqueeze_(0).to(device)

                    # print('input shape', input)
                    # cv2.imshow('img', img)
                    # cv2.waitKey(0)

                    # print("===> loading checkpoint '{}'".format(modelpath))



                    output = netG(input)
                    # print('output shape', output.shape)

                    outimg = revtransform(output.detach())#.cpu())

                    if True:    #int형으로 변환
                        outimg = np.around(outimg)
                        outimg = outimg.clip(0, 255)
                        outimg = outimg.astype(np.uint8)

                    # print('outimg', outimg)
                    resultNP[imgidx][patchidx] = outimg
                    # cv2.imshow('outimg', outimg)
                    # cv2.waitKey(0)
                    print('img shaved at', f'{imgsavepath}/{imgidx}_{patchidx}_output.png')
                    cv2.imwrite(f'{imgsavepath}/{imgidx}_{patchidx}_output.png', outimg)

                    # print('mse:', ((img - outimg) ** 2).mean())



                # print('')
        endtime = time.time()
        elapsedTime = endtime - starttime
        print('ended', elapsedTime)
        print('number of pixels', valnoisyblock.shape[0] * valnoisyblock.shape[1] * valnoisyblock.shape[2] * valnoisyblock.shape[3])

        sio.savemat(f'{resultpath}/results', dict([('results', resultNP)]))
    if False:
        print('********************ValidationNoisyBlocksSrgb******************')

        mat_dir = '/home/lab/works/datasets/ssd2/NTIREchallengeDB/sRGB/ValidationNoisyBlocksSrgb.mat'
        mat_file = sio.loadmat(mat_dir)

        # print(mat_file)

        for dictval in mat_file:
            print(dictval)

        valnoisyblock = mat_file['ValidationNoisyBlocksSrgb']

        print(valnoisyblock)
        print('valnoisyblock.shape', valnoisyblock.shape)

        print('********************Result mat file******************')

        mat_dir = '/home/lab/works/users/cjr/hevc_reconnet_2018/submit/submit/results.mat'
        mat_file = sio.loadmat(mat_dir)

        # print(mat_file)

        for dictval in mat_file:
            print(dictval)

        valnoisyblock = mat_file['results']

        print(valnoisyblock)
        print('valnoisyblock.shape', valnoisyblock.shape)