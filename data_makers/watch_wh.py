
import os
from os import listdir
from os.path import join
from utils.utils import load_BGR, make_dirs
import cv2
import tqdm


img_folder_dir = '/home/lab/works/datasets/ssd2/div2k/train/downsampled/GroundTruth'

img_dirs = [join(img_folder_dir, x) for x in sorted(listdir(img_folder_dir))]

h_min = 10000000000000
h_min_name = None

w_min = 10000000000000
w_min_name = None

for i, img_dir in enumerate(tqdm.tqdm(img_dirs)):
    img = load_BGR(img_dir)
    # print(i, img.shape)

    h = img.shape[0]
    w = img.shape[1]

    if h < h_min:
        h_min = h
        h_min_name = os.path.basename(img_dir)

    if w < w_min:
        w_min = w
        w_min_name = os.path.basename(img_dir)

print('h_min', h_min, h_min_name)
print('w_min', w_min, w_min_name)


