# entire dataset Ground truth 의 가장자리가 이상하여 잘라주었다.

import os
from os import listdir
from os.path import join
from utils.utils import load_BGR, make_dirs
import cv2
import tqdm


parent_folder_dir = '/home/lab/works/datasets/ssd2/ntire/train/origin_outer_clean'


folder_dirs = [join(parent_folder_dir, x) for x in sorted(listdir(parent_folder_dir))]

for folder_dir in folder_dirs:

    img_dirs = [join(folder_dir, x) for x in sorted(listdir(folder_dir))]

    print(folder_dir)

    for i, img_dir in tqdm.tqdm(enumerate(img_dirs)):
        img = load_BGR(img_dir)

        # 가장자리의 8 pixel 을 모두 잘라주었다.
        img = img[8:-8, 8:-8]

        basename = os.path.basename(img_dir).split('.')[0]
        cv2.imwrite(img_dir, img)
