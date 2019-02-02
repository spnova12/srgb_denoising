
import os
from os import listdir
from os.path import join
from utils.utils import load_BGR, make_dirs
import cv2
import tqdm

real_origin_folder_dir = '/home/lab/works/datasets/ssd2/div2k/train/origin'

real_origin_basename = os.path.dirname(real_origin_folder_dir)
real_splited_folder_dir = real_origin_basename + '/splited_overlaped/'

# 가로세로 열등분 하자.
patch_h_count = 6
patch_w_count = 6

big_img_folder_dirs = [join(real_origin_folder_dir, x) for x in sorted(listdir(real_origin_folder_dir))]

print(big_img_folder_dirs)

for big_img_folder_dir in big_img_folder_dirs:

    img_dirs = [join(big_img_folder_dir, x) for x in sorted(listdir(big_img_folder_dir))]
    folder_basename = os.path.basename(big_img_folder_dir)

    img_folder_dir_new = make_dirs(real_splited_folder_dir + folder_basename + '/')

    patch_count = 0

    for i, img_dir in enumerate(tqdm.tqdm(img_dirs)):
        basename = os.path.basename(img_dir).split('.')[0]

        img = load_BGR(img_dir)
        # print(i, img.shape)

        h = img.shape[0]
        w = img.shape[1]

        patch_h = h // patch_h_count
        patch_w = w // patch_w_count

        patch_h_count_ov = patch_h_count * 2 - 1
        patch_w_count_ov = patch_w_count * 2 - 1

        for i_h in range(patch_h_count_ov):
            for i_w in range(patch_w_count_ov):

                top = int(patch_h * i_h * 0.5)
                bottom = top + patch_h
                if i_h == patch_h_count_ov - 1:
                    bottom = h

                left = int(patch_w * i_w * 0.5)
                right = left + patch_w
                if i_w == patch_w_count_ov - 1:
                    right = w

                img_patch = img[top: bottom, left: right]

                img_patch_name = img_folder_dir_new + str(basename) + '_' + str(i_h) + '_' + str(i_w) + '.PNG'
                cv2.imwrite(img_patch_name, img_patch)

                patch_count += 1
