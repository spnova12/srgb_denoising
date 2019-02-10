# challenge 에서 제공받은 srgb 이미지가 너무 크니까 좀 작게 쪼개주자.
# 일단 겹치지 않게 w, h 각각 10개씩으로 쪼개주었다.

# real 이미지들(challenge 에서 제공받은 데이터 셋) 과
# 그 ground truth 에 합성노이즈를 추가해준 이미들이 들어있는 폴더 전체에 대해 스크립트가 작동한다.


import os
from os import listdir
from os.path import join
from utils.utils import load_BGR, make_dirs
import cv2
import tqdm

real_origin_folder_dir = '/home/lab/works/datasets/ssd2/ntire/train/origin_outer_clean'

real_origin_dirname = os.path.dirname(real_origin_folder_dir)
real_splited_folder_dir = real_origin_dirname + '/real_splited_none_overlaped/'

# 가로세로 열등분 하자.
patch_h_count = 10
patch_w_count = 10

big_img_folder_dirs = [join(real_origin_folder_dir, x) for x in sorted(listdir(real_origin_folder_dir))]

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

        for i_h in range(patch_h_count):
            for i_w in range(patch_w_count):
                top = patch_h * i_h
                bottom = top + patch_h

                if i_h == patch_h_count - 1:
                    bottom = h

                left = patch_w * i_w
                right = left + patch_w

                if i_w == patch_w_count - 1:
                    right = w

                img_patch = img[top: bottom, left: right]

                img_patch_name = img_folder_dir_new + str(basename) + '_' + str(i_h) + '_' + str(i_w) + '.PNG'
                cv2.imwrite(img_patch_name, img_patch)

                patch_count += 1

        # print(patch_count)

