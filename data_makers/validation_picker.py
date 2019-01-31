# 학습 데이터의 일부분을 빼서 Validation set 을 만드는 스크립트이다.

import os
from os import listdir
from os.path import join
from utils.utils import load_BGR, make_dirs
from random import *
import shutil


train_dirs = '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_none_overlaped'

validation_dirs = '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/validation'


# 폴더 안의 이미지 수 확인을 위해 임의의 폴더의 이미지 개수 확인.
train_folder_dirs = [join(train_dirs, x) for x in sorted(listdir(train_dirs))]
temp = [join(train_folder_dirs[0], x) for x in sorted(listdir(train_folder_dirs[0]))]
total_img_count = len(temp)
print(total_img_count)

# 0 부터 이미지의 최대 개수 사이의 숫자 중 임의의 숫자 100장 뽑음.
picked_img_from_trainset = set()
while len(picked_img_from_trainset) < 100:
    picked_img_from_trainset.add(randint(0, total_img_count-1))

for train_folder_dir in train_folder_dirs:
    print('-----------------------------------')
    folder_basename = os.path.basename(train_folder_dir)
    img_dirs = [join(train_folder_dir, x) for x in sorted(listdir(train_folder_dir))]

    img_folder_dir_new = make_dirs(validation_dirs + '/' + folder_basename)

    for i in picked_img_from_trainset:
        fullname = img_dirs[i]
        basename = os.path.basename(img_dirs[i])
        changedname = img_folder_dir_new + '/' + basename
        print(fullname)
        print(changedname)
        shutil.move(fullname, changedname)





#
# # 학습 데이터셋의 주소
# train_gt_dir = '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_Non-overlaped/GroundTruth'
# train_noisy_dir = '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_Non-overlaped/Noisy'
#
# # 새로 만들 Validation 이미지들의 저장될 주소.
# validation_gt_dir = make_dirs('/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/validation/GroundTruth')
# validation_noisy_dir = make_dirs('/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/validation/Noisy')
#
# train_gt_img_dirs = [join(train_gt_dir, x) for x in sorted(listdir(train_gt_dir))]
# train_noisy_img_dirs = [join(train_noisy_dir, x) for x in sorted(listdir(train_noisy_dir))]
#
# # 생성할 validation 이미지들의 개수
#
# picked_img_from_trainset = set()
#
# while len(picked_img_from_trainset) < 50:
#     picked_img_from_trainset.add(randint(0, len(train_gt_img_dirs)-1))
#
# print(picked_img_from_trainset)
#
# for i in picked_img_from_trainset:
#     # gt 옮겨주기.
#     fullname = train_gt_img_dirs[i]
#     basename = os.path.basename(train_gt_img_dirs[i])
#     changedname = validation_gt_dir + '/' + basename
#     shutil.move(fullname, changedname)
#
#     # noisy 옮겨주기
#     fullname = train_noisy_img_dirs[i]
#     basename = os.path.basename(train_noisy_img_dirs[i])
#     changedname = validation_noisy_dir + '/' + basename
#     shutil.move(fullname, changedname)
#
#








