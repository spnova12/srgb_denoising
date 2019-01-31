from os import listdir
from os.path import join
import numpy as np
import torch.utils.data as data
import random

import copy


# ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
class PairedImageDataSet(data.Dataset):
    """
    https://www.youtube.com/watch?v=zN49HdDxHi8 참고
    가장 기본적인 방식.
    """
    def __init__(self, input_folder_dir, target_folder_dir, loader, transform=None):
        self.input_img_dirs = [join(input_folder_dir, x) for x in sorted(listdir(input_folder_dir))]
        self.target_img_dirs = [join(target_folder_dir, x) for x in sorted(listdir(target_folder_dir))]
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        # default tensor type is (torch.float32), 때문에 torch 로 학습을 하기위해 float32 로 형변환을 해준다.
        input_img = self.loader(self.input_img_dirs[index]).astype(np.float32)
        target_img = self.loader(self.target_img_dirs[index]).astype(np.float32)

        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)
        return input_img, target_img

    def __len__(self):
        return len(self.input_img_dirs)


# ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

def uniformed_img_dirs_by_ratio(paired_to_ratio):
    print('# 설정한 비율 :', list(paired_to_ratio.values()))
    paired_to_paired_imgs = {}
    paired_to_len = {}

    max_len = 0
    max_key = None

    # 폴더별 이미지들을 읽어서 (input, target) 으로 묶어준다.
    for key in paired_to_ratio.keys():
        input_folder_dir = key[0]
        target_folder_dir = key[1]
        img_dirs_pair = [[join(input_folder_dir, x), join(target_folder_dir, y)]
                         for x, y in zip(sorted(listdir(input_folder_dir)), sorted(listdir(target_folder_dir)))]

        paired_to_paired_imgs[key] = img_dirs_pair

        if max_len < len(img_dirs_pair):
            max_key = key
            max_len = len(img_dirs_pair)

    # 가장 큰 폴더를 기준으로 설정한 비율에 맞게 폴더마다 이미지 장 수를 조절한다.
    paired_to_len[max_key] = max_len
    for key in paired_to_ratio.keys():
        if key != max_key:
            # 설정한 비울에 맞게 가장 큰 폴더를 기준으로 장 수 개산.
            paired_to_len[key] = (max_len * paired_to_ratio[key]) / paired_to_ratio[max_key]

            # 다양성을 위해 한번 섞어준다.
            random.shuffle(paired_to_paired_imgs[key])

            # 폴더 별 장 수를 조절해준다.
            num_to_mul = int(paired_to_len[key] // len(paired_to_paired_imgs[key]))
            remainder = int(paired_to_len[key] % len(paired_to_paired_imgs[key]))
            paired_to_paired_imgs[key] = paired_to_paired_imgs[key] * num_to_mul + paired_to_paired_imgs[key][:remainder]

    print('# 비율에 맞게 조정된 실제 사이즈 :', list(paired_to_len.values()))

    # values 들을 하나의 list 로 묶어준다. 그리고 shuffle 한번 먹여주었다.
    img_dirs = list(paired_to_paired_imgs.values())
    img_dirs = sum(img_dirs, [])
    random.shuffle(img_dirs)

    print('# 다 합쳐진 최종 데이터 셋 사이즈 :', len(img_dirs))

    return img_dirs


class UniformedPairedImageDataSet(data.Dataset):
    """
    여러 데이터셋을 지정한 비율대로 섞어서 사용할 수 있다.
    가장 크기가 큰 데이터셋을 기준으로 이미지 수를 조정해준다.
    """
    def __init__(self, paired_to_ratio, loader, transform=None):
        self.img_dirs = uniformed_img_dirs_by_ratio(paired_to_ratio)

        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        # default tensor type is (torch.float32), 때문에 torch 로 학습을 하기위해 float32 로 형변환을 해준다.
        input_img = self.loader(self.img_dirs[index][0]).astype(np.float32)
        target_img = self.loader(self.img_dirs[index][1]).astype(np.float32)

        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)
        return input_img, target_img

    def __len__(self):
        return len(self.img_dirs)


if __name__ == '__main__':
    folder_dirs = {('/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_non_overlaped/Noisy',
                    '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_non_overlaped/GroundTruth'): 2,
                   ('/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_non_overlaped/GroundTruth',
                    '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_non_overlaped/GroundTruth'): 1,
                   ('/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/validation/Noisy',
                    '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/validation/GroundTruth'): 4
                   }

    uniformed_img_dirs_by_ratio(folder_dirs)


