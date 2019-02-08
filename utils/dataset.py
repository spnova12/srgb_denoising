from os import listdir
from os.path import join
import os
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

class CatImgDirsByRatio(object):
    def __init__(self, paired_to_ratio):
        self.paired_to_ratio = paired_to_ratio
        self.paired_to_paired_imgs = {}

        self.max_len = 0
        self.max_key = None

        # 폴더별 이미지들을 읽어서 (input, target) 으로 묶어준다.
        for key in paired_to_ratio.keys():

            input_folder_dir = key[0]
            target_folder_dir = key[1]

            if paired_to_ratio[key] != 0:
                img_dirs_pair = [[join(input_folder_dir, x), join(target_folder_dir, y)]
                                 for x, y in zip(sorted(listdir(input_folder_dir)), sorted(listdir(target_folder_dir)))]
            else:
                img_dirs_pair = []

            random.shuffle(img_dirs_pair)
            self.paired_to_paired_imgs[key] = img_dirs_pair

            if self.max_len < len(img_dirs_pair):
                self.max_key = key
                self.max_len = len(img_dirs_pair)

    def get_dirs(self):
        # 가장 큰 폴더를 기준으로 설정한 비율에 맞게 폴더마다 이미지 장 수를 조절한다.
        paired_to_paired_imgs = {}

        paired_to_paired_imgs[self.max_key] = self.paired_to_paired_imgs[self.max_key]

        for key in self.paired_to_ratio.keys():
            if key != self.max_key:
                # 설정한 비울에 맞게 가장 큰 폴더를 기준으로 장 수 개산.
                paired_len = (self.max_len * self.paired_to_ratio[key]) / self.paired_to_ratio[self.max_key]

                # 폴더 별 장 수를 조절해준다.
                if self.paired_to_ratio[key] != 0:
                    num_to_mul = int(paired_len // len(self.paired_to_paired_imgs[key]))
                    remainder = int(paired_len % len(self.paired_to_paired_imgs[key]))
                    paired_to_paired_imgs[key] = self.paired_to_paired_imgs[key] * num_to_mul + self.paired_to_paired_imgs[key][:remainder]

                    # 다음 함수 호출때는 새로운 부분부터 remainder 을 할당하기 위해 리스트를 뒤집어준다.
                    self.paired_to_paired_imgs[key] = self.paired_to_paired_imgs[key][remainder:] + self.paired_to_paired_imgs[key][:remainder]

        for key in paired_to_paired_imgs.keys():
            print(f"{int(len(paired_to_paired_imgs[key]))} image pairs from {key[0]}")

        # values 들을 하나의 list 로 묶어준다.
        img_dirs = list(paired_to_paired_imgs.values())
        img_dirs = sum(img_dirs, [])

        # 그리고 shuffle 한번 먹여주었다.
        random.shuffle(img_dirs)

        print('tatal image pairs :', len(img_dirs))

        return img_dirs


class UniformedPairedImageDataSet(data.Dataset):
    """
    여러 데이터셋을 지정한 비율대로 섞어서 사용할 수 있다.
    가장 크기가 큰 데이터셋을 기준으로 이미지 수를 조정해준다.
    """
    def __init__(self, img_dirs, loader, transform=None):
        """
        :param img_dirs: [(input, target), (input, target) ... (input, target)]
        """
        self.img_dirs = img_dirs

        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        # default tensor type is (torch.float32), 때문에 torch 로 학습을 하기위해 float32 로 형변환을 해준다.
        input_img = self.loader(self.img_dirs[index][0])
        target_img = self.loader(self.img_dirs[index][1])

        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)
        return input_img.float(), target_img.float()

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



