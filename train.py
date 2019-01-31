from pathlib import Path
import time
import argparse

import os
import shutil
import copy

from os import listdir
from os.path import join

from utils import *
from utils.eval import EvalModule, LogCSV, psnr

from models.hevcNet import Generator_one2many_RDB_no_tanh
from models.subNets import weights_init

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

# cudnn.benchmark allows you to enable the inbuilt cudnn auto-tuner
# to find the best algorithm to use for your hardware.
cudnn.benchmark = True


class TrainModule(object):
    def __init__(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # 사용할 gpu 번호.
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        # training data set
        self.train_paired_folder_dirs = {('/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_non_overlaped/Noisy',
                                     '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_non_overlaped/GroundTruth'): 2,
                                    ('/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_non_overlaped/GroundTruth',
                                     '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/train/real_splited_non_overlaped/GroundTruth'): 1,
                                    }

        # test data set
        test_input_folder_dir = '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/validation/Noisy'
        test_target_folder_dir = '/home/lab/works/datasets/ssd2/NTIRE_challenge_DB/sRGB/validation/GroundTruth'

        # 실험 이름.
        exp_name = 'exp001'

        # 총 몇 epoch 돌릴것인가.
        self.total_epoch = 20

        self.lr_decay_period = None

        self.init_learning_rate = 0.0001

        # eval 할 iteration 주기.
        self.eval_period = 60

        self.batch_size = 16

        self.random_crop_size = 48

        self.img_loader = load_BGR

        # ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

        # sample image 가 저장될 경로.
        self.test_output_folder_dir = make_dirs(f'exp/{exp_name}/samples')

        # checkpoint 저장할 경로.
        self.load_checkpoint_dir = make_dirs(f'exp/{exp_name}') + '/' + 'checkpoint.pkl'
        self.saved_checkpoint_dir = make_dirs(f'exp/{exp_name}') + '/' + 'checkpoint.pkl'

        # log csv 파일이 저장될 경로
        self.log_dir = make_dirs(f'exp/{exp_name}')

        self.start_epoch = 1
        self.iter_count = 0
        self.best_psnr = 0
        self.best_iter = 0

        # ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

        self.device = torch.device('cuda:0')

        if not torch.cuda.is_available():
            raise Exception("No GPU found")
        else:
            print("===> GPU on")

        self.net = Generator_one2many_RDB_no_tanh(input_channel=3).to(self.device)
        self.net.apply(weights_init)

        print('===> Number of params: {}'.format(
            sum([p.data.nelement() for p in self.net.parameters()])))

        # optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.init_learning_rate, betas=(0.5, 0.999))

        # criterion
        self.mse = nn.MSELoss().to(self.device)

        # Load pre-trained weights
        self.weight_loader()

        # Make eval module based on net
        self.eval = EvalModule(self.net)

        self.test_input_img_dirs = [join(test_input_folder_dir, x) for x in sorted(listdir(test_input_folder_dir))]
        self.test_target_img_dirs = [join(test_target_folder_dir, x) for x in sorted(listdir(test_target_folder_dir))]

        self.logger = LogCSV(log_dir=self.log_dir + f"/{exp_name}_log.csv")

        # logger 의 header 를 입력해준다.
        self.logger.make_head(header=['epoch', 'iter_count', 'best_iter', 'average'] + sorted(listdir(test_input_folder_dir)))

        # ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

        print('===> first eval validation set')
        # 처음 validation 을 구해본다.

        # target 과 noisy 와의 psnr 들.
        input_psnrs_list = []

        # 불러온 checkpoint 혹은 초기 모델로 eval 한 결과의 target 과의 psnr 들.
        output_psnrs_list = []

        for test_input_img_dir, test_target_img_dir in zip(self.test_input_img_dirs, self.test_target_img_dirs):
            test_input_img = self.img_loader(test_input_img_dir)
            test_target_img = self.img_loader(test_target_img_dir)

            test_output_img = self.eval.recon(test_input_img)

            input_psnr = psnr(test_target_img, test_input_img)
            ouput_psnr = psnr(test_target_img, test_output_img)

            input_psnrs_list.append(input_psnr)
            output_psnrs_list.append(ouput_psnr)

        # target 과 noisy 와의 psnr 로그(csv 파일)에 기록.
        self.logger(['', '', ''] + [np.mean(input_psnrs_list)] + input_psnrs_list)

        # 불러온 checkpoint 혹은 초기 모델로 eval 한 결과의 target 과의 psnr 들 로그(csv 파일)에 기록.
        self.logger([0, self.iter_count, ''] + [np.mean(output_psnrs_list)] + output_psnrs_list)

        # ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

        print('===> Make train data loader')

        # Loading training data sets
        self.train_set = None
        self.train_data_loader = None
        self.init_training_dataset_loader()

    def init_training_dataset_loader(self):
        self.train_set = UniformedPairedImageDataSet(self.train_paired_folder_dirs, self.img_loader,
                                                     Compose([
                                                         RandomCrop(self.random_crop_size),
                                                         ToTensor()
                                                     ])
                                                     )

        self.train_data_loader = DataLoader(dataset=self.train_set,
                                            num_workers=4,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            )

    def weight_loader(self):
        if os.path.isfile(self.load_checkpoint_dir):
            print("===> loading checkpoint '{}'".format(self.load_checkpoint_dir))

            checkpoint = torch.load(self.load_checkpoint_dir)

            self.start_epoch = checkpoint['epoch']
            self.iter_count = checkpoint['iter_count']
            self.best_psnr = checkpoint['best_psnr']

            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("===> no checkpoint found at '{}'".format(self.load_checkpoint_dir))

    def weight_saver(self, current_epoch, filename='checkpoint.pkl'):
        state = {
                'epoch': current_epoch,
                'iter_count': self.iter_count,
                'best_psnr': self.best_psnr,
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, filename)

    def adjust_learning_rate(self, current_epoch):
        # Sets the learning rate to the initial learning rate decayed by 10 every n epochs
        if self.lr_decay_period:
            lr = self.init_learning_rate * (0.1 ** (current_epoch // self.lr_decay_period))
            print('learning rate : ', lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train(self):
        for current_epoch in range(self.start_epoch, self.total_epoch + 1):
            print(f'epoch {current_epoch} start -------------------------------------------------------------')
            self.adjust_learning_rate(current_epoch)

            for i, batch in enumerate(self.train_data_loader, 1):
                print(f'><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><')
                self.net.train()
                input_img, target_img = batch[0].to(self.device), batch[1].to(self.device)
                output_img = self.net(input_img)
                loss = self.mse(output_img, target_img)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 20 == 0:
                    print(f"===> Epoch[{current_epoch}/{self.total_epoch}]({i}/{len(self.train_data_loader)}): Loss: {loss.item():.6f}")

                # ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

                self.iter_count += 1

                # eval
                if self.iter_count % self.eval_period == 0:
                    print('eval validation set')

                    psnrs_list = []
                    for test_input_img_dir, test_target_img_dir in zip(self.test_input_img_dirs, self.test_target_img_dirs):
                        test_input_img = self.img_loader(test_input_img_dir)
                        test_target_img = self.img_loader(test_target_img_dir)

                        test_output_img = self.eval.recon(test_input_img)

                        ouput_psnr = psnr(test_target_img, test_output_img)
                        psnrs_list.append(ouput_psnr)

                        # write image
                        cv2.imwrite(self.test_output_folder_dir + '/'
                                    + str(os.path.basename(test_input_img_dir).split(".")[0])
                                    + '_' + str(self.iter_count).zfill(10)
                                    + '_' + str(current_epoch).zfill(3)
                                    + '.png', test_output_img)

                    psnrs_list_mean = np.mean(psnrs_list)

                    if psnrs_list_mean > self.best_psnr:
                        self.best_psnr = psnrs_list_mean
                        self.best_iter = self.iter_count
                        self.weight_saver(current_epoch, filename=self.saved_checkpoint_dir)

                    print(f'current_psnr(iter:{self.iter_count}) : {psnrs_list_mean}\n'
                          f'besr_psnr(iter:{self.best_iter}) : {self.best_psnr}')

                    self.logger([current_epoch, self.iter_count, self.best_iter, psnrs_list_mean] + psnrs_list)

            # 한 epoch 이 끝나면 data loader 을 다시 생성해서 좀 더 뼛 속까지 섞어준다.
            self.init_training_dataset_loader()


if __name__ == '__main__':
    train_module = TrainModule()
    train_module.train()






