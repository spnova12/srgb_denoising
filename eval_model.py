import numpy as np
import os
import cv2
import tqdm
from utils import *
from utils.eval import EvalModule, LogCSV, psnr
from models.hevcNet import Generator_one2many_RDB_no_tanh


def weight_loader(saved_weight_dir, net):
    # optionally resume from a checkpoint
    if os.path.isfile(saved_weight_dir):
        print("===> loading checkpoint '{}'".format(saved_weight_dir))

        checkpoint = torch.load(saved_weight_dir)

        current_epoch = checkpoint['epoch']
        print("current_epoch :", current_epoch)

        net.load_state_dict(checkpoint['state_dict_G'])  # 안되면 state_dict_G 로 바꾸기.
    else:
        print("===> no checkpoint found at '{}'".format(saved_weight_dir))


# test data set (Noisy, Target 순서대로)
test_folder_dir = [('/home/lab/works/datasets/ssd2/ntire/validation/set1/Noisy',
                    '/home/lab/works/datasets/ssd2/ntire/validation/set1/GroundTruth'),
                   # ('/home/lab/works/datasets/ssd2/flickr/validation/GroundTruth',
                   #  '/home/lab/works/datasets/ssd2/flickr/validation/GroundTruth'),
                   # ('/home/lab/works/datasets/ssd2/ntire/validation/set2/GroundTruth',
                   #  '/home/lab/works/datasets/ssd2/ntire/validation/set2/GroundTruth')
                   ]

# 생성된 영상을 저장하려면 경로 지정해주기.
test_output_folder_dir = None

# 생성 영상들의 psnr이 저장될 csv 파일 경로.
csv_dir = "first_submit_psnr.csv"

img_loader = load_BGR

#checkpoint_dir = '/home/lab/works/users/kdw/ntire_srgb_dn/exp/exp001/checkpoint.pkl'

checkpoint_dir = '/home/lab/works/users/kdw/ntire_srgb_dn/exp/exp006_retry_no_tanh_0to1/saved_model/checkpoint_best.pkl'

device = torch.device('cuda:0')
net = Generator_one2many_RDB_no_tanh(input_channel=3).to(device)
weight_loader(checkpoint_dir, net)

eval = EvalModule(net)

# 스케일을 변형해 주고 싶을때.
eval.set_transforms(Compose([Color0_255to0_1(), ToTensor()]), Compose([ToImage(), Color0_1to0_255()]))

img_dir_list_for_logger = []
test_input_img_dirs = []
test_target_img_dirs = []

for test_input_folder_dir, test_target_folder_dir in test_folder_dir:
    img_dir_list_for_logger += sorted(listdir(test_input_folder_dir))
    test_input_img_dirs += [join(test_input_folder_dir, x) for x in sorted(listdir(test_input_folder_dir))]
    test_target_img_dirs += [join(test_target_folder_dir, x) for x in sorted(listdir(test_target_folder_dir))]

# target 과 noisy 와의 psnr 들.
input_psnrs_list = []
# 불러온 checkpoint 혹은 초기 모델로 eval 한 결과의 target 과의 psnr 들.
output_psnrs_list = []
for test_input_img_dir, test_target_img_dir in tqdm.tqdm(zip(test_input_img_dirs, test_target_img_dirs)):
    test_input_img = img_loader(test_input_img_dir)
    test_target_img = img_loader(test_target_img_dir)

    if test_output_folder_dir:
        cv2.imwrite(test_output_folder_dir + '/'
                    + str(os.path.basename(test_input_img_dir).split(".")[0])
                    + '_input'
                    + '.PNG', test_input_img)

        cv2.imwrite(test_output_folder_dir + '/'
                    + str(os.path.basename(test_input_img_dir).split(".")[0])
                    + '_target'
                    + '.PNG', test_target_img)

    test_output_img = eval.recon(test_input_img)

    input_psnr = psnr(test_target_img, test_input_img)
    ouput_psnr = psnr(test_target_img, test_output_img)

    input_psnrs_list.append(input_psnr)
    output_psnrs_list.append(ouput_psnr)

logger = LogCSV(csv_dir)

# logger 의 header 를 입력해준다.
logger.make_head(header=['epoch', 'iter_count', 'best_iter', 'average'] + img_dir_list_for_logger)

# target 과 noisy 와의 psnr 로그(csv 파일)에 기록.
logger(['', '', ''] + [np.mean(input_psnrs_list)] + input_psnrs_list)

# 불러온 checkpoint 혹은 초기 모델로 eval 한 결과의 target 과의 psnr 들 로그(csv 파일)에 기록.
logger(['', '', ''] + [np.mean(output_psnrs_list)] + output_psnrs_list)