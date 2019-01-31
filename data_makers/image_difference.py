# 차영상을 구해볼때 사용한다.

import cv2
from utils.utils import load_BGR, make_dirs
import numpy as np

gt_dir = 'gt.PNG'
noise_dir = 'synthetic_noisy2.png'

gt_img = load_BGR(gt_dir).astype(np.float32)
noise_img = load_BGR(noise_dir).astype(np.float32)

diff_img = noise_img - gt_img

diff_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite('dif_synthetic.PNG', diff_img)

