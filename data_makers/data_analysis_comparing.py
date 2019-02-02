import pandas
import matplotlib.pyplot as plt
import numpy as np
import os


def str_to_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False


# ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
# 두 실험의 iteration 에 따른 average 비교
# https://stickie.tistory.com/80  참고

# 실험 제목
exp_names = ('exp001', 'exp002_1')

# input 과 target 의 psnr 을 같이 보여줄 것인가
print_input_psnr = False

# 비교할 열의 index
column = 3

# 실험에서 eval 하는 iteration 주기
eval_period = 5000

# 실험 별 psnr 추이에서 처음 읽히는 값이 input_psnr 이다.
input_psnr = 0

# 실험 별 psnr 추이를 저장할 dictionary
data_lists = {}

for exp_name in exp_names:
    exp_dir = os.path.dirname(os.getcwd()) + '/exp/' + exp_name
    data = pandas.read_csv(exp_dir + '/' + exp_name + '_log.csv')

    data_list = data.iloc[:, column].values.tolist()

    data_list = [float(data) for data in data_list if str_to_float(data)]

    # 실험 별 psnr 추이에서 처음 읽히는 값이 input_psnr 이다.
    input_psnr = data_list[0]

    for _ in range(10):
        data_list.remove(min(data_list))

    data_lists[exp_name] = data_list

if print_input_psnr:
    # input psnr 을 10 칸정도 그려준다.
    init = [input_psnr] * 10
    data_lists['init'] = init

for items in data_lists.items():

    tt = np.arange(0, len(items[1]) * eval_period, eval_period)

    plt.plot(tt, items[1], label=items[0])

plt.legend(loc='upper left')
plt.show()
