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

# 실험 제목.
exp_names = ('exp001', 'exp012') # 'exp011_1', 'exp011_2', 'exp011_3', 'exp011_4',

# input 과 target 의 psnr 을 같이 보여줄 것인가 말 것인가 선택.
print_input_psnr = False

title = 'my ntire validation + o2o'
# 비교할 열의 index 리스트.
#column_range = list(range(4, 134))
column_range = list(range(4, 104))
#column_range = list(range(104, 124))
#column_range = list(range(124, 134))

column_list = column_range  # [4,5,6]

# 실험에서 eval 하는 iteration 주기
eval_period = 5000

# 실험 별 psnr 추이에서 처음 읽히는 값이 input_psnr 이다.
input_psnr = 0

# 실험 별 psnr 추이를 저장할 dictionary
data_lists = {}

for exp_name in exp_names:
    exp_dir = os.path.dirname(os.getcwd()) + '/exp/' + exp_name

    # read csv file.
    data = pandas.read_csv(exp_dir + '/' + exp_name + '_log.csv')

    # 원하는 열들을 추출한다.
    d_l = []
    for column in column_list:
        temp = data.iloc[:, column].values.tolist()
        d_l.append([float(data) for data in temp if str_to_float(data)])

    # 추출한 열들의 평균을 구해준다.
    d_l = np.asarray(d_l)
    data_list = np.mean(d_l, axis=0)
    data_list = data_list.tolist()

    # 실험 별 psnr 추이에서 처음 읽히는 값이 input_psnr 이다.
    input_psnr = data_list[0]

    # 너무 낮은 값들은 제거해준다.
    for _ in range(10):
        data_list.remove(min(data_list))

    # 잘 손질된 리스트를 사전에 넣어준다.
    data_lists[exp_name] = data_list

if print_input_psnr:
    # input psnr 을 10 칸정도 그려준다.
    init = [input_psnr] * 10
    data_lists['init'] = init

for items in data_lists.items():

    tt = np.arange(0, len(items[1]) * eval_period, eval_period)

    plt.plot(tt, items[1], label=items[0])

plt.legend(loc='lower right')
plt.title(title)
plt.show()
