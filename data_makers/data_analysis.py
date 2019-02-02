import pandas
import matplotlib.pyplot as plt
import numpy as np
import os
import utils


def str_to_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False


# ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
# 한 실험에 사용된 모든 validation 에 대한 psnr 추이.
# https://stickie.tistory.com/80  참고

# 실험 제목
exp_name = 'exp002_1'

# psnr 이 출력되는 열의 index
data_start_column = 3

# 실험에서 eval 하는 iteration 주기
eval_period = 5000

# 실험 경로
exp_dir = os.path.dirname(os.getcwd()) + '/exp/' + exp_name

# 그래프들이 저장되는 경로
result_saved_dir = utils.make_dirs(exp_dir + '/' + 'plots')

# log 가 저장된 csv 파일의 경로
data = pandas.read_csv(exp_dir + '/' + exp_name + '_log.csv')

for i, column in enumerate(range(data_start_column, len(data.columns))):
    data_list = data.iloc[:, column].values.tolist()
    data_name = str(i).zfill(5) + '_' + data.columns[column].split('.')[0]
    print(data_name)

    data_list = [float(data) for data in data_list if str_to_float(data)]

    # 너무 낮게 나온 값들은 제거해준다
    for _ in range(10):
        data_list.remove(min(data_list))

    tt = np.arange(0, len(data_list) * eval_period, eval_period)

    plt.plot(tt, data_list)

    # plt.show()
    plt.savefig(result_saved_dir + '/' + data_name + '.jpg')
    plt.cla()



