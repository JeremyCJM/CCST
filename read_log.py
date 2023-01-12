import os
import argparse

# def max_test(path):
#     f = open(path)
#     print(path)
#     acc_list = []
#     line = f.readline()
#     while line:
#         if 'Global Test Class Acc' in line:
#             # import pdb; pdb.set_trace()
#             acc_list.append(float(line.split('Global Test Class Acc: ')[1].replace('\n','')))
#         line = f.readline()
#     f.close()

#     print(max(acc_list))
#     return max(acc_list)

def max_test(path):
    f = open(path)
    print(path)
    best_acc = 0
    incomplete_flag = True
    line = f.readline()
    line_pre = ''
    while line:
        if 'Saving current best checkpoints to' in line:
            # import pdb; pdb.set_trace()
            best_acc = float(line_pre.split('Global Test Class Acc: ')[1].replace('\n',''))
        if 'Global iter is 499' in line:
            incomplete_flag = False
        line_pre = line
        line = f.readline()
    f.close()

    print(best_acc)
    if incomplete_flag:
        print('Incomplete!')
    return best_acc

# dirpath = '/home/cjm/disk1/research/DG4FL/logs/fedavg_adain_overall_multi_resnet50_RSC'

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, help ='log dir path')
args = parser.parse_args()

dirpath = args.path

acc_list = []

for subdir, dirs, files in os.walk(dirpath):
    for filename in files:
        filepath = subdir + os.sep + filename

        # if filepath.endswith("fedavg.log"):
        if filepath.endswith(".log"):
            acc_list.append(max_test(filepath))

import numpy as np
print(f'Average: {np.mean(acc_list)}')
