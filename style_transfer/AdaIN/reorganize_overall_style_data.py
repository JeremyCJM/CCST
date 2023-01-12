# one content image has multiple style to transfer

import numpy as np
from PIL import Image
import scipy.misc
# import matplotlib.pyplot as plt

import numpy as np
# import SimpleITK as sitk
import os
from glob import glob
import time
import shutil
from PIL import Image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=["PACS", "OfficeHome", "camelyon17"])
parser.add_argument('--target', type=str)
parser.add_argument('--mode', type=str, choices=["Single", "Overall"])
parser.add_argument('--net2', action='store_true', help="use net2")
args = parser.parse_args()


if args.dataset == 'PACS':
    all_clients = ["art_painting", "cartoon", "photo", "sketch"]
    # data_labels = ["dog", "elephant", "giraffe",
    #             "guitar", "horse", "house", "person"]
    path = os.getcwd() + "/PACS/kfold/"
elif args.dataset == 'OfficeHome':
    all_clients = ['art', 'clipart', 'product', 'real_world']
elif args.dataset == 'DigitsFive':
    all_clients = ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits', 'USPS']
elif args.dataset == 'camelyon17':
    all_clients = ['hospital1', 'hospital2',
                   'hospital3', 'hospital4', 'hospital5']

data_labels = os.listdir(os.path.join(args.dataset, 'kfold', all_clients[0]))
source_clients = list(set(all_clients) - set([args.target]))
path = os.path.join(args.dataset, 'kfold')

test_list = []
for client in all_clients:
    with open(f'txt_lists/camelyon17/{client}_test.txt', 'r') as f:
        test = f.readlines()
    test_list = test_list + [mm.split(' ')[0] for mm in test]
base_path = '/disk1/cjm/research/DG4FL/data'

for client in source_clients:
    print("Client:"+client)
    cur_path = os.path.join(path, client)

    for data_label in data_labels:
        label_path = os.path.join(cur_path, data_label)
        print("Class:" + data_label)

        for image_path in os.listdir(label_path):
            inpath = os.path.join(label_path, image_path)
            if os.path.join(base_path, inpath) in test_list:
                continue

            outpath = inpath.replace(
                'kfold/', f'kfold_adain-{args.mode.lower()}-multi/'+args.target+'/')

            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))

            # target_choice = np.random.choice(clients, 1)[0]

            for target_choice in source_clients:
                if target_choice == client:
                    if os.path.exists(outpath):
                        continue
                    shutil.copyfile(inpath, outpath)
                    print(f"Target: {args.target}, {outpath}")
                else:
                    outpath_new = outpath.replace('.', '_'+target_choice+'.')
                    if os.path.exists(outpath_new):
                        continue

                    inpath_new = inpath.replace(f"kfold/{client}", f"all_style_transferred_{args.mode}/{client}/{target_choice}/train").replace(f"/{data_label}",'').replace('.png', f"_{target_choice}.png")
                    if not os.path.exists(inpath_new):
                        print(f"Not exist: {inpath_new}")
                        import pdb
                        pdb.set_trace()
                    shutil.copyfile(inpath_new, outpath_new)
                    print(f"Target: {args.target}, {outpath_new}")

print(f"Target {args.target} finished!")


# camelyon17
# python reorganize_dataset.py --dataset camelyon17 --mode Overall --target hospital1 &
# python reorganize_dataset.py --dataset camelyon17 --mode Overall --target hospital2 &
# python reorganize_dataset.py --dataset camelyon17 --mode Overall --target hospital3 &
# python reorganize_dataset.py --dataset camelyon17 --mode Overall --target hospital4 &
# python reorganize_dataset.py --dataset camelyon17 --mode Overall --target hospital5


# PACS
# python reorganize_dataset.py --dataset PACS --mode Overall --target art_painting &
# python reorganize_dataset.py --dataset PACS --mode Overall --target cartoon &
# python reorganize_dataset.py --dataset PACS --mode Overall --target photo &
# python reorganize_dataset.py --dataset PACS --mode Overall --target sketch 