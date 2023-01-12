# one content image has multiple style to transfer

import numpy as np
from PIL import Image
import scipy.misc
# import matplotlib.pyplot as plt

import numpy as np
# import SimpleITK as sitk
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from glob import glob
import time
import shutil
from PIL import Image
from ImageLoader import _dataset_info

# set seed for reproduction
np.random.seed(seed=1)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=["PACS", "OfficeHome", "camelyon17"])
parser.add_argument('--target', type=str)
# parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--mode', type=str)
parser.add_argument('--style', type=str)
parser.add_argument('--K', type=int)
args = parser.parse_args()

if args.dataset == 'PACS':
    all_clients = ["art_painting", "cartoon", "photo", "sketch"]
    # data_labels = ["dog", "elephant", "giraffe",
    #             "guitar", "horse", "house", "person"]
    path = os.getcwd() + "/PACS/kfold/"
elif args.dataset == 'OfficeHome':
    all_clients = ['art', 'clipart', 'product', 'real_world']
elif args.dataset == 'camelyon17':
    all_clients = ['hospital1', 'hospital2',
                   'hospital3', 'hospital4', 'hospital5']
    
data_labels = os.listdir(os.path.join(args.dataset,'kfold',all_clients[0]))
target_client = args.target
clients = list(set(all_clients) - set([target_client]))
path = os.path.join(args.dataset, 'kfold')

# mode = 'single'
# K = 1

for client in clients:
    print("Loading  %s ..." % client)
    txt_path = os.path.join(os.path.dirname(__file__), 'txt_lists', args.dataset.lower(), '%s_train.txt' % client)
    name_train, labels_train, = _dataset_info(txt_path)
    new_txt_path = txt_path.replace(args.dataset.lower(), args.dataset.lower() + f'_{args.style}-{args.mode}-K{args.K}/{target_client}' )
    if not os.path.exists(os.path.dirname(new_txt_path)):
        os.makedirs(os.path.dirname(new_txt_path))
    f = open(new_txt_path, 'a')
    
    for inpath, label in zip(name_train,labels_train):
        outpath = inpath.replace('kfold/', f'kfold_{args.style}-{args.mode}-multi/'+target_client+'/')

        
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))

        target_choice_list = np.random.choice(clients, args.K, replace=False)

        
        for target_choice in target_choice_list:
            if target_choice == client:
                f.write(outpath + ' ' + str(label) + '\n')
                print(outpath)
                continue

            pick_path = os.path.dirname(inpath.replace(client, target_choice))
            image_target = np.random.choice(os.listdir(pick_path), 1)[0]
            style_img_path = os.path.join(pick_path, image_target)

            outpath_new = outpath.replace('.', '_'+target_choice+'.')
            print(f"Target={args.target}, K={args.K}, {outpath_new}")
            f.write(outpath_new + ' ' + str(label) + '\n')
            
    f.close()

# ############### PACS #################
## Overall, K1
# python data_list_generator.py --dataset PACS --target art_painting --mode overall --style adain --K 1 &
# python data_list_generator.py --dataset PACS --target cartoon --mode overall --style adain --K 1 &
# python data_list_generator.py --dataset PACS --target photo --mode overall --style adain --K 1 &
# python data_list_generator.py --dataset PACS --target sketch --mode overall --style adain --K 1 &
## Overall, K2
# python data_list_generator.py --dataset PACS --target art_painting --mode overall --style adain --K 2 &
# python data_list_generator.py --dataset PACS --target cartoon --mode overall --style adain --K 2 &
# python data_list_generator.py --dataset PACS --target photo --mode overall --style adain --K 2 &
# python data_list_generator.py --dataset PACS --target sketch --mode overall --style adain --K 2 &
## Overall, K3
# python data_list_generator.py --dataset PACS --target art_painting --mode overall --style adain --K 3 &
# python data_list_generator.py --dataset PACS --target cartoon --mode overall --style adain --K 3 &
# python data_list_generator.py --dataset PACS --target photo --mode overall --style adain --K 3 &
# python data_list_generator.py --dataset PACS --target sketch --mode overall --style adain --K 3 &

# ############### OfficeHome #################
## Overall, K1
# python data_list_generator.py --dataset OfficeHome --target art --mode overall --style adain --K 1 &
# python data_list_generator.py --dataset OfficeHome --target clipart --mode overall --style adain --K 1 &
# python data_list_generator.py --dataset OfficeHome --target product --mode overall --style adain --K 1 &
# python data_list_generator.py --dataset OfficeHome --target real_world --mode overall --style adain --K 1 &
## Overall, K2
# python data_list_generator.py --dataset OfficeHome --target art --mode overall --style adain --K 2 &
# python data_list_generator.py --dataset OfficeHome --target clipart --mode overall --style adain --K 2 &
# python data_list_generator.py --dataset OfficeHome --target product --mode overall --style adain --K 2 &
# python data_list_generator.py --dataset OfficeHome --target real_world --mode overall --style adain --K 2 &
## Overall, K3
# python data_list_generator.py --dataset OfficeHome --target art --mode overall --style adain --K 3 &
# python data_list_generator.py --dataset OfficeHome --target clipart --mode overall --style adain --K 3 &
# python data_list_generator.py --dataset OfficeHome --target product --mode overall --style adain --K 3 &
# python data_list_generator.py --dataset OfficeHome --target real_world --mode overall --style adain --K 3 &

# ############### camelyon17 #################
## Overall, K4
# python data_list_generator.py --dataset camelyon17 --target hospital4 --mode overall --style adain --K 4 &
# python data_list_generator.py --dataset camelyon17 --target hospital5 --mode overall --style adain --K 4 &
## Singel, K4
# python data_list_generator.py --dataset camelyon17 --target hospital4 --mode single --style adain --K 4 &
# python data_list_generator.py --dataset camelyon17 --target hospital5 --mode single --style adain --K 4 &