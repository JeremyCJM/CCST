# Rebuttal for verify whether we can reconstruct the original image from our style vectors.
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
from nets.models import nets_map, get_network
import argparse
from utils.Logger import Logger
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils
from rebuttal_util import data_helper
from rebuttal_util.data_helper import available_datasets
# from utils.excel_log import xlsx_init, xlsx_append
from utils import rsc_utils, rsc_utils_densenet
from math import log10, log2
from tensorboardX import SummaryWriter
from collections import OrderedDict
import random
from lightweight_gan import Generator
from torchvision.utils import save_image
import shutil

seed = 1
random.seed(a=seed)
np.random.seed(seed)
torch.manual_seed(seed)     
torch.cuda.manual_seed_all(seed) 





parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true', help ='whether to make a log')
parser.add_argument('--test', action='store_true', help ='test the pretrained model')
parser.add_argument('--tent_test', action='store_true', help ='test the pretrained model with the Tent test-time optimization')
parser.add_argument('--tent_test_on-the-fly', action='store_true', help ='test the pretrained model with the Tent test-time optimization one by one')
parser.add_argument('--IN_test', action='store_true', help ='test the pretrained model using IN with affine')
parser.add_argument('--batch', type = int, default= 32, help ='batch size')
parser.add_argument('--epochs', type = int, default=1000, help = 'epochs')
parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
parser.add_argument('--mode', type = str, default='fedavg', choices = ['fedavg', 'fedbn', 'adafea', 'fedprox', 'deepall'], help='fedavg | fedprox | fedbn')
parser.add_argument('--mu', type=float, default=1e-2, help='The hyper dparameter for fedprox')
parser.add_argument('--save_path', type = str, default='../checkpoint', help='path to save the checkpoint')
parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
# FedProx hyper params
# parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
# Dataset setting
parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
parser.add_argument("--dataset", choices=['pacs', 'officehome', 'digitsfive', 'camelyon17'], default='pacs', help="Dataset, to indicate the txt path")
parser.add_argument("--source", choices=available_datasets, help="Source")
parser.add_argument("--target", choices=available_datasets, help="Target",  nargs='+')
parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")
parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
# Experiment setting
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')   
parser.add_argument("--image_size", type=int, default=225, help="Image size")
parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")    
parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
parser.add_argument('--seed', type = int, default=1, help = 'random seed number')
parser.add_argument('--save_freq', type = int, default=1, help = 'how long to save a latest checkpoint')
parser.add_argument("--dg_method", choices=['no_DG', 'RSC', 'SWAD', 'Jigsaw', 'MixStyle', 'feddg'], default='no_DG', help="DG methods combined with FL")
parser.add_argument("--fusion_mode", choices=['no_fusion'], 
                        default='no_fusion', help="Data Fusion Modes")    
# Jigsaw hyper params
parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
parser.add_argument("--jig_weight", type=float, default=0.7, help="Weight for the jigsaw puzzle loss")
# FedDG params
parser.add_argument('--meta_step_size', type=float, default=1e-3, help='meta learning rate')
parser.add_argument('--clip_value', type=float, default=1.0, help='gradient clip')
parser.add_argument('--vgg', type=str, default='../style_transfer/AdaIN/models/vgg_normalised.pth')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(image_size=args.image_size, latent_dim=1024).to(device)
generator.eval()

criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(generator.parameters(), lr=args.lr)

print('===> Loading model')

ckpt = torch.load(f"./logs/{args.source}/model_best.pth")
if not os.path.exists(f"./out_images/{args.source}/"):
    os.makedirs(f"./out_images/{args.source}/")
shutil.copyfile(f"./logs/{args.source}/model_best.pth", f"./out_images/{args.source}/model_best.pth")
generator.load_state_dict(ckpt['model'])


datasets = args.target + [args.source]
for target in datasets:
    test_loader = data_helper.get_test_dataloader(args, target)
    out_dir = f"./out_images/{args.source}/{target}/"
    if not os.path.exists(os.path.join(out_dir, 'reconstructed')):
        os.makedirs(os.path.join(out_dir, 'reconstructed'))
    psnr_sum, count = 0, 0
    psnr_list, path_list = [], []

    style_stats = np.load(f"../style_transfer/AdaIN/style_stats/{args.dataset.lower()}/{target}_mean_std.npy")

    # import pdb; pdb.set_trace()
    style_stats = torch.from_numpy(style_stats).squeeze()
    style_stats = torch.cat((style_stats[0], style_stats[1]))
    style_stats = style_stats.unsqueeze(0)

    # import pdb; pdb.set_trace()
    recon_imgs = generator(style_stats.to(device))

    
    out_name = os.path.join(out_dir, "overall_reconstructed.png")
    if not os.path.exists(os.path.dirname(out_name)):
        os.makedirs(os.path.dirname(out_name))
    # import pdb; pdb.set_trace()
    save_image(recon_imgs[0], out_name)
print('Finished')


# CUDA_VISIBLE_DEVICES=0 python test_overall.py --dataset pacs --target photo cartoon sketch --source art_painting --image_size 256 &
# CUDA_VISIBLE_DEVICES=1 python test_overall.py --dataset pacs --target art_painting cartoon sketch --source photo --image_size 256 &
# CUDA_VISIBLE_DEVICES=2 python test_overall.py --dataset pacs --target art_painting photo sketch --source cartoon --image_size 256 &
# CUDA_VISIBLE_DEVICES=3 python test_overall.py --dataset pacs --target art_painting photo cartoon --source sketch --image_size 256 &

# CUDA_VISIBLE_DEVICES=8 python test_overall.py --dataset camelyon17 --source hospital1 --target hospital2 --image_size 128 &
# CUDA_VISIBLE_DEVICES=9 python test_overall.py --dataset camelyon17 --source hospital2 --target hospital1 --image_size 128 &


