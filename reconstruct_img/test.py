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
import lpips

seed = 1
random.seed(a=seed)
np.random.seed(seed)
torch.manual_seed(seed)     
torch.cuda.manual_seed_all(seed) 

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    # import pdb; pdb.set_trace() # feat:[1, 512, 87, 64], feat_mean&std:[1, 512, 1, 1], 
    style_stat = torch.cat((feat_mean, feat_std), dim=1)
    return style_stat.squeeze()

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


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
parser.add_argument("--source", choices=available_datasets, default='sketch', help="Source")
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
parser.add_argument("--train_data", choices=['original', 'imagenet_mse', 'imagenet_lpips'], default='original', help="choose ckpt trained on which dataset")
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
print('===> Building model')

vgg.eval()
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)

generator = Generator(image_size=args.image_size, latent_dim=1024).to(device)
generator.eval()

criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(generator.parameters(), lr=args.lr)

print('===> Loading model')

if args.train_data == 'original':
    ckpt = torch.load(f"./logs/{args.source}/model_best.pth")
else:
    ckpt = torch.load(f"./logs/{args.train_data}/model_best.pth.tar")

if not os.path.exists(f"./out_images/{args.source}/"):
    os.makedirs(f"./out_images/{args.source}/")
shutil.copyfile(f"./logs/{args.source}/model_best.pth", f"./out_images/{args.source}/model_best.pth")
# import pdb; pdb.set_trace()
# generator.load_state_dict(ckpt['model'])

ckpt_new = {}
for k_old in ckpt['state_dict'].keys():
    k_new = k_old.replace('module.', '')
    ckpt_new[k_new] = ckpt['state_dict'][k_old]

generator.load_state_dict(ckpt_new)
lpips_loss = lpips.LPIPS(net='vgg').cuda()

for target in args.target:
    test_loader = data_helper.get_test_dataloader(args, target)
    out_dir = f"./out_images/{args.train_data}/{args.source}/{target}/"
    if 'imagenet' in args.train_data:
        out_dir = f"./out_images/{args.train_data}/{target}/"
    if not os.path.exists(os.path.join(out_dir, 'reconstructed')):
        os.makedirs(os.path.join(out_dir, 'reconstructed'))
    psnr_sum, lpips_sum, count = 0, 0, 0
    psnr_list, lpips_list, path_list = [], [], []
    with torch.no_grad():
        for batch, fpaths in test_loader:
            img = batch.to(device)
            vgg_feat = vgg(img)
            style_stats = calc_mean_std(vgg_feat)

            # import pdb; pdb.set_trace()
            recon_imgs = generator(style_stats)
            mse = criterion(recon_imgs, img)
            mse = mse.reshape(mse.shape[0], -1).mean(dim=1)
            # import pdb; pdb.set_trace()
            psnrs = 10 * torch.log10(1 / mse)
            lpipses = lpips_loss(recon_imgs, img, normalize=True)

            for fpath, psnr, lpips, r_img in zip(fpaths, psnrs, lpipses, recon_imgs):
                print(fpath)
                path_list.append(fpath)
                psnr_list.append(psnr.item())
                lpips_list.append(lpips.item())
                lpips_sum += lpips
                psnr_sum += psnr
                count += 1
                out_name = os.path.join(out_dir, 'reconstructed', fpath.split(target+'/')[-1])
                if not os.path.exists(os.path.dirname(out_name)):
                    os.makedirs(os.path.dirname(out_name))
                # import pdb; pdb.set_trace()
                save_image(r_img, out_name)

    print("===> Avg. Test PSNR: {:.4f} dB".format(psnr_sum / count))
    out_dic = {}
    out_dic['psnr_list'] = psnr_list
    out_dic['lpips_list'] = lpips_list
    out_dic['path_list'] = path_list
    out_dic['average_psnr'] = psnr_sum / count
    out_dic['average_lpips'] = lpips_sum / count
    np.save(os.path.join(out_dir, 'psnr_lpips.npy'), out_dic)


### Original
# CUDA_VISIBLE_DEVICES=0 python test.py --dataset pacs --target photo cartoon sketch --source art_painting --batch 64 --image_size 256 &
# CUDA_VISIBLE_DEVICES=1 python test.py --dataset pacs --target art_painting cartoon sketch --source photo --batch 64 --image_size 256 &
# CUDA_VISIBLE_DEVICES=2 python test.py --dataset pacs --target art_painting photo sketch --source cartoon --batch 64 --image_size 256 &
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset pacs --target art_painting photo cartoon --source sketch --batch 64 --image_size 256 &

# CUDA_VISIBLE_DEVICES=8 python test.py --dataset camelyon17 --source hospital1 --target hospital2 --batch 64 --image_size 128 &
# CUDA_VISIBLE_DEVICES=9 python test.py --dataset camelyon17 --source hospital2 --target hospital1 --batch 64 --image_size 128 &

### Imagenet_mse
# CUDA_VISIBLE_DEVICES=6 python test.py --dataset pacs --target photo cartoon sketch art_painting --train_data imagenet_mse --batch 32 --image_size 256 &
# CUDA_VISIBLE_DEVICES=7 python test.py --dataset camelyon17 --target hospital1 hospital2 --train_data imagenet_mse --batch 32 --image_size 128 &
### Imagenet_lpips
# CUDA_VISIBLE_DEVICES=8 python test.py --dataset pacs --target photo cartoon sketch art_painting --train_data imagenet_lpips --batch 32 --image_size 256 &
# CUDA_VISIBLE_DEVICES=9 python test.py --dataset camelyon17 --target hospital1 hospital2 --train_data imagenet_lpips --batch 32 --image_size 128 &


