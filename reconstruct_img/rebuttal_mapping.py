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

def cast_to_image(tensor):
    tensor = tensor.clamp(0.0,1.0)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


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
parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
parser.add_argument('--save_path', type = str, default='../checkpoint', help='path to save the checkpoint')
parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
# FedProx hyper params
# parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
# Dataset setting
parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
parser.add_argument("--dataset", choices=['pacs', 'officehome', 'digitsfive', 'camelyon17'], default='pacs', help="Dataset, to indicate the txt path")
parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
parser.add_argument("--target", choices=available_datasets, default='cartoon', help="Target")
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


writer = SummaryWriter(f"./logs/{args.source[0]}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('===> Building model')

vgg.eval()
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)

generator = Generator(image_size=args.image_size, latent_dim=1024).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=args.lr)

print('===> Loading model')
# train_loaders, val_loaders = data_helper.get_train_dataloader(args)
train_loader, val_loader = data_helper.get_train_dataloader(args)

# TODO: check which domain it is.

# import pdb; pdb.set_trace()
idx = 0
# train_loader = train_loaders[idx]
# val_loader = val_loaders[idx]

def train(epoch):
    print('===> Training')
    epoch_loss = 0
    avg_psnr = 0
    for it,data in enumerate(train_loader): 
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            # vgg_feat = vgg(img.to(device).unsqueeze(0))
            vgg_feat = vgg(img)
        style_stats = calc_mean_std(vgg_feat)

        recon_imgs = generator(style_stats)

        # import pdb; pdb.set_trace()
        loss = criterion(recon_imgs, img)
        epoch_loss += loss.item()
        
        # import pdb; pdb.set_trace()
        psnr = 10 * log10(1 / loss.item())
        avg_psnr += psnr

        loss.backward()
        optimizer.step()
        print("===> Epoch[{}]({}/{}): Train Loss: {:.4f}".format(epoch, it, len(train_loader), loss.item()))
    print("===> Epoch {} Complete: Avg. Train Loss: {:.4f}, PSNR: {:.4f}".format(epoch, epoch_loss / len(train_loader), avg_psnr / len(train_loader)))

    writer.add_scalar('train loss', epoch_loss / len(train_loader), epoch) 
    writer.add_scalar('train psnr', avg_psnr / len(train_loader), epoch) 

    if epoch % args.save_freq == 0:
        writer.add_image('train gt', cast_to_image(img[0]), epoch)
        writer.add_image('train pred', cast_to_image(recon_imgs[0]), epoch)
    # style_stat = [stat.to(device) for stat in style_stat]

def val(epoch):
    avg_psnr = 0
    with torch.no_grad():
        for batch in val_loader:
            img, _ = batch
            img = img.to(device)
            vgg_feat = vgg(img)
            style_stats = calc_mean_std(vgg_feat)

            # import pdb; pdb.set_trace()
            recon_imgs = generator(style_stats)
            mse = criterion(recon_imgs, img)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. Val PSNR: {:.4f} dB".format(avg_psnr / len(val_loader)))

    writer.add_scalar('val psnr', avg_psnr / len(val_loader), epoch) 
    if epoch % args.save_freq == 0:
        writer.add_image('val gt', cast_to_image(img[0]), epoch)
        writer.add_image('val pred', cast_to_image(recon_imgs[0]), epoch)
    return avg_psnr


start_epoch, best_psnr = 1, 0.0
if args.resume:
    ckpt = torch.load(f"./logs/{args.source[0]}/model_lastest.pth")
    generator.load_state_dict(ckpt['model'])
    start_epoch = ckpt['epoch'] + 1
    best_psnr = ckpt['best_psnr'] if 'best_psnr' in ckpt.keys() else best_psnr
    if 'optimizer' in ckpt.keys():
        generator.load_state_dict(ckpt['optimizer'])


for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    psnr = val(epoch)
    model_dicts = {
                'model': generator.state_dict(),
                'epoch': epoch,
                'best_psnr': best_psnr,
                'optimizer': optimizer.state_dict()
                }
    torch.save(model_dicts, f"./logs/{args.source[0]}/model_lastest.pth")
    if psnr > best_psnr:
        best_psnr = psnr
        best_dicts = {
                'model': generator.state_dict(),
                'epoch': epoch
                }
        torch.save(best_dicts, f"./logs/{args.source[0]}/model_best.pth")

# CUDA_VISIBLE_DEVICES=2 python rebuttal_mapping.py --dataset pacs --source art_painting --epochs 1000000 --batch 32 --image_size 256 --save_freq 10 &
# CUDA_VISIBLE_DEVICES=3 python rebuttal_mapping.py --dataset pacs --source photo --epochs 1000000 --batch 32 --image_size 256 --save_freq 10 &
# CUDA_VISIBLE_DEVICES=4 python rebuttal_mapping.py --dataset pacs --source cartoon --epochs 1000000 --batch 32 --image_size 256 --save_freq 10 &
# CUDA_VISIBLE_DEVICES=5 python rebuttal_mapping.py --dataset pacs --source sketch --epochs 1000000 --batch 32 --image_size 256 --save_freq 10 &
# CUDA_VISIBLE_DEVICES=6 python rebuttal_mapping.py --dataset camelyon17 --source hospital1 --epochs 1000000 --batch 64 --image_size 128 --save_freq 1 &
# CUDA_VISIBLE_DEVICES=7 python rebuttal_mapping.py --dataset camelyon17 --source hospital2 --epochs 1000000 --batch 64 --image_size 128 --save_freq 1 &
