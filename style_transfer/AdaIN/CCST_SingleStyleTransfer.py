# Takes content feature and the normalization statistics of style feature
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaIN_StyleStat_ContentFeat, coral

import numpy as np
from cjm_util import data_helper
from cjm_util.ImageLoader import _dataset_info
from datetime import datetime 

import os

import random
seed = 1
random.seed(a=seed)
np.random.seed(seed)
torch.manual_seed(seed)     
torch.cuda.manual_seed_all(seed) 

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style_stat, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaIN_StyleStat_ContentFeat(content_f, style_stat)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaIN_StyleStat_ContentFeat(content_f, style_stat)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def calc_sum(feat):
    feat = feat.detach()
    size = feat.shape
    assert (len(size) == 4)
    N, C, H, W = size
    count = N * H * W
    feat = feat.transpose(1,0)

    feat_sum = feat.reshape(C, -1).sum(axis=1).reshape(1, C, 1, 1)
    feat_square = feat ** 2
    feat_square_sum = feat_square.reshape(C, -1).sum(axis=1).reshape(1, C, 1, 1)
    # import pdb; pdb.set_trace() 
    return feat_sum, feat_square_sum, count


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--dataset', type=str,
                    help='dataset')
parser.add_argument('--target', type=str,
                    help='content domain')
parser.add_argument('--style_stat', type=str,
                    help='File path to the style statistics')
# parser.add_argument('--style', type=str,
#                     help='File path to the style image, or multiple style \
#                     images separated by commas if you want to do style \
#                     interpolation or spatial control')
# parser.add_argument('--style_dir', type=str,
#                     help='Directory path to a batch of style images')

parser.add_argument('--vgg', type=str, 
                    default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str,
                    default='models/decoder.pth')

# Additional options
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--output_name', type=str, default='out_image.png',
                    help='output image path')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

parser.add_argument('--batch', type = int, default= 32, help ='batch size')
parser.add_argument('--image_size', type=int, default=512, help='image size')
parser.add_argument('--output_size', type=int, default=-1,
                    help='transform images into final size')
args = parser.parse_args()



if args.dataset.lower() == 'pacs':
    all_clients = ["art_painting", "cartoon", "photo", "sketch"]
    path = os.getcwd() + "/PACS/kfold/"
elif args.dataset.lower() == 'officehome': 
    all_clients = ['art', 'clipart', 'product', 'real_world']
elif args.dataset.lower() == "digitsfive":
    all_clients = ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits', 'USPS']
elif args.dataset.lower() == "camelyon17":
    all_clients = ['hospital1', 'hospital2', 'hospital3', 'hospital4', 'hospital5']

style_domains = list(set(all_clients) - set([args.target]))

do_interpolation = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)



decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

train_txt_path = os.path.join('cjm_util/txt_lists', args.dataset.lower())
name_train, labels_train, = _dataset_info(os.path.join(train_txt_path, '%s_train.txt' % args.target))

data_loader = data_helper.get_train_dataloader(args)

style_tf = test_transform(args.style_size, args.crop)

if args.output_size > 0:
    resize = transforms.Resize(args.output_size)

for style in style_domains:
    print(f"Content: {args.target} | Style: {style}")
    if args.dataset == 'camelyon17':
        style_img_list_path = f"cjm_util/txt_lists/camelyon17_discardBlackWhite/{style}_train.txt"
    else:
        style_img_list_path = f"cjm_util/txt_lists/{args.dataset}/{style}_train.txt"
    with open(style_img_list_path, 'r') as f:
        style_img_list = f.readlines()
    style_img_list = [mm.split(' ')[0] for mm in style_img_list]

    # style_stat = [torch.Tensor(stat) for stat in style_stat]
    # style_stat = [stat.to(device) for stat in style_stat]

    start_time = datetime.now() 
    img_count = 0
    for it, (data, fpaths) in enumerate(data_loader):
        img_count += len(data)
        # continue_flag = True
        # for fpath in fpaths:
        #     file_name, ext = os.path.splitext(os.path.basename(fpath))
        #     # out_name = os.path.join(out_dir, f"{file_name}_{style}{ext}")
        #     out_name = fpath.replace('kfold', 'all_style_transferred_Single')
        #     out_name = out_name.replace(f'{args.target}', f'{args.target}/{style}')
        #     out_name = out_name.replace(f'{ext}', f'_{style}{ext}')
        #     if not os.path.exists(out_name):
        #         continue_flag = False
        #         break
        # if continue_flag:
        #     print(f"    [Skip!] Style: {style}, Iteration: {it}/{len(data_loader)}")
        #     continue
        print(f"    Style: {style}, Iteration: {it}/{len(data_loader)}")

        style_img_path = random.choice(style_img_list)
        style_img = style_tf(Image.open(str(style_img_path)).convert('RGB'))
        style_img_feat = vgg(style_img.to(device).unsqueeze(0))
        del style_img
        feat_sum, feat_square_sum, count = calc_sum(style_img_feat)
        del style_img_feat
        feat_mean = feat_sum / float(count)
        feat_var = feat_square_sum / float(count) - feat_mean ** 2
        feat_std = torch.sqrt(feat_var + 1e-5)
        style_stat = [feat_mean, feat_std]
        style_stat = [stat.to(device) for stat in style_stat]

        with torch.no_grad():
            output = style_transfer(vgg, decoder, data.to(device), style_stat, args.alpha)
        del data, style_stat
        if args.output_size > 0:
            output = resize(output)
        output = output.cpu()
        
        for out_img, fpath in zip(output,fpaths):
            # import pdb; pdb.set_trace()
            file_name, ext = os.path.splitext(os.path.basename(fpath))
            out_name = fpath.replace('kfold', 'all_style_transferred_Single')
            out_name = out_name.replace(f'{args.target}', f'{args.target}/{style}')
            out_name = out_name.replace(f'{ext}', f'_{style}{ext}')
            out_dir = os.path.dirname(out_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_image(out_img, out_name)
            # idx += 1
    # rebuttal
    end_time = datetime.now() 
    with open(f"{args.dataset}_{args.target}_single_stylize_time.txt", 'w') as f:
        f.write(f"Target {args.target} with style {style}: Finished in {(end_time - start_time).seconds} seconds\n")
        f.write(f"Images number: {img_count}\n")
        f.write(f"Image resolution: {args.image_size}\n")
        f.write(f"Batch_size: {args.batch}\n")
    import pdb; pdb.set_trace()

# pacs
# CUDA_VISIBLE_DEVICES=5 python CCST_SingleStyleTransfer.py --dataset pacs --target art_painting --batch 32 --image_size 512 &
# CUDA_VISIBLE_DEVICES=6 python CCST_SingleStyleTransfer.py --dataset pacs --target cartoon --batch 6 --image_size 512 &
# CUDA_VISIBLE_DEVICES=7 python CCST_SingleStyleTransfer.py --dataset pacs --target photo --batch 6 --image_size 512 &
# CUDA_VISIBLE_DEVICES=8 python CCST_SingleStyleTransfer.py --dataset pacs --target sketch --batch 6 --image_size 512 &

# camelyon17
# CUDA_VISIBLE_DEVICES=0 python CCST_SingleStyleTransfer.py --dataset camelyon17 --target hospital1 --batch 32 --image_size 512 --output_size 96 &
# CUDA_VISIBLE_DEVICES=0 python CCST_SingleStyleTransfer.py --dataset camelyon17 --target hospital2 --batch 32 --image_size 512 --output_size 96 &
# CUDA_VISIBLE_DEVICES=2 python CCST_SingleStyleTransfer.py --dataset camelyon17 --target hospital3 --batch 32 --image_size 512 --output_size 96 &
# CUDA_VISIBLE_DEVICES=2 python CCST_SingleStyleTransfer.py --dataset camelyon17 --target hospital4 --batch 32 --image_size 512 --output_size 96 &
# CUDA_VISIBLE_DEVICES=3 python CCST_SingleStyleTransfer.py --dataset camelyon17 --target hospital5 --batch 64 --image_size 512 --output_size 96

# CUDA_VISIBLE_DEVICES=3 python CCST_SingleStyleTransfer.py --dataset camelyon17 --target hospital1 --batch 64 --image_size 512 --output_size 96 &
# CUDA_VISIBLE_DEVICES=3 python CCST_SingleStyleTransfer.py --dataset camelyon17 --target hospital2 --batch 32 --image_size 512 --output_size 96 &
# CUDA_VISIBLE_DEVICES=4 python CCST_SingleStyleTransfer.py --dataset camelyon17 --target hospital5 --batch 64 --image_size 512 --output_size 96
