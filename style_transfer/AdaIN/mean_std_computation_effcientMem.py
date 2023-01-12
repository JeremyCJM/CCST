import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral

from cjm_util import data_helper
import numpy as np
from datetime import datetime 
import os

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()

parser.add_argument('--vgg', type=str, 
                    default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str,
                    default='models/decoder.pth')

# Additional options
# parser.add_argument('--content_size', type=int, default=512,
#                     help='New (minimum) size for the content image, \
#                     keeping the original size if set to 0')
# parser.add_argument('--style_size', type=int, default=512,
#                     help='New (minimum) size for the style image, \
#                     keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--output_name', type=str,
                    help='output image path')
parser.add_argument('--image_size', type=int, default=222, help='image size')
parser.add_argument('--dataset', type=str, default='pacs', help='dataset name')

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
parser.add_argument("--target", help="Target")
args = parser.parse_args()

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


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    
    size = feat.shape
    assert (len(size) == 4)
    N, C, _, _ = size
    feat = feat.swapaxes(1,0)

    feat_var = feat.reshape(C, -1).var(axis=1) + eps
    feat_std = torch.sqrt(feat_var.reshape(1, C, 1, 1))
    feat_mean = feat.reshape(C, -1).mean(axis=1).reshape(1, C, 1, 1)
    # import pdb; pdb.set_trace() # feat:[1, 512, 87, 64], feat_mean&std:[1, 512, 1, 1], 
    return feat_mean, feat_std

def calc_sum(feat):
    feat = feat.detach()
    size = feat.shape
    assert (len(size) == 4)
    N, C, H, W = size
    count = N * H * W
    feat = feat.swapaxes(1,0)

    feat_sum = feat.reshape(C, -1).sum(axis=1).reshape(1, C, 1, 1)
    feat_square = feat ** 2
    feat_square_sum = feat_square.reshape(C, -1).sum(axis=1).reshape(1, C, 1, 1)
    # import pdb; pdb.set_trace() 
    return feat_sum, feat_square_sum, count

all_feat_sum, all_feat_square_sum, all_count, img_count = 0, 0, 0, 0
data_loader = data_helper.get_train_dataloader(args)
start_time = datetime.now() 
for it, (data, _) in enumerate(data_loader):
    data = data.to(device)
    feat = vgg(data) # torch.Size([4, 512, 64, 64])
    img_count += len(data)
    del data

    feat_sum, feat_square_sum, count = calc_sum(feat)
    del feat
    
    all_feat_sum += feat_sum
    all_feat_square_sum += feat_square_sum
    all_count += count
    print(f"{it}/{len(data_loader)}")

# import pdb; pdb.set_trace()
feat_mean = all_feat_sum / float(all_count)
feat_var = all_feat_square_sum / float(all_count) - feat_mean ** 2
feat_std = torch.sqrt(feat_var + 1e-5)
end_time = datetime.now() 

print(feat_mean.shape, feat_std.shape)

if not os.path.exists(f'style_stats/{args.dataset}/'):
    os.makedirs(f'style_stats/{args.dataset}/')

# uncomment this line to save the results
# np.save( f'style_stats/{args.dataset}/{args.target}_mean_std.npy', [feat_mean.cpu().numpy(), feat_std.cpu().numpy()])


# rebuttal
print(f"Target {args.target}: Finished in {(end_time - start_time).seconds} seconds")
with open(f"style_stats/{args.dataset}/{args.target}_style_comp_time.txt", 'w') as f:
    f.write(f"Target {args.target}: Finished in {(end_time - start_time).seconds} seconds\n")
    f.write(f"Images number: {img_count}\n")
    f.write(f"Image resolution: {args.image_size}\n")
    f.write(f"Batch_size: {args.batch}\n")



############################ PACS ############################
# CUDA_VISIBLE_DEVICES=2 python mean_std_computation_effcientMem.py --dataset pacs --image_size 512 --target art_painting --batch 32 &&
# CUDA_VISIBLE_DEVICES=3 python mean_std_computation_effcientMem.py --dataset pacs --image_size 512 --target cartoon --batch 32 &&
# CUDA_VISIBLE_DEVICES=4 python mean_std_computation_effcientMem.py --dataset pacs --image_size 512 --target photo --batch 32 &&
# CUDA_VISIBLE_DEVICES=5 python mean_std_computation_effcientMem.py --dataset pacs --image_size 512 --target sketch --batch 32

############################ officehome ############################
# CUDA_VISIBLE_DEVICES=2 python mean_std_computation_effcientMem.py --dataset officehome --target art --batch 32 &
# CUDA_VISIBLE_DEVICES=3 python mean_std_computation_effcientMem.py --dataset officehome --target clipart --batch 32 &
# CUDA_VISIBLE_DEVICES=4 python mean_std_computation_effcientMem.py --dataset officehome --target product --batch 32 &
# CUDA_VISIBLE_DEVICES=5 python mean_std_computation_effcientMem.py --dataset officehome --target real_world --batch 32

############################ camelyon17 ############################
# CUDA_VISIBLE_DEVICES=5 python mean_std_computation_effcientMem.py --dataset camelyon17 --image_size 512 --target hospital1 --batch 4 &
# CUDA_VISIBLE_DEVICES=6 python mean_std_computation_effcientMem.py --dataset camelyon17 --image_size 512 --target hospital2 --batch 4 &
# CUDA_VISIBLE_DEVICES=7 python mean_std_computation_effcientMem.py --dataset camelyon17 --image_size 512 --target hospital3 --batch 4 &
# CUDA_VISIBLE_DEVICES=8 python mean_std_computation_effcientMem.py --dataset camelyon17 --image_size 512 --target hospital4 --batch 4 &
# CUDA_VISIBLE_DEVICES=9 python mean_std_computation_effcientMem.py --dataset camelyon17 --image_size 512 --target hospital5 --batch 4
