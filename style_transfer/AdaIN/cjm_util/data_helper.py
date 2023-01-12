# taken from https://github.com/fmcarlucci/JigenDG/blob/master/data/data_helper.py

from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cjm_util.ImageLoader import ImageDataset, ImageTestDataset, get_split_dataset_info, _dataset_info


# vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
# office_datasets = ["amazon", "dslr", "webcam"]
# digits_datasets = [mnist, mnist, svhn, usps]
officehome_datasets = ['art', 'clipart', 'product', 'real_world']
digit_datasets = ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits', 'USPS']
camelyon17_datasets = ['hospital1', 'hospital2', 'hospital3', 'hospital4', 'hospital5']
available_datasets = pacs_datasets + officehome_datasets + digit_datasets + camelyon17_datasets
#office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
#pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
#vlcs_paths = {dataset: "/home/enoon/data/images/VLCS/%s/test" % dataset for dataset in pacs_datasets}
#paths = {**office_paths, **pacs_paths, **vlcs_paths}

# dataset_mean = {'digitsfive':(0.5,0.5,0.5),
#                 'pacspacs': (0.485, 0.456, 0.406),
#                 'officehome':(0.485, 0.456, 0.406),
#                 'camelyon17':(0.485, 0.456, 0.406),
#                 }

# dataset_std = {'digitsfive':(0.5,0.5,0.5),
#                'pacs': (0.229, 0.224, 0.225),
#                'officehome':(0.229, 0.224, 0.225),
#                'camelyon17':(0.229, 0.224, 0.225),
#                }


def get_train_dataloader(args):
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', f'{args.dataset.lower()}','%s_train.txt' % args.target))
    img_tr = get_transformer(args)
    val_dataset = ImageTestDataset(names, labels, img_transformer=img_tr)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    
    return loader

def get_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor() ]# ,
            #   transforms.Normalize(dataset_mean[args.dataset.lower()], std=dataset_std[args.dataset.lower()])] # may not need normalize.
    return transforms.Compose(img_tr)

