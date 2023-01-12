# taken from https://github.com/fmcarlucci/JigenDG/blob/master/data/data_helper.py

from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from rebuttal_util.ImageLoader import ImageDataset, ImageTestDataset, get_split_dataset_info, _dataset_info


# vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
# office_datasets = ["amazon", "dslr", "webcam"]
# digits_datasets = [mnist, mnist, svhn, usps]
officehome_datasets = ['art', 'clipart', 'product', 'real_world']
digit_datasets = ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits', 'USPS']
camelyon17_datasets = ['hospital1', 'hospital2', 'hospital3', 'hospital4', 'hospital5']
available_datasets = pacs_datasets + officehome_datasets + digit_datasets + camelyon17_datasets



def get_train_dataloader(args):
    dataset_list = args.source
    dname = args.source[0]
    assert isinstance(dataset_list, list)
    img_tr = get_transformer(args)

    limit = args.limit_source

    train_txt_path = join(dirname(__file__), 'txt_lists', args.dataset.lower())
    name_train, name_val, labels_train, labels_val = get_split_dataset_info(f'../data/txt_lists/{args.dataset}/%s_train.txt' % dname, args.val_size)

    train_dataset = ImageDataset(name_train, labels_train, img_transformer=img_tr)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    val_dataset = ImageDataset(name_val, labels_val, img_transformer=img_tr)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    return train_loader, val_loader

def get_test_dataloader(args, target):
    names, labels = _dataset_info(f'../data/txt_lists/{args.dataset}/{target}_test.txt')
    img_tr = get_transformer(args)
    test_dataset = ImageTestDataset(names, labels, img_transformer=img_tr)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    return loader

def get_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor() ]# ,
            #   transforms.Normalize(dataset_mean[args.dataset.lower()], std=dataset_std[args.dataset.lower()])] # may not need normalize.
    return transforms.Compose(img_tr)

