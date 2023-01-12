# taken from https://github.com/fmcarlucci/JigenDG/blob/master/data/data_helper.py

from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.ImageLoader import ImageDataset, ImageTestDataset, get_split_dataset_info, _dataset_info, JigsawDataset, FedDGDataset, PACS_AMP, OfficeHome_AMP, Camelyon17_AMP
from data.concat_dataset import ConcatDataset


pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
officehome_datasets = ['art', 'clipart', 'product', 'real_world']
digit_datasets = ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits', 'USPS']
camelyon17_datasets = ['hospital1', 'hospital2', 'hospital3', 'hospital4', 'hospital5']
available_datasets = pacs_datasets + officehome_datasets + digit_datasets + camelyon17_datasets


dataset_mean = {
                'pacs': (0.485, 0.456, 0.406),
                'officehome':(0.485, 0.456, 0.406),
                'camelyon17':(0.485, 0.456, 0.406),
                }

dataset_std = {
               'pacs': (0.229, 0.224, 0.225),
               'officehome':(0.229, 0.224, 0.225),
               'camelyon17':(0.229, 0.224, 0.225),
               }

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(args):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    loader_list, val_loader_list = [], []
    if args.dg_method.lower() == 'jigsaw':
        train_img_transformer, train_tile_transformer = get_train_transformers(args)
    else:
        train_img_transformer = get_train_transformers(args)
    val_img_transformer = get_val_transformer(args)
    limit = args.limit_source

    if args.dg_method.lower() == 'feddg':
        if args.dataset.lower() == 'pacs':
            amp_loader = PACS_AMP(args)
        elif args.dataset.lower() == 'officehome':
            amp_loader = OfficeHome_AMP(args)
        elif args.dataset.lower() == 'camelyon17':
            amp_loader = Camelyon17_AMP(args)

    if args.mode == 'deepall':
        name_train_all = []
        labels_train_all = []
    for dname in dataset_list:
        print("Prepare %s ..." % dname)
 
        train_txt_path = join(dirname(__file__), 'txt_lists', f'{args.dataset.lower()}_{args.fusion_mode}/{args.target}')

        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(train_txt_path, '%s_train.txt' % dname), args.val_size)

        
        name_train, labels_train = creat_train_loader_list(name_train, labels_train, args.fusion_mode, args.source, args.target)

        
        import pdb; pdb.set_trace()
        if args.mode == 'deepall':
            name_train_all +=  name_train
            labels_train_all += labels_train
        else:
            if args.dg_method.lower() == 'jigsaw':
                train_dataset = JigsawDataset(name_train, labels_train, img_transformer=train_img_transformer, tile_transformer=train_tile_transformer,
                                                bias_whole_image=args.bias_whole_image)
            elif args.dg_method.lower() == 'feddg':
                train_dataset = FedDGDataset(name_train, labels_train, img_transformer=train_img_transformer, amp_loader=amp_loader)
            else:
                train_dataset = ImageDataset(name_train, labels_train, img_transformer=train_img_transformer)

            val_dataset = ImageTestDataset(name_val, labels_val, img_transformer=val_img_transformer)
            if limit:
                train_dataset = Subset(train_dataset, limit)
                val_dataset = Subset(val_dataset, limit)
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
            loader_list.append(loader)
            val_loader_list.append(val_loader)

    if args.mode == 'deepall':
        name_train = name_train_all
        labels_train = labels_train_all
        if args.dg_method.lower() == 'jigsaw':
            train_dataset = JigsawDataset(name_train, labels_train, img_transformer=train_img_transformer, tile_transformer=train_tile_transformer,
                                             bias_whole_image=args.bias_whole_image)
        else:
            train_dataset = ImageDataset(name_train, labels_train, img_transformer=train_img_transformer)

        val_dataset = ImageTestDataset(name_val, labels_val, img_transformer=val_img_transformer)
        if limit:
            train_dataset = Subset(train_dataset, limit)
            val_dataset = Subset(val_dataset, limit)
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
        loader_list.append(loader)
        val_loader_list.append(val_loader)



    return loader_list, val_loader_list

def creat_train_loader_list(name_train, labels_train, mode, source, target):
    # substitute train to the transformed version
    if mode != 'no_fusion' and '-K' not in mode:
        name_train = [name.replace('kfold', 'kfold_overall-multi' + '/' + target) for name in name_train]

    if "single-K" in mode:
        name_train = [name.replace(
            'overall-multi', 'single-multi') for name in name_train]

    if 'multi' in mode:
        temp_name_list = []
        temp_label_list = []
        for domain in source:
            name_train_temp = [name.replace('.', '_'+domain+'.') for name in name_train if domain not in name]
            temp_name_list += name_train_temp
            temp_label_list += labels_train
        name_train += temp_name_list
        labels_train += temp_label_list

    
    return name_train, labels_train
        

def get_test_dataloader(args):
    '''target test loader'''
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', args.dataset, '%s_test.txt' % args.target))
    img_tr = get_val_transformer(args)
    val_dataset = ImageTestDataset(names, labels, img_transformer=img_tr)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    # dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=True) # True is for tent optimization
    return loader


def get_train_transformers(args):

    if args.dg_method.lower() == 'jigsaw':
        img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
        if args.random_horiz_flip > 0.0:
            img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
        tile_tr = [transforms.ToTensor(), transforms.Normalize(dataset_mean[args.dataset], std=dataset_std[args.dataset])]
        return transforms.Compose(img_tr), transforms.Compose(tile_tr)
    else:
        img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale)), 
                transforms.ToTensor(), transforms.Normalize(dataset_mean[args.dataset], std=dataset_std[args.dataset])]
        if args.random_horiz_flip > 0.0:
            img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
        return transforms.Compose(img_tr)



def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize(dataset_mean[args.dataset], std=dataset_std[args.dataset])]
    return transforms.Compose(img_tr)
