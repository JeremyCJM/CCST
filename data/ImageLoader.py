# taken from https://github.com/fmcarlucci/JigenDG/blob/master/data/JigsawLoader.py

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
import os


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage=None):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


class ImageDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.data_path = ""
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        
    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        # framename = self.data_path + '/' + self.names[index].replace('/research/dept5/mrjiang/', '/home/mrjiang/Server_mrjiang/')
        # import pdb; pdb.set_trace()
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)
        
    def __getitem__(self, index):

        img = self.get_image(index)
        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)



class ImageTestDataset(ImageDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        
        return self._image_transformer(img), int(self.labels[index])


class JigsawDataset(ImageDataset):
    def __init__(self, names, labels, img_transformer=None, tile_transformer=None,  bias_whole_image=0.9):
        super(JigsawDataset,self).__init__(names, labels, img_transformer)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image
        print('Bias for whole image :', self.bias_whole_image)
        self.permutations = self.retrieve_permutations()
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer

        def make_grid(x):
            return torchvision.utils.make_grid(x, self.grid_size, padding=0)
        self.returnFunc = make_grid
    
    def retrieve_permutations(self, classes=30):
        all_perm = np.load(f'{os.path.dirname(__file__)}/permutations_{classes}.npy')
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm

    def get_tile(self, img, n):
        
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile
         
    def __getitem__(self, idx):
        img = self.get_image(idx)
        label = int(self.labels[idx])
    
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > np.random.rand():
                order = 0
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
        
        data = torch.stack(data, 0)
        return self.returnFunc(data), int(order), int(label)

# class ImageTestDatasetMultiple(ImageDataset):
#     def __init__(self, *args, **xargs):
#         super().__init__(*args, **xargs)
#         self._image_transformer = transforms.Compose([
#             transforms.Resize(255, Image.BILINEAR),
#         ])
#         self._image_transformer_full = transforms.Compose([
#             transforms.Resize(225, Image.BILINEAR),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         self._augment_tile = transforms.Compose([
#             transforms.Resize((75, 75), Image.BILINEAR),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#     def __getitem__(self, index):
#         framename = self.data_path + '/' + self.names[index]
#         _img = Image.open(framename).convert('RGB')
#         img = self._image_transformer(_img)
#         return img, int(self.labels[index])

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ,ratio=1.0):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    # print (b)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    ratio = np.random.randint(1,10)/10

    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    # a_trg[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2]
    # a_trg = np.fft.ifftshift( a_trg, axes=(-2, -1) )
    return a_src

def source_to_target_freq( src_img, amp_trg, L=0.1 ,ratio=1.0):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img.transpose((2, 0, 1))
    # amp_trg = amp_trg.transpose((2, 0, 1))
    # print('##', src_img.shape)
    src_img_np = src_img #.cpu().numpy()
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    # print('##', amp_trg.shape)
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L ,ratio=1.0)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg.transpose(1, 2, 0)


class PACS_AMP():
    def __init__(self, args):
        print('Loading pacs amp lists...')
        self.src_sites = args.source
        self.amp_paths = []
        min_len = 1e8
        for dname in self.src_sites:
            name_train, labels_train, = _dataset_info(os.path.join(os.path.dirname(__file__), 'txt_lists', args.dataset, '%s_train.txt' % dname))
            self.amp_paths.append([p.replace('kfold', 'kfold_amp') for p in name_train])
            if min_len > len(name_train): min_len = len(name_train)

        # share_source_list = [ s for s in args.source if s!=dname]
        # amp_paths = [[p.replace('kfold', 'amps').replace(dname, sname) for p in names] for sname in share_source_list]
        # amp_paths.insert(0, [p.replace('kfold', 'amps') for p in names])
        self.min_len = min_len
        print('seen domain freqs: ', self.src_sites)
        

    def get_amp(self):
        site_idx = np.random.choice(len(self.src_sites))
        # print(site_idx)
        # follow ELCFS, 1/8 length of train data
        tar_freq_path = self.amp_paths[site_idx][np.random.randint(self.min_len//8)]
        # print(tar_freq_path)
        img_format = tar_freq_path.split('.')[-1]
        tar_freq_path = tar_freq_path.replace(img_format, 'npy')#.replace('/research/dept5/mrjiang/', '/home/mrjiang/Server_mrjiang/')
        # print(tar_freq_path)
        tar_freq = np.load(os.path.join(tar_freq_path))
        return tar_freq

class OfficeHome_AMP():
    def __init__(self, args):
        print('Loading officehome amp lists...')
        self.src_sites = args.source
        self.amp_paths = []
        min_len = 1e8
        for dname in self.src_sites:
            name_train, labels_train, = _dataset_info(args, os.path.join(os.path.dirname(__file__), 'txt_lists', args.dataset, '%s_train.txt' % dname))
            self.amp_paths.append([p.replace('OfficeHome', 'OfficeHome/amp') for p in name_train])
            if min_len > len(name_train): min_len = len(name_train)

        # share_source_list = [ s for s in args.source if s!=dname]
        # amp_paths = [[p.replace('kfold', 'amps').replace(dname, sname) for p in names] for sname in share_source_list]
        # amp_paths.insert(0, [p.replace('kfold', 'amps') for p in names])
        self.min_len = min_len
        print('seen domain freqs: ', self.src_sites)
        
        

    def get_amp(self):
        site_idx = np.random.choice(len(self.src_sites))
        # follow ELCFS, 1/8 length of train data
        tar_freq_path = self.amp_paths[site_idx][np.random.randint(self.min_len//8)]
        # print(tar_freq_path)
        img_format = tar_freq_path.split('.')[-1]
        tar_freq_path = tar_freq_path.replace(img_format, 'npy')#.replace('/research/dept5/mrjiang/', '/home/mrjiang/Server_mrjiang/')
        # print(tar_freq_path)
        tar_freq = np.load(os.path.join(tar_freq_path))
        return tar_freq

class Camelyon17_AMP():
    def __init__(self, args):
        print('Loading pacs amp lists...')
        self.src_sites = args.source
        self.amp_paths = []
        min_len = 1e8
        for dname in self.src_sites:
            name_train, labels_train, = _dataset_info(os.path.join(os.path.dirname(__file__), 'txt_lists', args.dataset, '%s_train.txt' % dname))
            self.amp_paths.append([p.replace('kfold', 'kfold_amp') for p in name_train])
            if min_len > len(name_train): min_len = len(name_train)

        # share_source_list = [ s for s in args.source if s!=dname]
        # amp_paths = [[p.replace('kfold', 'amps').replace(dname, sname) for p in names] for sname in share_source_list]
        # amp_paths.insert(0, [p.replace('kfold', 'amps') for p in names])
        self.min_len = min_len
        print('seen domain freqs: ', self.src_sites)
        

    def get_amp(self):
        site_idx = np.random.choice(len(self.src_sites))
        # print(site_idx)
        # follow ELCFS, 1/8 length of train data
        tar_freq_path = self.amp_paths[site_idx][np.random.randint(self.min_len//8)]
        # print(tar_freq_path)
        img_format = tar_freq_path.split('.')[-1]
        tar_freq_path = tar_freq_path.replace(img_format, 'npy')#.replace('/research/dept5/mrjiang/', '/home/mrjiang/Server_mrjiang/')
        # print(tar_freq_path)
        tar_freq = np.load(os.path.join(tar_freq_path))
        return tar_freq

class FedDGDataset(ImageDataset):
    def __init__(self, names, labels, img_transformer, amp_loader):
        super(FedDGDataset,self).__init__(names, labels, img_transformer)
        self.amp_loader = amp_loader
         
    def __getitem__(self, idx):
        # img = self.get_image(idx)
        framename = self.names[idx]#.replace('/research/dept5/mrjiang/', '/home/mrjiang/Server_mrjiang/')
        img = Image.open(framename).convert('RGB')
        # have to resize, ohterwise fft fails 
        img = img.resize( (222,222), Image.BICUBIC)
        # if self._image_transformer is not None:
        #     img = self._image_transformer(img)
        label = int(self.labels[idx])

        img_np = np.asarray(img, dtype=np.float32)
        tar_freq = self.amp_loader.get_amp()
        # tar_freq = np.load('/home/mrjiang/Server_mrjiang/Dataset/PACS/kfold_amp/art_painting/dog/pic_001.npy')
        # tar_freq = np.load('/home/mrjiang/Server_mrjiang/Dataset/OfficeHome/amp/art/Alarm_Clock/00001.npy')
        
        image_tar_freq = source_to_target_freq(img_np, tar_freq[:3,...], L=0,ratio=1.0)
        image_tar_freq = np.clip(image_tar_freq, 0, 255)

        image_tar_freq = Image.fromarray(image_tar_freq.astype(np.uint8))
        if self._image_transformer is not None:
            img = self._image_transformer(img)
            image_tar_freq = self._image_transformer(image_tar_freq)


        return img, image_tar_freq, int(label)
