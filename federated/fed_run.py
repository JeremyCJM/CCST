"""
Based on FedBN
"""
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
from data import data_helper
from data.data_helper import available_datasets
from utils import rsc_utils, rsc_utils_densenet
import random
from tensorboardX import SummaryWriter
from collections import OrderedDict
import math

# sys.path.append('/home/cjm/disk1/research/tent')
# import tent
# from conf import cfg

def train(model, train_loader, optimizer, loss_fun, client_num, device, args, iter_idx, logger):
    model.to(device)
    model.train()
    num_data = 0
    correct = 0
    class_correct = 0
    jig_correct = 0
    loss_all = 0

    for it, data in enumerate(train_loader):
        if args.dg_method.lower() == 'jigsaw':
            img, jig_l, class_l = data
            jig_l = jig_l.to(device)
        else:
            img, class_l = data

        img, class_l = img.to(device), class_l.to(device)

        optimizer.zero_grad()
        
        if args.dg_method == 'RSC':
            if 'resnet' in args.network.lower():
                class_logit = rsc_utils.update(img, class_l, args.n_classes, model, device)
            elif 'densenet' in args.network.lower():
                class_logit = rsc_utils_densenet.update(img, class_l, args.n_classes, model, device)
            loss = loss_fun(class_logit, class_l)
        elif args.dg_method.lower() == 'jigsaw':
            # import pdb; pdb.set_trace()
            class_logit, jig_logit = model(img)
            loss = loss_fun(class_logit, class_l) + args.jig_weight * loss_fun(jig_logit, jig_l)
            _, jig_pred = jig_logit.max(dim=1)
            jig_correct += torch.sum(jig_pred == jig_l.data)
        else:
            class_logit = model(img)
            loss = loss_fun(class_logit, class_l)

        _, cls_pred = class_logit.max(dim=1)

        loss_all += loss.item()

        class_correct += torch.sum(cls_pred == class_l.data)
        num_data += img.size(0)  # 32+32+...

        logger.log(it, len(train_loader),
                        {"train_loss": loss.item()},
                        {"class_acc": torch.sum(cls_pred == class_l.data).item(),},
                        img.shape[0])
        
        loss.backward()
        optimizer.step()
        del img, class_l
        
    train_loss = loss_all/(it+1)
    train_acc = (float(class_correct)/num_data)
    model.to('cpu')
    if args.dg_method.lower() == 'jigsaw':
        print(' Jigsaw acc: ', float(jig_correct)/num_data)
    return train_loss, train_acc   

def train_meta(model, train_loader, optimizer, loss_fun, client_num, device, args, iter_idx, logger):
    model.to(device)
    model.train()
    num_data = 0
    correct = 0
    class_correct = 0
    jig_correct = 0
    loss_all = 0
    
    for it, data in enumerate(train_loader): 
        # init_params = copy.deepcopy(model.state_dict())
        img, image_freq, class_l = data
        img, image_freq, class_l = img.to(device), image_freq.to(device), class_l.to(device)
        model.zero_grad()
        
        # Fast Adapt
        # output_inner = model(img, device=device)
        output_inner = model(img)
        loss_inner = loss_fun(output_inner, class_l)
        grads = torch.autograd.grad(loss_inner, model.parameters(), retain_graph=True)
        meta_step_size = args.meta_step_size
        clip_value = args.clip_value
        fast_weights = OrderedDict((name, param - torch.mul(meta_step_size, torch.clamp(grad, 0-clip_value, clip_value))) for
                                                ((name, param), grad) in
                                                zip(model.named_parameters(), grads))
        learner = copy.deepcopy(model)
        learner.load_state_dict(fast_weights, strict=False)

        output_outer = learner(image_freq)
        del fast_weights
        loss_outer = loss_fun(output_outer, class_l)

        loss = loss_inner + loss_outer 


        _, cls_pred = output_inner.max(dim=1)

        loss_all += loss.item()

        class_correct += torch.sum(cls_pred == class_l.data)
        num_data += img.size(0)  # 32+32+...

        logger.log(it, len(train_loader),
                        {"train_loss": loss.item()},
                        {"class_acc": torch.sum(cls_pred == class_l.data).item(),},
                        img.shape[0])

        if it % math.ceil(len(train_loader)*0.2) == 0:
                    print(' [Step-{}|{}]| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(it, len(train_loader), loss.item(), torch.sum(cls_pred == class_l.data).item()/ img.size(0)), end='\r')
        loss.backward()
        optimizer.step()
        
        
    train_loss = loss_all/(it+1)
    train_acc = (float(class_correct)/num_data)
    model.to('cpu')
    return train_loss, train_acc   

def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)
        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data


def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)
        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data


def test(model, test_loader, loss_fun, device, args):
    model.to(device)
    model.eval()

    if args.IN_test:
        def _set_module(model, submodule_key, module):
            tokens = submodule_key.split('.')
            sub_tokens = tokens[:-1]
            cur_mod = model
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            setattr(cur_mod, tokens[-1], module)

        for name,m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                IN = torch.nn.InstanceNorm2d(m.num_features, eps=m.eps, momentum=m.momentum, affine=True).to(device)
                IN.state_dict()['weight'].data.copy_(m.state_dict()['weight'])
                IN.state_dict()['bias'].data.copy_(m.state_dict()['bias'])
                _set_module(model, name, IN)


    class_correct = 0
    loss_all = 0
    num_data = 0
    
    # for it, ((data, jig_l, class_l), _) in enumerate(test_loader):
    for it, (data, class_l) in enumerate(test_loader):
        # print(data.shape)
        data, class_l = data.to(device), class_l.to(device)
        # class_logit = model(data, class_l, False)
        if args.dg_method.lower() == 'jigsaw':
            class_logit, _ = model(data)
        else:
            class_logit = model(data)
        loss = loss_fun(class_logit, class_l)
        loss_all += loss.item()
        
        _, cls_pred = class_logit.max(dim=1)
        class_correct += torch.sum(cls_pred == class_l.data)
        # print(f'it: {it}  acc: {torch.sum(cls_pred == class_l.data) / data.shape[0]}') # Debug
        num_data += data.size(0) 
        del data, class_l  
    class_acc = float(class_correct) / num_data
    test_loss = loss_all / (it+1)
    model.to('cpu')
    return test_loss, class_acc

def tent_test(model, test_loader, loss_fun, device, args):
    model.to(device)

    log_path = args.save_path.replace('checkpoint', 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = open(os.path.join(log_path,'{}.log'.format(args.mode+'_test')), 'a')
    logfile.write(log_path+'\n')
    logfile.flush()
    
    for epoch in range(10):
        class_correct = 0
        loss_all = 0
        num_data = 0

        model.train()
        for it, (data, class_l) in enumerate(test_loader):
            data, class_l = data.to(device), class_l.to(device)
            class_logit = model(data)
            loss = loss_fun(class_logit, class_l)
            loss_all += loss.item()
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            # print(f'epoch: {epoch} it: {it}  acc: {torch.sum(cls_pred == class_l.data) / data.shape[0]}') # Debug
            num_data += data.size(0) 
            del data, class_l  
        class_acc = float(class_correct) / num_data
        print(f'epoch {epoch}: Running accuracy is {class_acc}')
        logfile.write(f'epoch {epoch}: Running accuracy is {class_acc}\n')
        test_loss = loss_all / (it+1)

        class_correct = 0
        loss_all = 0
        num_data = 0
        model.eval()
        for it, (data, class_l) in enumerate(test_loader):
            data, class_l = data.to(device), class_l.to(device)
            class_logit = model(data)
            loss = loss_fun(class_logit, class_l)
            loss_all += loss.item()
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            num_data += data.size(0) 
            del data, class_l  
        class_acc = float(class_correct) / num_data
        print(f'epoch {epoch}: Last model accuracy is {class_acc}')
        logfile.write(f'epoch {epoch}: Last model accuracy is {class_acc}\n')
        test_loss = loss_all / (it+1)
        logfile.flush()

    model.to('cpu')
    logfile.close()
    return test_loss, class_acc

def tent_test_on_the_fly(model, test_loader, loss_fun, device, args):
    model.to(device)

    log_path = args.save_path.replace('checkpoint', 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = open(os.path.join(log_path,'{}.log'.format(args.mode+'_test')), 'a')
    logfile.write(log_path+'\n')
    logfile.flush()

    class_correct = 0
    loss_all = 0
    num_data = 0
    for batch_id, (data, class_l) in enumerate(test_loader):
        data, class_l = data.to(device), class_l.to(device)
        model.train()
        for times in range(10):
            class_logit = model(data)
        loss = loss_fun(class_logit, class_l)
        loss_all += loss.item()
        _, cls_pred = class_logit.max(dim=1)
        class_correct += torch.sum(cls_pred == class_l.data)
        num_data += data.size(0)
        del data, class_l  
        print(f'batch {batch_id}: Batch Accuracy is {torch.sum(cls_pred == class_l.data) / data.shape[0]}')

    
    class_acc = float(class_correct) / num_data
    print(f'Train on the fly accuracy: {class_acc}')

    model.to('cpu')
    logfile.close()
    return test_loss, class_acc


def test_fedbn(server_model, models, test_loader, loss_fun, device, args):
    client_num = len(models)
    client_weights = [float(1./client_num) for i in range(client_num)]

    with torch.no_grad():
        for key in models[0].state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                if 'bn' in key and 'num_batches_tracked' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
    server_model.eval()
    server_model.to(device)
    class_correct = 0
    loss_all = 0
    num_data = 0
    for it, (data, class_l) in enumerate(test_loader):
        data, class_l = data.to(device), class_l.to(device)
        class_logit = server_model(data)
        loss = loss_fun(class_logit, class_l)
        loss_all += loss.item()
        
        _, cls_pred = class_logit.max(dim=1)
        class_correct += torch.sum(cls_pred == class_l.data)
        num_data += data.size(0) 
        
    class_acc = float(class_correct) / num_data
    test_loss = loss_all / (it+1)
    server_model.to('cpu')
    return test_loss, class_acc

        
################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    if 'bn' not in key:                    
                        for client_idx in range(len(client_weights)):
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'fedavg' or 'fedprox':
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    # Aggregate client models as server model
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    # Copy aggregated server model to client models
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'adafea':
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                elif key.endswith('bn3.weight') or key.endswith('bn3.bias'):
                    # weight and bias is appeared before mean and var in stat_dict
                    # Aggregate client models as server model
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                elif key.endswith('bn3.running_var'):
                    # Aggregate client models as server model
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    ###
                    for client_idx in range(len(client_weights)):
                        var = server_model.state_dict()[key]
                        models[client_idx].state_dict()[key.replace('running_var', 'weight')].data.copy_(torch.sqrt(var+1e-05))
                elif key.endswith('bn3.running_mean'):
                    # Aggregate client models as server model
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    ### 
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key.replace('running_mean','bias')].data.copy_(server_model.state_dict()[key])
                else:
                    # Aggregate client models as server model
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    # Copy aggregated server model to client models
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--tent_test', action='store_true', help ='test the pretrained model with the Tent test-time optimization')
    parser.add_argument('--tent_test_on-the-fly', action='store_true', help ='test the pretrained model with the Tent test-time optimization one by one')
    parser.add_argument('--IN_test', action='store_true', help ='test the pretrained model using IN with affine')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--iters', type = int, default=500, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedavg', choices = ['fedavg', 'fedbn', 'adafea', 'fedprox', 'deepall'], help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    # FedProx hyper params
    # parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    # Dataset setting
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument("--n_classes", "-c", type=int, default=10, help="Number of classes")
    parser.add_argument("--dataset", choices=['pacs', 'officehome', 'digitsfive', 'camelyon17'], default='pacs', help="Dataset, to indicate the txt path")
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    # Experiment setting
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument("--fusion_mode", choices=['no_fusion', 'adain-single-K1', 'adain-single-K2', 'adain-single-K3', 'adain-single-K4',
                                                    'adain-overall-K1', 'adain-overall-K2', 'adain-overall-K3', 'adain-overall-K4'], 
                        default='no_fusion', help="Data Fusion Modes")    
    parser.add_argument("--dg_method", choices=['no_DG', 'RSC', 'Jigsaw', 'MixStyle', 'feddg'], default='no_DG', help="DG methods combined with FL")
    parser.add_argument("--network", choices=nets_map.keys(), help="Which network to use", default="resnet")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")    
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument('--gpu', type = int, default=0, help = 'gpu device number')
    parser.add_argument('--seed', type = int, default=1, help = 'random seed number')
    parser.add_argument('--save_freq', type = int, default=1, help = 'how long to save a latest checkpoint')
    # Jigsaw hyper params
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--jig_weight", type=float, default=0.7, help="Weight for the jigsaw puzzle loss")
    # FedDG params
    parser.add_argument('--meta_step_size', type=float, default=1e-3, help='meta learning rate')
    parser.add_argument('--clip_value', type=float, default=1.0, help='gradient clip')
    

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= args.seed
    random.seed(a=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 

    print('Device:', device)
    exp_folder = f'{args.dataset}/{args.mode}_{args.fusion_mode}_{args.dg_method}_{args.network}_locIter{args.wk_iters}/Target_{args.target}_seed_{seed}'
    if args.net2:
        exp_folder = exp_folder.replace(args.network, f'{args.network}_net2')
    args.save_path = os.path.join(args.save_path, exp_folder)
    log = args.log
    if log:
        log_path = args.save_path.replace('checkpoint', 'logs')
        # import pdb; pdb.set_trace()
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
        logfile.write('    wk_iters: {}\n'.format(args.wk_iters))
        
        if args.tf_logger:
            writer = SummaryWriter(os.path.join(log_path))
        
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
   
    print("Building server's model...")
    server_model = get_network(args.network)(args,classes=args.n_classes)
    # import pdb;pdb.set_trace()
    if args.mode.lower() == 'adafea':
        if args.network == 'resnet50':
            for p in server_model.layer1[2].bn3.parameters():
                p.requires_grad = False
        else:
            print('Only implemented for Resnet50')
            raise NotImplementedError


    loss_fun = nn.CrossEntropyLoss()
    
    # prepare the data
    print("Preparing data...")
    train_loaders, val_loaders = data_helper.get_train_dataloader(args) # train, val loaders of source domains



    target_test_loader = data_helper.get_test_dataloader(args)  # test loader of target domain

    
    # name of each client dataset
    datasets = args.source  # ["cartoon", "photo", "sketch"]
    target_dataset = [args.target]  # only one "art_painting"
    print("Source domain (for training) is {}".format(datasets))
    print("Target domain (for testing) is {}".format(target_dataset))

    
    if not args.mode == 'deepall':
        # federated setting
        client_num = len(datasets)
    else:
        client_num = 1
    client_weights = [float(1./client_num) for i in range(client_num)]
    # models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    models = [copy.deepcopy(server_model) for idx in range(client_num)]

    ''' Test Only '''
    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
                # _, test_acc = test(models[client_idx], target_test_loader, loss_fun, device,args)
            _, test_acc = test_fedbn(server_model, models, target_test_loader, loss_fun, device,args)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[client_idx], test_acc))
        else:
            _, test_acc = test( server_model, target_test_loader, loss_fun, device, args)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(args.target, test_acc))
        exit(0)
    
    if args.tent_test or args.tent_test_on_the_fly:
        print('Loading snapshots...')
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        server_model = tent.configure_model(server_model)
        params, param_names = tent.collect_params(server_model)

        tent_optimizer = optim.SGD(params,
                                # lr=cfg.OPTIM.LR,
                                lr=1e-4,
                                momentum=cfg.OPTIM.MOMENTUM,
                                dampening=cfg.OPTIM.DAMPENING,
                                weight_decay=cfg.OPTIM.WD,
                                nesterov=cfg.OPTIM.NESTEROV)
        tented_model = tent.Tent(server_model, tent_optimizer)
        
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                # fedbn tent test: TODO
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
                # _, test_acc = test(models[client_idx], target_test_loader, loss_fun, device,args)
            _, test_acc = test_fedbn(tented_model, models, target_test_loader, loss_fun, device,args)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[client_idx], test_acc))
        else:
            _, test_acc = tent_test(tented_model, target_test_loader, loss_fun, device, args)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(args.target, test_acc))
        exit(0)


    ''' Resume ''' 
    if args.resume:
        checkpoint = torch.load(SAVE_PATH+'_latest')
        # checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0

    logger = Logger(args, update_frequency=30)
    results = {"val-local": torch.zeros(args.iters), \
                "val-global": torch.zeros(args.iters), \
                "test": torch.zeros(args.iters)}

    # start training
    best_val_class_acc = 0.
    for a_iter in range(resume_iter, args.iters):
        print("=============Global iter is {} ===============".format(a_iter))
        print("----------------Training----------------")
        if args.log: 
            logfile.write("=============Global iter is {} ===============\n".format(a_iter))
            logfile.write("----------------Training----------------\n")
            # running_data = dict()
            # running_data['a_iter'] = a_iter
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            iter_idx = wi + a_iter * args.wk_iters
            print("== Train epoch {} ===".format(iter_idx))
            if args.log: logfile.write("== Train epoch {} ===\n".format(iter_idx))
            
            for client_idx in range(client_num):
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_loss, train_acc = train_fedprox(args, models[client_idx], train_loaders[client_idx], optimizers[client_idx], loss_fun, client_num, device)
                    else:
                        train_loss, train_acc = train(models[client_idx], train_loaders[client_idx], optimizers[client_idx], loss_fun, client_num, device, args, iter_idx, logger)
                else:
                    if args.dg_method.lower() == 'feddg':
                        train_loss, train_acc = train_meta(models[client_idx], train_loaders[client_idx], optimizers[client_idx], loss_fun, client_num, device, args, iter_idx, logger)
                    else:
                        train_loss, train_acc = train(models[client_idx], train_loaders[client_idx], optimizers[client_idx], loss_fun, client_num, device, args, iter_idx, logger)
                print(' {:<11s}| Train Loss: {:.4f}'.format(datasets[client_idx], train_loss))
                print(' {:<11s}| Train Class Acc: {:.4f}'.format(datasets[client_idx], train_acc))
                if args.log:
                    logfile.write("Train Loss is {:.4f}\n".format(train_loss))
                    logfile.write("Train Class Accuracy is {:.4f}\n".format(train_acc))
                    # running_data['train_loss'] = train_loss
                    # running_data['train_acc_c'] = train_acc
                    logfile.flush()
        with torch.no_grad():
            # aggregation
            server_model, models = communication(args, server_model, models, client_weights)
                    
            # validate global model
            print("----------------Validate global model on source domains----------------")
            val_loss_average, val_class_acc_average, best_test, test_acc_ = 0,0,0,0
            for client_idx in range(client_num):
                val_loader = val_loaders[client_idx]
                # val_loader = train_loaders[client_idx]
                # print('use model',client_idx)
                if args.mode=='fedbn':
                    # FedBN test local model on source domains
                    val_loss, val_acc = test(models[client_idx], val_loader, loss_fun, device, args) 
                else:
                    val_loss, val_acc = test(server_model, val_loader, loss_fun, device, args) 
                print(' {:<11s}| Global Val Loss: {:.4f}'.format(datasets[client_idx], val_loss))
                print(' {:<11s}| Global Val Class Acc: {:.4f}'.format(datasets[client_idx], val_acc))
                val_loss_average += val_loss
                val_class_acc_average += val_acc
                if args.log:
                    logfile.write("----------------Validate global model on source domains----------------\n")
                    logfile.write(' {:<11s}| Global Val Loss: {:.4f}\n'.format(datasets[client_idx], train_loss))
                    logfile.write(' {:<11s}| Global Val Class Acc: {:.4f}\n'.format(datasets[client_idx], val_acc))



            # record
            val_loss_average /= client_num
            val_class_acc_average /= client_num
            if args.tf_logger:
                writer.add_scalar('val_class_acc_average', val_class_acc_average, a_iter)

            print("-------------Test server model on target domain testset----------------")
            # test_loss, test_acc = test(server_model, target_test_loader, loss_fun, device, args)
            test_loss, test_acc = test(server_model, target_test_loader, loss_fun, device, args)
            test_acc_ = test_acc
            print(' {:<11s}| Global Test Loss: {:.4f}'.format(target_dataset[0], test_loss))
            print(' {:<11s}| Global Test Class Acc: {:.4f}'.format(target_dataset[0], test_acc))
            if args.log:
                logfile.write("-------------Test server model on  target domain testset----------------\n")
                logfile.write(' {:<11s}| Global Test Loss: {:.4f}\n'.format(target_dataset[0], test_loss))
                logfile.write(' {:<11s}| Global Test Class Acc: {:.4f}\n'.format(target_dataset[0], test_acc))
                # running_data['test_loss'] = test_loss
                # running_data['test_c_acc'] = test_acc
            if args.tf_logger:
                writer.add_scalar('target_domain_test_acc', test_acc, a_iter) 


            

            # save latest model
            if a_iter % args.save_freq == 0 and a_iter > 0 : # 20
                if args.mode.lower() == 'fedbn':
                    model_dicts = {'server_model': server_model.state_dict(),
                                    'a_iter': a_iter}
                    for model_idx, model in enumerate(models):
                        model_dicts['model_{}'.format(model_idx)] = model.state_dict()    
                else:
                    model_dicts = {
                            'server_model': server_model.state_dict(),
                            'a_iter': a_iter
                        }

                torch.save(model_dicts, SAVE_PATH+'_latest')

            # save best model
            if val_class_acc_average > best_val_class_acc:
                best_val_class_acc = val_class_acc_average
                best_test = test_acc_
                print(' Saving current best checkpoints to {}...'.format(SAVE_PATH))
                if args.log:
                    logfile.write(' Saving current best checkpoints to {}...\n'.format(SAVE_PATH))
                if args.mode.lower() == 'fedbn':
                    model_dicts = {'server_model': server_model.state_dict(),
                                    'a_iter': a_iter}
                    for model_idx, model in enumerate(models):
                        model_dicts['model_{}'.format(model_idx)] = model.state_dict()
                    
                    torch.save(model_dicts, SAVE_PATH)
                else:
                    torch.save({
                            'server_model': server_model.state_dict(),
                            'a_iter': a_iter
                        }, SAVE_PATH)



    if log:
        logfile.write(f'Test result using the global model with best val accuracy: {best_test} on {args.target}')
        logfile.flush()
        logfile.close()
