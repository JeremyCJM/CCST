import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F


# class RSC(object):
#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(RSC, self).__init__(input_shape, num_classes, num_domains,
#                                    hparams)
#         self.drop_f = (1 -1/3) * 100
#         self.drop_b = (1 - 1/3) * 100
#         self.num_classes = num_classes

def featurizer(x, model):
    '''for resnet only'''
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

def classifier(x, model):
    return model.class_classifier(x)

def update(x, y, num_classes, model, device):
    drop_f = (1 -1/3) * 100
    drop_b = (1 - 1/3) * 100
    num_classes = num_classes
    # inputs
    all_x = x
    # labels
    all_y = y
    # one-hot labels
    all_o = torch.nn.functional.one_hot(all_y, num_classes)
    # features
    all_f = featurizer(all_x, model)
    # predictions
    all_p = classifier(all_f, model)

    # Equation (1): compute gradients with respect to representation
    all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

    # Equation (2): compute top-gradient-percentile mask
    percentiles = np.percentile(all_g.cpu(), drop_f, axis=1)
    percentiles = torch.Tensor(percentiles)
    percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
    mask_f = all_g.lt(percentiles.to(device)).float()

    # Equation (3): mute top-gradient-percentile activations
    all_f_muted = all_f * mask_f

    # Equation (4): compute muted predictions
    all_p_muted = classifier(all_f_muted, model)

    # Section 3.3: Batch Percentage
    all_s = F.softmax(all_p, dim=1)
    all_s_muted = F.softmax(all_p_muted, dim=1)
    changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
    percentile = np.percentile(changes.detach().cpu(), drop_b)
    mask_b = changes.lt(percentile).float().view(-1, 1)
    mask = torch.logical_or(mask_f, mask_b).float()

    # Equations (3) and (4) again, this time mutting over examples
    all_p_muted_again = classifier(all_f * mask, model)

    # Equation (5): update
    # loss = F.cross_entropy(all_p_muted_again, all_y)
    return all_p_muted_again
