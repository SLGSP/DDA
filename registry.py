from datafree.models import classifiers
from torchvision import datasets, transforms as T
from datafree.utils import sync_transforms as sT

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


import os
import torch
import torchvision
import datafree
import torch.nn as nn 
from PIL import Image

NORMALIZE_DICT = {
    'svhn': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    
}


MODEL_DICT = {
    'wrn16_1': classifiers.wresnet.wrn_16_1,
    'wrn16_2': classifiers.wresnet.wrn_16_2,
    'wrn40_1': classifiers.wresnet.wrn_40_1,
    'wrn40_2': classifiers.wresnet.wrn_40_2,
    
    'resnet50':  classifiers.resnet.resnet50,
    'resnet18':  classifiers.resnet.resnet18,
    'resnet34':  classifiers.resnet.resnet34,

    'vgg8': classifiers.vgg.vgg8_bn,
    'vgg11': classifiers.vgg.vgg11_bn,
    'vgg13': classifiers.vgg.vgg13_bn,

    'mobilenetv2': classifiers.mobilenetv2.mobile_half,


}


def get_model(name: str, num_classes, pretrained=False, **kwargs):
    model = MODEL_DICT[name](num_classes=num_classes)
    return model 


def get_dataset(name: str, data_root: str='data', return_transform=False, split=['A', 'B', 'C', 'D']):
    name = name.lower()
    data_root = os.path.expanduser( data_root )

    if name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'CIFAR10' ) 
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)
    elif name=='cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'CIFAR100' ) 
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
    elif name=='svhn':
        num_classes = 10
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'SVHN' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, train_transform, val_transform
    return num_classes, train_dst, val_dst
