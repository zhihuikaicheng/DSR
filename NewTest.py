from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import time
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import ft_net, ft_net_dense, PCB
import json
from utils.sampler import RandomIdentitySampler
from utils.resnet import resnet50
from utils.Model import Model
from utils.TripletLoss import TripletLoss
from utils.loss import global_loss
from utils.utils import AverageMeter
from utils.utils import set_devices
from torch.nn.parallel import DataParallel
import scipy

#args
##############################################
parser = argparse.ArgumentParser()
parser.add_argument('--sys_device_ids', type=eval, default=(0,1,2,3))
parser.add_argument('--test_dir', type=str)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--model_save_dir', type=str)
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--img_h', type=int, default=256)
parser.add_argument('--img_w', type=int, default=256)
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--gallery_feature_dir', type=str)
parser.add_argument('--query_feature_dir', type=str)

args = parser.parse_args()

data_transforms = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {x: datasets.ImageFolder(os.path.join(args.test_dir, x) ,data_transforms) for x in ['gallery','query']}
labelsloader = {x: iter(image_datasets[x].imgs) for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=False, num_workers=4, drop_last=True) for x in ['gallery','query']}

class_names = image_datasets['query'].classes


def load_network(network):
    save_path = os.path.join(args.model_save_dir, 'net_%s.pth'%args.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

model = Model() #last_conv_stride=args.last_conv_stride
TVT, TMO = set_devices(args.sys_device_ids)
model = DataParallel(model)
model.cuda()
model = load_network(model)


# def fliplr(img):
#     '''flip horizontal'''
#     inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
#     img_flip = img.index_select(3,inv_idx)
#     return img_flip

# def get_id(img_path):
#     labels = []
#     for path, _ in img_path:
#         filename = path.split('_')[-1]
#         label = filename[0:4]
#         if label[0:2]=='-1':
#             labels.append(-1)
#         else:
#             labels.append(int(label))
#     return labels

def get_id(img_path):
    filename = img_path[0].split('_')[0]
    length = len(filename)
    label = ''
    for i in range(length):
        if filename[length-i-1]=='/':
            break
        label = filename[length-i-1] + label

    if label=='-1':
        return (-1)
    else:
        return int(label)

def extract_feature(model,dataloaders,labelsloader,Is_gallery=True):
    features = []
    special_features = []
    labels = []
    count = 0
    for data in dataloaders:
        # img, label = data
        count += 1
        img, _ = data
        
        n, c, h, w = img.size()
        
        input_img = Variable(TVT(img.float()))
        f, sf = model(input_img)
        # count += n

        # print(count)
        # ff = torch.FloatTensor(n,2048).zero_()
        # for i in range(2):
        #     if (i == 1):
        #         img = fliplr(img)
        #     input_img = Variable(img.cuda())
        #     outputs = model(input_img) 
        #     f = outputs.data.cpu()
        #     ff = ff + f

        #     fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        #     ff = ff.div(fnorm.expand_as(ff))

        for i in range(args.batch_size):
            lab = next(labelsloader)
            label = get_id(lab)
            labels.append(label)
        
        features.append(f.data.cpu())
        special_features.append(sf.data.cpu())

        if (count % 500 == 0):
            print(count * args.batch_size)
            part = int (count / 500)
            features = torch.cat(features, 0)
            special_features = torch.cat(special_features, 0)
            if (Is_gallery):
                result_f = {'gallery_f':features.numpy(),'gallery_label':labels}
                scipy.io.savemat(os.path.join(args.gallery_feature_dir, 'pytorch_result_gallery_{:d}.mat'.format(part)),result_f)
                result_sf = {'gallery_f':special_features.numpy(),'gallery_label':labels}
                scipy.io.savemat(os.path.join(args.gallery_feature_dir, 'pytorch_result_gallery_multi_{:d}.mat'.format(part)),result_sf)
            else:
                result_f = {'query_f':features.numpy(),'query_label':labels}
                scipy.io.savemat(os.path.join(args.query_feature_dir, 'pytorch_result_query_{:d}.mat'.format(part)),result_f)
                result_sf = {'gallery_f':special_features.numpy(),'gallery_label':labels}
                scipy.io.savemat(os.path.join(args.query_feature_dir, 'pytorch_result_query_multi_{:d}.mat'.format(part)),result_sf)
            features = []
            special_features = []
            labels = []

    part = int (count / 500 ) + 1
    features = torch.cat(features, 0)
    special_features = torch.cat(special_features, 0)
    if (Is_gallery):
        result_f = {'gallery_f':features.numpy(),'gallery_label':labels}
        scipy.io.savemat(os.path.join(args.gallery_feature_dir, 'pytorch_result_gallery_{:d}.mat'.format(part)),result_f)
        result_sf = {'gallery_f':special_features.numpy(),'gallery_label':labels}
        scipy.io.savemat(os.path.join(args.gallery_feature_dir, 'pytorch_result_gallery_multi_{:d}.mat'.format(part)),result_sf)
    else:
        result_f = {'query_f':features.numpy(),'query_label':labels}
        scipy.io.savemat(os.path.join(args.query_feature_dir, 'pytorch_result_query_{:d}.mat'.format(part)),result_f)
        result_sf = {'gallery_f':special_features.numpy(),'gallery_label':labels}
        scipy.io.savemat(os.path.join(args.query_feature_dir, 'pytorch_result_query_multi_{:d}.mat'.format(part)),result_sf)
        # features = torch.cat((features, f), 0)
        # special_features = torch.cat((special_features, sf), 0)
    return count

if not os.path.exist(args.gallery_feature_dir):
    os.makedirs(args.gallery_feature_dir)

if not os.path.exist(args.query_feature_dir):
    os.makedirs(args.query_feature_dir)
    
gallery_feature = extract_feature(model,dataloaders['gallery'],labelsloader['gallery'])
query_feature = extract_feature(model,dataloaders['query'],labelsloader['query'],Is_gallery=False)
# pdb.set_trace()

# gallery_path = image_datasets['gallery'].imgs
# query_path = image_datasets['query'].imgs

# gallery_label = get_id(gallery_path)
# query_label = get_id(query_path)

# Save to Matlab for check
# result_f = {'gallery_f':gallery_feature[0].numpy(),'gallery_label':gallery_label,'query_f':query_feature[0].numpy(),'query_label':query_label}
# scipy.io.savemat('pytorch_result.mat',result_f)

# result_sf = {'gallery_f':gallery_feature[1].numpy(),'gallery_label':gallery_label,'query_f':query_feature[1].numpy(),'query_label':query_label}
# scipy.io.savemat('pytorch_result_multiscale.mat',result_sf)
