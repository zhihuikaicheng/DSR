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
from torchvision import transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import json
from utils.Dataset import Dataset
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
parser.add_argument('--useCAM', type=bool, default=False)

args = parser.parse_args()

data_transforms = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# image_datasets = {x: datasets.ImageFolder(os.path.join(args.test_dir, x) ,data_transforms) for x in ['gallery','query']}
image_datasets = {x: Dataset(os.path.join(args.test_dir, x), data_transforms, CAM=args.useCAM) for x in ['gallery','query']}
# labelsloader = {x: iter(image_datasets[x].imgs) for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=False, num_workers=4) for x in ['gallery','query']}

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

def save_feature(part, features, special_features, labels, cams=None, Is_gallery=True):
    part_feat = 'feat'
    part_label = 'label'
    part_cam = 'cam'

    if cams is not None:
        result_f = {part_feat: features.numpy(), part_label: labels, part_cam: cams}
        result_sf = {part_feat: special_features.numpy(), part_label: labels, part_cam: cams}
    else:
        result_f = {part_feat: features.numpy(), part_label: labels}
        result_sf = {part_feat: special_features.numpy(), part_label: labels}

    pdb.set_trace()

    if Is_gallery:
        scipy.io.savemat(os.path.join(args.gallery_feature_dir, 'pytorch_result_gallery_{:d}.mat'.format(part)), result_f)
        scipy.io.savemat(os.path.join(args.gallery_feature_dir, 'pytorch_result_gallery_multi_{:d}.mat'.format(part)), result_sf)
    else:
        scipy.io.savemat(os.path.join(args.query_feature_dir, 'pytorch_result_query_{:d}.mat'.format(part)),result_f)
        scipy.io.savemat(os.path.join(args.query_feature_dir, 'pytorch_result_query_multi_{:d}.mat'.format(part)),result_sf)

def extract_feature(model, dataloaders, Is_gallery=True, useCAM=False):
    features = []
    special_features = []
    labels = []
    if useCAM:
        cams = []
    count = 0
    for data in dataloaders:
        count += 1
        if useCAM:
            img, _, label, cam = data
        else:
            img, _, label = data
        
        # n, c, h, w = img.size()
        
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

        # for i in range(args.batch_size):
        #     lab = next(labelsloader)
        #     label = get_id(lab)
        #     labels.append(label)
        
        features.append(f.data.cpu())
        special_features.append(sf.data.cpu())
        labels.append(label)
        if useCAM:
            cams.append(cam)

        if (count % 10 == 0):
            print(count * args.batch_size)
            part = int (count / 100)
            features = torch.cat(features, 0)
            special_features = torch.cat(special_features, 0)
            labels = torch.cat(labels, 0)
            if useCAM:
                cams = torch.cat(cams, 0)
                save_feature(part, features, special_features, labels, cams, Is_gallery=Is_gallery)
            else:
                save_feature(part, features, special_features, labels, Is_gallery=Is_gallery)

            features = []
            special_features = []
            labels = []
            if useCAM:
                cams = []

    part = int (count / 100 ) + 1
    features = torch.cat(features, 0)
    special_features = torch.cat(special_features, 0)
    labels = torch.cat(labels, 0)
    if useCAM:
        cams = torch.cat(cams, 0)
        save_feature(part, features, special_features, labels, cams, Is_gallery=Is_gallery)
    else:
        save_feature(part, features, special_features, labels, Is_gallery=Is_gallery)

    print(count * args.batch_size)
    return count

if not os.path.exists(args.gallery_feature_dir):
    os.makedirs(args.gallery_feature_dir)

if not os.path.exists(args.query_feature_dir):
    os.makedirs(args.query_feature_dir)

gallery_feature = extract_feature(model,dataloaders['gallery'], useCAM=args.useCAM)
query_feature = extract_feature(model,dataloaders['query'], Is_gallery=False, useCAM=args.useCAM)

