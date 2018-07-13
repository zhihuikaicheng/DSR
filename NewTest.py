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
from utils.utils import to_scalar
from utils.utils import load_state_dict
from utils.utils import load_ckpt
from torch.nn.parallel import DataParallel

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

args = parser.parse_args()

data_transforms = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {x: datasets.ImageFolder(os.path.join(args.test_dir, x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}

class_names = image_datasets['query'].classes


def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

model = Model() #last_conv_stride=args.last_conv_stride
model = load_network(model)
model_w = DataParallel(model)

# def fliplr(img):
#     '''flip horizontal'''
#     inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
#     img_flip = img.index_select(3,inv_idx)
#     return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    special_features = torch.FloatTensor()
    # count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        
        input_img = Variable(img.cuda())
        f, sf = model(input_img).data.cpu()
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

        features = torch.cat((features, f), 0)
        special_features = torch.cat((special_features, sf), 0)
    return features, special_features

gallery_feature = extract_feature(model,dataloaders['gallery'])
query_feature = extract_feature(model,dataloaders['query'])

def get_id(img_path):
    labels = []
    for path, _ in img_path:
        filename = path.split('_')[-1]
        label = filename[0:4]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
    return labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_label = get_id(gallery_path)
query_label = get_id(query_path)

# Save to Matlab for check
result_f = {'gallery_f':gallery_feature[0].numpy(),'gallery_label':gallery_label,'query_f':query_feature[0].numpy(),'query_label':query_label}
scipy.io.savemat('pytorch_result.mat',result_f)

result_sf = {'gallery_f':gallery_feature[1].numpy(),'gallery_label':gallery_label,'query_f':query_feature[1].numpy(),'query_label':query_label}
scipy.io.savemat('pytorch_result_multiscale.mat',result_sf)

# class ExtractFeature(object):
#   """A function to be called in the val/test set, to extract features.
#   Args:
#     TVT: A callable to transfer images to specific device.
#   """

#   def __init__(self, model, TVT):
#     self.model = model
#     self.TVT = TVT

#   def __call__(self, ims):
#     old_train_eval_model = self.model.training
#     # Set eval mode.
#     # Force all BN layers to use global mean and variance, also disable
#     # dropout.
#     self.model.eval()
#     ims = Variable(self.TVT(torch.from_numpy(ims).float()))
#     feat, spatialFeature = self.model(ims)
#     feat = feat.data.cpu().numpy()
#     #feat1 = feat1.data.cpu().numpy()
#     spatialFeature = spatialFeature.data.cpu().numpy()
#     # Restore the model to its old train/eval mode.
#     self.model.train(old_train_eval_model)
#     return feat, spatialFeature

# def test(load_model_weight=False):
#     if load_model_weight:
#         if args.model_save_dir != '':
#             map_location = (lambda storage, loc: storage)
#             sd = torch.load(args.model_save_dir, map_location=map_location)
#             load_state_dict(model, sd)
#             print('Loaded model weights from {}'.format(args.model_save_dir))
#         else:
#             load_ckpt(modules_optims, args.test_dir)

#     for test_set, name in zip(test_sets, test_set_names):
#         test_set.set_feat_func(ExtractFeature(model_w, TVT))
#         print('\n=========> Test on dataset: {} <=========\n'.format(name))
#         test_set.eval(
#             normalize_feat=False, #normalize_feat=cfg.normalize_feature
#             verbose=True)
