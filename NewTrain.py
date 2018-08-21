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
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import json
from utils.sampler import RandomIdentitySampler
from utils.resnet import resnet50
from utils.Model import Model
from utils.TripletLoss import TripletLoss
from utils.loss import global_loss
from utils.utils import AverageMeter
from utils.utils import set_devices
from utils.utils import to_scalar
from torch.nn.parallel import DataParallel

#args
##############################################
parser = argparse.ArgumentParser()
# parser.add_argument('--sys_device_ids', type=eval, default=(0,1,2,3))
parser.add_argument('--sys_device_ids', type=str, default='0')
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--num_epochs', type=int, default=60)
parser.add_argument('--lr_decay_epochs', type=int, default=40)
parser.add_argument('--steps_per_log', type=int, default=1)
parser.add_argument('--model_save_dir', type=str)
parser.add_argument('--img_h', type=int, default=256)
parser.add_argument('--img_w', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()

##############################################

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_use

#data input
##############################################
image_dir = args.dataset_dir

data_transform = transforms.Compose([
    transforms.Resize((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(image_dir), data_transform)

dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size,
                                            sampler=RandomIdentitySampler(image_datasets['train'].imgs),
                                            num_workers=8)

dataset_sizes = len(image_datasets['train'])

class_names = image_datasets['train'].classes

inputs, classes = next(iter(dataloaders))

##############################################

y_loss = []
y_err = []

# model = ft_net(len(class_names))
model = Model()

os.environ['CUDA_VISIBLE_DEVICES'] = args.sys_device_ids

# TVT, TMO = set_devices(args.sys_device_ids)

# criterion = nn.CrossEntropyLoss()
margin = args.margin

tri_loss = TripletLoss(margin)

#ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))

#base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
base_params = model.parameters()

optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.01}
             # {'params': model.model.fc.parameters(), 'lr': 0.1},
             # {'params': model.classifier.parameters(), 'lr': 0.1}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.lr_decay_epochs, gamma=0.1)

# model = DataParallel(model)

model = model.cuda()

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(args.model_save_dir, save_filename)
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    torch.save(network.cpu().state_dict(), save_path)


def train_model(model, optimizer, scheduler, num_epochs):

    model_weights = model.state_dict()

    best_acc = 0.0

    save_network(model, 0)

    st_time = time.time()

    for epoch in range(num_epochs):
        print ('Now {} epochs, total {} epochs'.format(epoch, num_epochs))
        print ('*' * 12)

        scheduler.step()
        model.cuda()
        model.train(True)

        running_loss = 0.0
        running_corrects = 0

        prec_meter = AverageMeter()
        sm_meter = AverageMeter()
        dist_ap_meter = AverageMeter()
        dist_an_meter = AverageMeter()
        loss_meter = AverageMeter()

        for data in dataloaders:
            inputs, labels = data
            inputs = Variable(inputs.float().cuda())
            labels = labels.long().cuda()

            # inputs = Variable(TVT(inputs.float()))
            # labels = TVT(labels.long())

            optimizer.zero_grad()

            outputs_x, outputs_spatialFeature = model(inputs) 

            # DSR AND TRIPLET LOSS ADD IN HERE
            #################################################
            # _, preds = torch.max(outputs.data, 1)
            # loss = critertion(outputs, labels)
            loss, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
                tri_loss, outputs_x, outputs_spatialFeature, labels,
                normalize_feature=False) 

            #################################################

            loss.backward()
            optimizer.step()

            prec = (dist_an > dist_ap).data.float().mean()
            # the proportion of triplets that satisfy margin
            sm = (dist_an > dist_ap + margin).data.float().mean()
            # average (anchor, positive) distance
            d_ap = dist_ap.data.mean()
            # average (anchor, negative) distance
            d_an = dist_an.data.mean()

            prec_meter.update(prec)
            sm_meter.update(sm)
            dist_ap_meter.update(d_ap)
            dist_an_meter.update(d_an)
            loss_meter.update(to_scalar(loss))
            # if step % args.steps_per_log == 0:
            #     time_log = '\tStep {}/Ep {}, {:.2f}s'.format(
            #       step, ep + 1, time.time() - step_st, )

            #     tri_log = (', prec {:.2%}, sm {:.2%}, '
            #            'd_ap {:.4f}, d_an {:.4f}, '
            #            'loss {:.4f}'.format(
            #     prec_meter.val, sm_meter.val,
            #     dist_ap_meter.val, dist_an_meter.val,
            #     loss_meter.val, ))

            #     log = time_log + tri_log
            #     print(log)

        #     running_loss += loss.data[0]
        #     running_corrects += torch.sum(preds == labels.data)

        # epoch_loss = running_loss / dataset_sizes
        # epoch_acc = running_corrects / dataset_sizes

        # print ('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # y_loss.append(epoch_loss)
        # y_err.append(1.0 - epoch_acc)
        time_log = 'Ep {}, {:.2f}s'.format(epoch + 1, time.time() - st_time)

        tri_log = (', prec {:.2%}, sm {:.2%}, '
               'd_ap {:.4f}, d_an {:.4f}, '
               'loss {:.4f}'.format(
        prec_meter.avg, sm_meter.avg,
        dist_ap_meter.avg, dist_an_meter.avg,
        loss_meter.avg, ))

        log = time_log + tri_log
        print(log)


        # model.load_state_dict(model_weights)
        if (epoch + 1) % 10 == 0:
            save_network(model, epoch)

    save_network(model, 'last')

model = train_model(model, optimizer_ft, exp_lr_scheduler,
                       args.num_epochs)