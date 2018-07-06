from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
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

os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"

image_dir = '/world/data-gpu-94/sysu-reid/person-reid-data/OPPO_partial_dataset_raw/training/'

data_transform = transforms.Compose([  
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(image_dir), data_transform)

dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64,
                                             shuffle=True, num_workers=8)

dataset_sizes = len(image_datasets['train'])

class_names = image_datasets['train'].classes

print (len(class_names))

# pdb.set_trace()

inputs, classes = next(iter(dataloaders))

y_loss = []
y_err = []

model = ft_net(len(class_names))
model = model.cuda()

criterion = nn.CrossEntropyLoss()

ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))

base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.01},
             {'params': model.model.fc.parameters(), 'lr': 0.1},
             {'params': model.classifier.parameters(), 'lr': 0.1}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model', 'ft_ResNet50', save_filename)
    torch.save(network.cpu().state_dict(), save_path)


def train_model(model, critertion, optimizer, scheduler, num_epochs):

    model_weights = model.state_dict()

    best_acc = 0.0

    for epoch in range(num_epochs):
        print ('Now {} epochs, total {} epochs'.format(epoch, num_epochs))
        print ('*' * 12)

        scheduler.step()
        model.train(True)

        running_loss = 0.0
        running_corrects = 0

        for data in dataloaders:
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = critertion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects / dataset_sizes

        print ('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        y_loss.append(epoch_loss)
        y_err.append(1.0 - epoch_acc)

    # model.load_state_dict(model_weights)
    save_network(model, epoch)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=15)