import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# from .resnet import resnet50
from torchvision import models
import pdb

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

class Model(nn.Module):
  def __init__(self, last_conv_stride=2):
    super(Model, self).__init__()
    # self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)
    self.base = models.resnet50(pretrained=True)
    # classifier = []
    # classifier += [nn.Linear(2048, 751)]
    # classifier = nn.Sequential(*classifier)
    # classifier.apply(weights_init_classifier)
    # self.classifier = classifier
    # self.base.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.AvgPool1 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    self.AvgPool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
    self.AvgPool3 = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
    self.AvgPool4 = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
    # self.AvgPool5 = nn.AvgPool2d((16, 8), stride=16)
    #self.conv1 = nn.Conv2d(2048, 256, 1, 1, 0)
    # self.fc1 = nn.
    # self.bn1 = nn.BatchNorm2d(2048)
    # self.relu = nn.ReLU(inplace=True)
    # self.fc = nn.Linear(2048, 751)

  def forward(self, x):
    # shape [N, C, H, W]
    x = self.base.conv1(x)
    x = self.base.bn1(x)
    x = self.base.relu(x)
    x = self.base.maxpool(x)
    x = self.base.layer1(x)
    x = self.base.layer2(x)
    x = self.base.layer3(x)
    x = self.base.layer4(x)
    # x = self.base.avgpool(x)
    # x = self.base(x)

    #logits = self.fc(feature)
    x1 = self.AvgPool1(x)
    x2 = self.AvgPool2(x)
    # x3 = self.AvgPool3(x)
    # x4 = self.AvgPool4(x)
    # x5 = self.AvgPool5(x)
    # x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
    x1 = x1.view(x1.size(0), x1.size(1), x1.size(2) * x1.size(3))
    x2 = x2.view(x2.size(0), x2.size(1), x2.size(2) * x2.size(3))
    # x3 = x3.view(x3.size(0), x3.size(1), x3.size(2) * x3.size(3))
    # x4 = x4.view(x4.size(0), x4.size(1), x4.size(2) * x4.size(3))
    # x5 = x5.view(x5.size(0), x5.size(1), x5.size(2) * x5.size(3))

    x = F.avg_pool2d(x, x.size()[2:])
    x = x.view(x.size(0), -1)

    feature = x

    # x = self.classifier(x)

    spatialFeature = torch.cat((x, x1, x2), 2)

    return feature, spatialFeature
    # return feature
    # return x, feature
