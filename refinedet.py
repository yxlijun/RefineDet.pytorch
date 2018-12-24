from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

from data.config import cfg
from layers import *

from torch.autograd import Variable


class ARM(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, base,  extras, head):
        super(ARM, self).__init__()

        self.vgg = nn.ModuleList(base)
        self.L2Norm4_3 = L2Norm(512, 10)
        self.L2Norm5_3 = L2Norm(512, 8)

        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        for k in xrange(23):
            x = self.vgg[k](x)
        s = self.L2Norm4_3(x)
        sources.append(s)

        for k in range(23, 30):
            x = self.vgg[k](x)
        s = self.L2Norm5_3(x)
        sources.append(s)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return sources, loc, conf


class RefineDet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, armhead, odmhead, num_classes):
        super(RefineDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.armnet = ARM(base, extras, armhead)

        self.odm_loc = nn.ModuleList(odmhead[0])
        self.odm_conf = nn.ModuleList(odmhead[1])

        self.latlayer6_1 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1)
        self.latlayer6_2 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1)
        self.smoothlayer6_3 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1)

        self.upsample1 = nn.ConvTranspose2d(
            512, 1024, kernel_size=4, stride=2, padding=1)
        self.latlayerfc7_1 = nn.Conv2d(
            1024, 1024, kernel_size=3, stride=1, padding=1)
        self.latlayerfc7_2 = nn.Conv2d(
            1024, 1024, kernel_size=3, stride=1, padding=1)
        self.smoothlayerfc7_3 = nn.Conv2d(
            1024, 1024, kernel_size=3, stride=1, padding=1)

        self.upsample2 = nn.ConvTranspose2d(
            1024, 512, kernel_size=4, stride=2, padding=1)
        self.latlayer5_1 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1)
        self.latlayer5_2 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1)
        self.smoothlayer5_3 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1)

        self.upsample3 = nn.ConvTranspose2d(
            512, 512, kernel_size=4, stride=2, padding=1)
        self.latlayer4_1 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1)
        self.latlayer4_2 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1)
        self.smoothlayer4_3 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1)

        if self.phase == 'test':
            self.detect = Detect(cfg)

    def forward(self, x):
        size = x.size()[2:]
        arm_sources, arm_loc, arm_conf = self.armnet(x)

        conv4, conv5, convfc7, conv6 = arm_sources

        odm_loc = list()
        odm_conf = list()

        x = F.relu(self.latlayer6_1(conv6), inplace=True)
        x = F.relu(self.latlayer6_2(x), inplace=True)
        efconv6 = F.relu(self.smoothlayer6_3(x), inplace=True)

        upsamling = self.upsample1(x)
        x = F.relu(self.latlayerfc7_1(convfc7), inplace=True)
        x = self.latlayerfc7_2(x)
        x = F.relu(x + upsamling, inplace=True)
        efconvfc7 = F.relu(self.smoothlayerfc7_3(x))

        upsamling = self.upsample2(x)
        x = F.relu(self.latlayer5_1(conv5), inplace=True)
        x = self.latlayer5_2(x)
        x = F.relu(x + upsamling, inplace=True)
        efconv5 = F.relu(self.smoothlayer5_3(x))

        upsamling = self.upsample3(x)
        x = F.relu(self.latlayer4_1(conv4), inplace=True)
        x = self.latlayer4_2(x)
        x = F.relu(x + upsamling)
        efconv4 = F.relu(self.smoothlayer4_3(x), inplace=True)

        sources = (efconv4, efconv5, efconvfc7, efconv6)

        for (x, l, c) in zip(sources, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        feature_maps = []
        for x in odm_loc:
            feature_maps += [[x.size(1), x.size(2)]]

        self.priorbox = PriorBox(size, feature_maps, cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)

        output = (
            arm_loc.view(arm_loc.size(0), -1, 4),
            arm_conf.view(arm_conf.size(0), -1, 2),
            odm_loc.view(odm_loc.size(0), -1, 4),
            odm_conf.view(odm_conf.size(0), -1, self.num_classes),
            self.priors
        )
        if self.phase == 'test':
            return self.detect.forward(output)

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            iteration = mdata['iteration']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return iteration

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def arm_multibox(vgg, extra_layers, cfg, num_classes=2):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, 28, -2]
    extra_source = [-1]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_source):
        loc_layers += [nn.Conv2d(extra_layers[v].out_channels, cfg[-1]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(extra_layers[v].out_channels, cfg[-1]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


def odm_multibox(vgg, extra_layers, cfg, num_classes=21):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, 28, -2]
    extra_source = [-1]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_source):
        loc_layers += [nn.Conv2d(extra_layers[v].out_channels, cfg[-1]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(extra_layers[v].out_channels, cfg[-1]
                                  * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512]

extras = [256, 'S', 512]

mbox = [4, 4, 4, 4]


def build_net(phase, num_classes=21):
    base_, extras_, armhead_ = arm_multibox(
        vgg(base, 3), add_extras(extras, 1024), mbox, 2)

    odmhead_ = odm_multibox(vgg(base, 3), add_extras(
        extras, 1024), mbox, num_classes)

    return RefineDet(phase, base_,  extras_, armhead_, odmhead_, num_classes)


if __name__ == '__main__':
    net = build_net('train', num_classes=21)
    inputs = Variable(torch.randn(1, 3, 320, 320))
    output = net(inputs)
