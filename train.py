from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


from data.config import cfg
from layers.modules import MultiBoxLoss
from layers.functions import Detect
from refinedet import build_net
from data.voc0712 import VOCDetection, detection_collate
from utils.augmentations import RefineAugmentation, RefineBasicTransform


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset',
                    default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--basenet',
                    default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size',
                    default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=False, type=str2bool,
                    help='Use multi GPU training')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


train_dataset = VOCDetection(root=cfg.VOC_ROOT,
                             transform=RefineAugmentation(cfg.INPUT_SIZE, cfg.MEANS))

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True, collate_fn=detection_collate,
                               pin_memory=True)

val_batchsize = args.batch_size // 2
val_dataset = VOCDetection(root=cfg.VOC_ROOT,
                           image_sets=[('2007', 'test')],
                           transform=RefineBasicTransform(cfg.INPUT_SIZE, cfg.MEANS))

val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False, collate_fn=detection_collate,
                             pin_memory=True)

min_loss = np.inf
iteration  = 0

RefineDet = build_net('train', cfg.NUM_CLASSES)
net = RefineDet

for layer in net.modules():
    layer.apply(RefineDet.weights_init)

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    iteration = net.load_weights(args.resume)
else:
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    RefineDet.armnet.vgg.load_state_dict(vgg_weights)

if args.cuda:
    if args.multigpu:
        net = torch.nn.DataParallel(RefineDet)
    net = net.cuda()
    cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

criterion_arm = MultiBoxLoss(cfg, 2, args.cuda)
criterion_odm = MultiBoxLoss(cfg, cfg.NUM_CLASSES, args.cuda)

print('Loading the dataset...')

print('Training Refinedet on:', train_dataset.name)
print('Using the specified args:')
print(args)



def train():
    step_index = 0
    net.train()
    global iteration
    for epoch in xrange(cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets, _) in enumerate(train_loader):
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            arm_loss_l, arm_loss_c = criterion_arm(out, targets)

            odm_loss_l, odm_loss_c = criterion_odm(
                out, targets, use_arm=True, filter_object=True)

            loss = arm_loss_l + arm_loss_c + odm_loss_c + odm_loss_l

            loss.backward()
            optimizer.step()
            t1 = time.time()

            losses += loss.data[0]

            if iteration % 10 == 0:
                loss_ = losses / (batch_idx + 1)
                print('Timer: %.4f sec.' % (t1 - t0))
                print('->>epoch ' + repr(epoch) + ' || iter ' +
                      repr(iteration) + ' || Loss:%.4f' % (loss_))
                print('->>arm_L:{:.4f} || arm_C:{:.4f}'.format(
                    arm_loss_l.data[0], arm_loss_c.data[0]))
                print('->>odm_L:{:.4f} || odm_C:{:.4f}'.format(
                    odm_loss_l.data[0], odm_loss_c.data[0]))
                print('->>lr:{:.6f}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 5000 == 0:
                torch.save(RefineDet.state_dict(), os.path.join(args.save_folder, 'refine320_voc_' +
                                                                repr(iteration) + '.pth'))
            iteration += 1

        val(epoch)
        if(iteration==cfg.MAX_STEPS):
            break

def val(epoch):
    net.eval()
    losses = 0
    step = 0
    t1 = time.time()
    for batch_idx, (images, targets, _) in enumerate(val_loader):
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        out = net(images)
        arm_loss_l, arm_loss_c = criterion_arm(out, targets)
        odm_loss_l, odm_loss_c = criterion_odm(
            out, targets, use_arm=True, filter_object=True)

        loss = arm_loss_l + arm_loss_c + odm_loss_c + odm_loss_l
        losses += loss.data[0]
        step += 1

    tloss = losses / step

    t2 = time.time()
    print('Timer: %.4f sec.' % (t1 - t1))
    print('test epoch ' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    global iteration
    if tloss < min_loss:
        print('Saving best state')
        torch.save(RefineDet.state_dict(), os.path.join(
            args.save_folder, 'refine320_voc.pth'))
        min_loss = tloss

    states = {
        'iteration': iteration,
        'weight': RefineDet.state_dict(),
    }
    torch.save(states, os.path.join(
        args.save_folder, 'refine320_voc_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
