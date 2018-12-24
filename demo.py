#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import torch
import cv2
import torch.backends.cudnn as cudnn
import argparse

import os
import os.path as osp
from data.config import cfg
from refinedet import build_net
from torch.autograd import Variable
from utils.augmentations import RefineBasicTransform

import time


parser = argparse.ArgumentParser(description='refinedet demo')
parser.add_argument('--save_dir', type=str, default='tmp/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='weights/refine320_voc_240000.pth', help='trained model')
parser.add_argument('--thresh', default=0.5, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def test_net(net, imagelist, transform, thresh):
    for image_path in imagelist:
        img = cv2.imread(image_path)
        image = img.copy()
        img = transform(img)[0]
        img = img[:, :, (2, 1, 0)]
        x = torch.from_numpy(img).permute(2, 0, 1)

        x = Variable(x.unsqueeze(0))
        if use_cuda:
            x = x.cuda()

        scale = torch.Tensor([image.shape[1], image.shape[0],
                              image.shape[1], image.shape[0]])
        t1 = time.time()
        detections = net(x)      # forward pass
        print('detect:{} ------- time:{}'.format(image_path, time.time() - t1))
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                score = detections[0, i, j, 0]
                label_name = cfg.LABEL_MAP[str(i - 1)]
                color = cfg.COLORS[i - 1]

                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                left_up, right_bottom = (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3]))
                cv2.rectangle(image, left_up, right_bottom, color, 2)
                j += 1
                label = label_name + str(round(score, 2))
                text_size, baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

                p1 = (left_up[0], left_up[1] - text_size[1])
                cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                              (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
                cv2.putText(image, label, (p1[0], p1[
                            1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

            #cv2.imwrite('test.jpg', image)
        cv2.imwrite(osp.join(args.save_dir, osp.basename(image_path)), image)


if __name__ == '__main__':
    net = build_net('test', cfg.NUM_CLASSES)  # initialize SSD
    net.load_state_dict(torch.load(args.model))
    net.eval()
    print('Finished loading model!')

    if use_cuda:
        net = net.cuda()
        cudnn.benchmark = True

    image_path = './img'
    images = [osp.join(image_path, x)
              for x in os.listdir(image_path) if x.endswith('.jpg')]

    test_net(net, images, RefineBasicTransform(
        cfg.INPUT_SIZE, cfg.MEANS), args.thresh)
