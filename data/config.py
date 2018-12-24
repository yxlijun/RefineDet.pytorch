# config.py
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os.path
from easydict import EasyDict

# gets home dir cross platform
#HOME = os.path.expanduser("~")

_C = EasyDict()
cfg = _C

_C.HOME = '/home/lj/data/'
_C.VOC_ROOT = os.path.join(_C.HOME, "VOCdevkit/")

# for making bounding boxes pretty
_C.COLORS = [[0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],[128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]]

_C.MEANS = (104, 117, 123)


_C.VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

_C.LABEL_MAP = dict(zip([str(x)
                     for x in range(len(_C.VOC_CLASSES))], _C.VOC_CLASSES))


# train config
_C.EPOCHES = 300
_C.LR_STEPS = (80000, 100000, 120000)
_C.MAX_STEPS = 150000


# anchor config
_C.SIZE320 = EasyDict()
_C.FEATURE_MAPS = [40, 20, 10, 5]
_C.INPUT_SIZE = 320
_C.STEPS = [8, 16, 32, 64]
_C.MIN_SIZES = [32, 64, 128, 256]
_C.MAX_SIZES = [64, 128, 256, 315]
_C.ASPECT_RATIOS = [[2], [2], [2], [2]]
_C.VARIANCE = [0.1, 0.2]
_C.CLIP = True
_C.NAME = 'VOC'


# loss config
_C.NUM_CLASSES = 21
_C.OVERLAP_THRESH = 0.5
_C.NEG_POS_RATIOS = 3


## detection config 
_C.NMS_THRESH=0.45
_C.NMS_TOP_K=1000
_C.KEEP_TOP_K = 500
_C.CONF_THRESH=0.01
