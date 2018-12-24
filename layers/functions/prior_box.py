from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """docstring for PriorBox"""

    def __init__(self, input_size, feature_maps, cfg):
        super(PriorBox, self).__init__()
        self.imh = input_size[0]
        self.imw = input_size[1]

        self.variance = cfg.VARIANCE or [0.1]
        self.num_priors = len(cfg.ASPECT_RATIOS)

        self.min_sizes = cfg.MIN_SIZES
        self.max_sizes = cfg.MAX_SIZES
        self.steps = cfg.STEPS
        self.aspect_ratios = cfg.ASPECT_RATIOS
        self.clip = cfg.CLIP
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        self.feature_maps = feature_maps

    def forward(self):
        mean = []
        for k in range(len(self.feature_maps)):
            feath = self.feature_maps[k][0]
            featw = self.feature_maps[k][1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh
                mean += [cx, cy, s_kw, s_kh]
                
                s_k_primew = sqrt(s_kw * (self.max_sizes[k] / self.imw))
                s_k_primeh = sqrt(s_kh * (self.max_sizes[k] / self.imh))
                mean += [cx, cy, s_k_primew, s_k_primeh]

                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_kw * sqrt(ar), s_kh / sqrt(ar)]
                    mean += [cx, cy, s_kw / sqrt(ar), s_kh * sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    feature_maps = [[40, 40], [20, 20], [10, 10], [5, 5]]
    from data.config import cfg 
    p = PriorBox([320, 320], feature_maps,cfg)
    out = p.forward()
    print(out.size())
