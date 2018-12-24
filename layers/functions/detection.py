from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch

import torch.nn.functional as F
from torch.autograd import Function
from ..box_utils import decode, nms, center_size


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self,cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.KEEP_TOP_K
        # Parameters used in nms.
        self.nms_thresh = cfg.NMS_THRESH
        self.nms_top_k = cfg.NMS_TOP_K
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE

    def forward(self, predictions):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        arm_loc, arm_conf, loc, conf, priors = predictions
        priors = priors.cuda()
        arm_conf = F.softmax(arm_conf.view(-1, 2), dim=1)
        conf = F.softmax(conf.view(-1, self.num_classes), dim=1)
        arm_loc_data = arm_loc.data
        arm_conf_data = arm_conf.data
        arm_object_conf = arm_conf_data[:, 1:]
        no_object_index = arm_object_conf <= 0.01
        conf.data[no_object_index.expand_as(conf.data)] = 0

        loc_data = loc.data
        conf_data = conf.data
        prior_data = priors.data[:loc_data.size(1), :]

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)

        batch_prior = prior_data.view(-1, num_priors,
                                      4).expand(num, num_priors, 4)
        batch_prior = batch_prior.contiguous().view(-1, 4)

        default = decode(arm_loc_data.view(-1, 4), batch_prior, self.variance)
        default = center_size(default)
        decoded_boxes = decode(loc_data.view(-1, 4), default, self.variance)

        conf_preds = conf_data.view(num, num_priors, self.num_classes)

        scores = conf_preds.view(
            num, num_priors, self.num_classes).transpose(2, 1)
        
        boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        
        for k in range(boxes.size(0)):
            decoder_boxes = boxes[k].clone()
            scores_ = scores[k].clone()
            for j in range(1, self.num_classes):
                c_mask = scores_[j].gt(self.conf_thresh)
                score_ = scores_[j][c_mask]
                if score_.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoder_boxes)
                boxes_ = decoder_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes_, score_,self.nms_thresh, self.nms_top_k)
                count = count if count<self.top_k else self.top_k
                output[k, j, :count] = torch.cat((score_[ids[:count]].unsqueeze(1),
                                                  boxes_[ids[:count]]), 1)

        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        return output
