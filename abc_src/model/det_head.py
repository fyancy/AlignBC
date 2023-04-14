import numpy as np
import torch
from torch import nn

from model.head.det_predictor import make_predictor
from model.head.det_loss import make_loss_evaluator
from model.head.det_infer import make_post_processor
from model.head.det_corner import Discrepancy_Computation

from utils.nms import ext_soft_nms, cpu_soft_nms
from utils.nms_ext import soft_nms_ext, soft_nms_bev, soft_nms_3d
import time


class Detect_Head(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Detect_Head, self).__init__()

        self.predictor = nn.ModuleList([make_predictor(cfg, in_channels, 1),
                                        make_predictor(cfg, in_channels, 2)])
        self.loss_evaluator = [make_loss_evaluator(cfg, 1), make_loss_evaluator(cfg, 2)]
        self.post_processor = [make_post_processor(cfg, 1), make_post_processor(cfg, 2)]
        self.discrepancy = Discrepancy_Computation(cfg)
        self.merge_fn = aggregate
        self.iou_method = cfg.TEST.IOU_METHOD
        # print('cfg iou method:', cfg.TEST.IOU_METHOD)

    def forward(self, features, targets=None, test=False,
                dis_only=False, single_det=False, nms_time=False):

        x1 = self.predictor[0](features[0], targets)
        x2 = self.predictor[1](features[1], targets)

        if self.training:
            if not dis_only:
                loss_dict, log_loss_dict = {}, {}
                loss_dict1, log_loss_dict1 = self.loss_evaluator[0](x1, targets)
                loss_dict2, log_loss_dict2 = self.loss_evaluator[1](x2, targets)
                for key, value in loss_dict1.items():
                    loss_dict[key] = loss_dict1[key] + loss_dict2[key]
                for key, value in log_loss_dict1.items():
                    log_loss_dict[key] = log_loss_dict1[key] + log_loss_dict2[key]
                return loss_dict, log_loss_dict, log_loss_dict1, log_loss_dict2
            else:
                loss_dict, log_loss_dict = self.discrepancy(x1, x2, targets)
                return loss_dict, log_loss_dict
        else:
            result1 = self.post_processor[0](x1, targets, test=test)
            result2 = self.post_processor[1](x2, targets, test=test)
            t0 = time.time()
            result = self.merge_fn(results=[result1, result2], method=self.iou_method)
            t1 = time.time()
            if single_det:
                return result, result1, result2
            else:
                if nms_time:
                    return result, t1-t0
                else:
                    return result


def aggregate(results: list, method='ext'):
    results = torch.cat(results, 0).cpu().numpy()  # [N, 14]
    box2d = results[:, 2:6]
    dims = results[:, 6:9]  # h, w, l
    dims = dims[:, [2, 0, 1]]  # l, h, w
    locs = results[:, 9:12]
    rys = results[:, [12]]
    scores = results[:, [13]]
    if method == "ext":
        boxes = np.concatenate([box2d, scores, locs[:, [2]], locs[:, [0]]], axis=1)
        keep = soft_nms_ext(box=boxes, sigma=0.5, gamma=2, Nt=0.3, threshold=0.2, method=2)
    elif method == "bev":
        boxes = np.concatenate([dims, locs, rys, scores], axis=1)
        keep = soft_nms_bev(box=boxes, sigma=0.5, Nt=0.3, threshold=0.2, method=2)
    elif method == "3d":
        boxes = np.concatenate([dims, locs, rys, scores], axis=1)
        keep = soft_nms_3d(box=boxes, sigma=0.5, Nt=0.3, threshold=0.2, method=2)
        # 因为3d-iou比较小，weight_decay很小，衰减后的score仍很大，需要提高threshold
    elif method == "ext_cpy":
        boxes = np.concatenate([box2d, scores, locs[:, [2]], locs[:, [0]]], axis=1)
        keep = ext_soft_nms(box=boxes, sigma=0.5, gamma=2, Nt=0.3, threshold=0.2, method=2)  # thresh, 0.45-0.7
    else:
        exit(f"Method Error. Got {method}")
    results = results[keep]
    return torch.from_numpy(results)


def build_head(cfg, in_channels):
    return Detect_Head(cfg, in_channels)
