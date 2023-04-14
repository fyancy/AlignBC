
from torch import nn

from structures.image_list import to_image_list

from model.backbone.dla import build_backbone
from model.det_head import build_head
from model.neck.neck import AdNeck

# from model.layers.uncert_wrapper import make_multitask_wrapper


class KeypointDetector(nn.Module):
    """
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    """

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.neck = AdNeck(cfg, self.backbone.out_channels)  # TODO: Neck? 参考工作站上的project: AdCenter
        self.heads = build_head(cfg, self.backbone.out_channels)
        self.test = True  # cfg.DATASETS.TEST_SPLIT == 'test'

    def forward(self, images, targets=None, dis_only=False, single_det=False, nms_time=False):
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        features = self.neck(features)  # [feat1, feat2]

        if self.training:
            return self.heads(features, targets, dis_only=dis_only)
        else:
            result = self.heads(features, targets, test=self.test, single_det=single_det, nms_time=nms_time)
            return result
