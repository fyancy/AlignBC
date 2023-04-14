import torch
import torch.nn.functional as F


def Real_MultiBin_loss(vector_ori, gt_ori, num_bin=4):
    gt_ori = gt_ori.view(-1, gt_ori.shape[-1])  # bin1 cls, bin1 offset, bin2 cls, bin2 offst

    cls_losses = 0
    reg_losses = 0
    reg_cnt = 0
    for i in range(num_bin):
        # bin cls loss
        cls_ce_loss = F.cross_entropy(vector_ori[:, (i * 2): (i * 2 + 2)], gt_ori[:, i].long(), reduction='none')
        # regression loss
        valid_mask_i = (gt_ori[:, i] == 1)
        cls_losses += cls_ce_loss.mean()
        if valid_mask_i.sum() > 0:
            s = num_bin * 2 + i * 2
            e = s + 2
            pred_offset = F.normalize(vector_ori[valid_mask_i, s: e])
            reg_loss = F.l1_loss(pred_offset[:, 0], torch.sin(gt_ori[valid_mask_i, num_bin + i]), reduction='none') + \
                       F.l1_loss(pred_offset[:, 1], torch.cos(gt_ori[valid_mask_i, num_bin + i]), reduction='none')

            reg_losses += reg_loss.sum()
            reg_cnt += valid_mask_i.sum()

    return cls_losses / num_bin + reg_losses / reg_cnt
