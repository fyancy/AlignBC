# https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx

import torch
import numpy as np
from utils.get_bev_box import compute_bev, BEV_IoU, get_iou_3d


def soft_nms_ext(box: np.ndarray, sigma: float = 0.5, gamma: float = 2,
                 Nt: float = 0.3, threshold: float = 0.001, method=2):
    # cdef unsigned int N = box.shape[0]
    # cdef float iw, ih, box_area
    # cdef float ua
    # cdef int pos = 0
    # cdef float maxscore = 0
    # cdef int maxpos = 0
    # cdef float x1, x2, y1, y2, tx1, tx2, ty1, ty2, ts, area, weight, ov, tz, z, tx, x  # add depth and loc_x
    # cdef float n, tn  # order number

    N = box.shape[0]
    order = np.arange(N, dtype=float).reshape(-1, 1)
    boxes = np.concatenate((box, order), axis=1, dtype=float)

    for i in range(N):  # 0, 1, ..., N-1
        # print('******************')
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]
        tz = boxes[i, 5]  # depth
        tx = boxes[i, 6]  # x
        tn = boxes[i, -1]  # order

        pos = i + 1
        # get box with max score from the rest boxes
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, :] = boxes[maxpos, :]

        # swap i-th box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts
        boxes[maxpos, 5] = tz
        boxes[maxpos, 6] = tx
        boxes[maxpos, -1] = tn

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]
        tz = boxes[i, 5]
        tx = boxes[i, 6]
        tn = boxes[i, -1]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]
            z = boxes[pos, 5]
            x = boxes[pos, 6]
            n = boxes[pos, -1]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    w_dep = 1 - np.exp(-(z - tz) ** 2 / gamma) + 1e-7
                    weight = weight * w_dep

                    if abs(z - tz) > 2.0 or abs(x - tx) > 2.0:  # width
                        weight = 1.0

                    boxes[pos, 4] = weight * boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    # print(boxes[pos, 2], boxes[pos, 3], boxes[pos, 4])
                    if boxes[pos, 4] < threshold:
                        boxes[pos, :] = boxes[N - 1, :]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    # keep = [i for i in range(N)]
    keep = boxes[:N, -1]
    keep = [int(k) for k in keep]
    return keep


# clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores
def soft_nms_bev(box: np.ndarray, sigma: float = 0.5,
                 Nt: float = 0.3, threshold: float = 0.001, method=2):
    # box: [dims, locs, ry, score], shape-(N, 3+3+1+1)

    N = box.shape[0]
    order = np.arange(N, dtype=float).reshape(-1, 1)
    boxes = np.concatenate((box, order), axis=1, dtype=float)
    # [dims, locs, ry, score, order]

    for i in range(N):  # 0, 1, ..., N-1
        # print('******************')
        maxscore = boxes[i, -2]
        maxpos = i

        tdims = boxes[i, 0:3]
        tlocs = boxes[i, 3:6]
        trys = boxes[i, 6]
        ts = boxes[i, 7]
        tn = boxes[i, 8]  # order

        pos = i + 1
        # get box with max score from the rest boxes
        while pos < N:
            if maxscore < boxes[pos, -2]:
                maxscore = boxes[pos, -2]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, :] = boxes[maxpos, :]  # change the current as the maximum

        # swap i-th box with position of max box
        boxes[maxpos, 0:3] = tdims  # 别忘了tx1中可是保存了boxes[i,0]备份的
        boxes[maxpos, 3:6] = tlocs
        boxes[maxpos, 6] = trys
        boxes[maxpos, 7] = ts
        boxes[maxpos, 8] = tn

        tdims = boxes[i, 0:3]  # 此时tx1就保存的maxpos位置的bbox信息了
        tlocs = boxes[i, 3:6]
        trys = boxes[i, 6]
        ts = boxes[i, 7]
        tn = boxes[i, 8]  # order

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:  # 向后做NMS比较
            dims = boxes[pos, 0:3]  # 当前位置的bbox
            locs = boxes[pos, 3:6]
            rys = boxes[pos, 6]
            s = boxes[pos, 7]
            n = boxes[pos, 8]  # order

            bbox = compute_bev(rys, dims, locs)  # vector: 8
            tbbox = compute_bev(trys, tdims, tlocs)  # vector: 8
            ov = BEV_IoU(tbbox, bbox)  # iou between max box and detection box

            if ov > 0:
                if method == 1:  # linear
                    if ov > Nt:
                        weight = 1 - ov
                    else:
                        weight = 1
                elif method == 2:  # gaussian
                    weight = np.exp(-(ov * ov) / sigma)
                else:  # original NMS
                    if ov > Nt:
                        weight = 0
                    else:
                        weight = 1

                boxes[pos, -2] = weight * boxes[pos, -2]

                # if box score falls below threshold, discard the box by swapping with last box
                # update N
                # 如果bbox调整后的权重，已经小于阈值threshold，那么这个bbox就可以忽略了，
                # 操作方式是直接用最后一个有效的bbox替换当前pos上的bbox
                if boxes[pos, -2] < threshold:
                    boxes[pos, :] = boxes[N - 1, :]
                    N = N - 1
                    pos = pos - 1

            pos = pos + 1

    keep = boxes[:N, -1]
    keep = [int(k) for k in keep]
    return keep


def soft_nms_3d(box: np.ndarray, sigma: float = 0.5,
                Nt: float = 0.3, threshold: float = 0.001, method=2):
    # box: [N, [dims=(l,h,w), locs=(x,y,z), ry, score]], shape-(N, 3+3+1+1)
    N = box.shape[0]
    boxes_3d = np.zeros([N, 1, 8, 3])
    for i in range(N):  # before nms, change params to 3d corners
        # print(box.shape, box[i, 0:3].shape)
        # boxes_3d[i] = encode_3dbox(ry=box[i, -2], dims=box[i, 0:3],
        #                            locs=box[i, 3:6]).transpose([1, 0])[None, :]  # 1x8x3
        boxes_3d[i] = encode_3dbox_v2(rotys=box[i, -2], dims=box[i, 0:3], locs=box[i, 3:6])  # 1x8x3

    order = np.arange(N, dtype=float).reshape(-1, 1)
    boxes = np.concatenate((box[:, [-1]], order), axis=1, dtype=float)
    # [score, order], Nx2

    for i in range(N):  # 0, 1, ..., N-1
        # print('******************')
        maxscore = boxes[i, -2]
        maxpos = i

        tcorner = boxes_3d[i]  # corner
        ts = boxes[i, 0]  # score
        tn = boxes[i, 1]  # order

        pos = i + 1
        # get box with max score from the rest boxes
        while pos < N:
            if maxscore < boxes[pos, -2]:
                maxscore = boxes[pos, -2]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes_3d[i] = boxes_3d[maxpos]
        boxes[i, :] = boxes[maxpos, :]  # change the current as the maximum

        # swap i-th box with position of max box
        boxes_3d[maxpos] = tcorner  # 别忘了tx1中可是保存了boxes[i,0]备份的
        boxes[maxpos, 0] = ts
        boxes[maxpos, 1] = tn

        tcorner = boxes_3d[i]  # corner, 此时tcorner就保存的maxpos位置的bbox信息了
        ts = boxes[i, 0]  # score
        tn = boxes[i, 1]  # order

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        # print('score before: ', boxes[pos, -2])
        while pos < N:  # 向后做NMS比较
            corner = boxes_3d[pos]  # 当前位置的bbox
            s = boxes[pos, 0]  # score
            n = boxes[pos, 1]  # order

            ov = get_iou_3d(corner, tcorner)  # 3d_iou between max box and detection box
            # print('3d iou', ov)

            if ov > 0:
                if method == 1:  # linear
                    if ov > Nt:
                        weight = 1 - ov
                    else:
                        weight = 1
                elif method == 2:  # gaussian
                    weight = np.exp(-(ov * ov) / sigma)
                else:  # original NMS
                    if ov > Nt:
                        weight = 0
                    else:
                        weight = 1

                boxes[pos, -2] = weight * boxes[pos, -2]

                # if box score falls below threshold, discard the box by swapping with last box
                # update N
                # 如果bbox调整后的权重，已经小于阈值threshold，那么这个bbox就可以忽略了，
                # 操作方式是直接用最后一个有效的bbox替换当前pos上的bbox
                # print('score after: ', boxes[pos, -2])
                if boxes[pos, -2] < threshold:
                    boxes_3d[pos, :] = boxes_3d[N - 1, :]
                    boxes[pos, :] = boxes[N - 1, :]
                    N = N - 1
                    pos = pos - 1

            pos = pos + 1

    keep = boxes[:N, -1]
    keep = [int(k) for k in keep]
    return keep


def encode_3dbox(ry, dims, locs):
    # K = np.array(
    #     [7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01,
    #      0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01,
    #      0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03],
    #     dtype=np.float32)

    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]

    # assert l >= w, "the input 'dims' format should be (l, h, w)"

    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]

    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)  # kitti takes the bottom center as the 3D-Box center
    z_corners += - np.float32(w) / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])
    c, s = np.cos(ry), np.sin(ry)
    rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])  # shape: 3x8

    return corners_3d


def encode_3dbox_v2(rotys, dims, locs):
    """

    :param rotys:
    :param dims: (N, 3) or (3,), l, h, w
    :param locs: (N, 3) or (3,), x, y, z
    :return:
    """
    # print(dims.shape)
    if not isinstance(rotys, torch.Tensor):
        rotys = torch.tensor(rotys).float()
    if not isinstance(dims, torch.Tensor):
        dims = torch.tensor(dims).float()
    if not isinstance(locs, torch.Tensor):
        locs = torch.tensor(locs).float()

    rotys = rotys.flatten()
    dims = dims.view(-1, 3)
    locs = locs.view(-1, 3)  # (N, 3)
    locs[:, 1] = locs[:, 1] - dims[:, 1]/2  # change oring to box center

    N = rotys.shape[0]
    ry = rad_to_matrix(rotys, N)

    dims_corners = dims.view(-1, 1).repeat(1, 8)
    dims_corners = dims_corners * 0.5
    dims_corners[:, 4:] = -dims_corners[:, 4:]
    index = torch.tensor([[4, 5, 0, 1, 6, 7, 2, 3],
                          [0, 1, 2, 3, 4, 5, 6, 7],
                          [4, 0, 1, 5, 6, 2, 3, 7]]).repeat(N, 1)

    box_3d_object = torch.gather(dims_corners, 1, index)
    box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1))
    box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)  # (N, 3, 8)

    return box_3d.permute(0, 2, 1)


def rad_to_matrix(rotys, N):
    device = rotys.device

    cos, sin = rotys.cos(), rotys.sin()

    i_temp = torch.tensor([[1, 0, 1],
                           [0, 1, 0],
                           [-1, 0, 1]]).to(dtype=torch.float32, device=device)

    ry = i_temp.repeat(N, 1).view(N, -1, 3)

    ry[:, 0, 0] *= cos
    ry[:, 0, 2] *= sin
    ry[:, 2, 0] *= sin
    ry[:, 2, 2] *= cos

    return ry
