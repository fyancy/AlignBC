
import numpy as np
cimport numpy as np


cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b


def cpu_soft_nms(np.ndarray[float, ndim=2] box, float sigma=0.5, float Nt=0.3, float threshold=0.001,
                 unsigned int method=0):
    cdef unsigned int N = box.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1, x2, y1, y2, tx1, tx2, ty1, ty2, ts, area, weight, ov
    cdef float n, tn  # order number

    order = np.arange(N, dtype=float).reshape(-1, 1)
    boxes = np.concatenate((box, order), axis=1, dtype=float)

    for i in range(N):
        # print('******************')
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]
        tn = boxes[i, -1]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, :] = boxes[maxpos, :]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts
        boxes[maxpos, -1] = tn

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]
        tn = boxes[i, -1]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]
            n = boxes[pos, -1]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  #iou between max box and detection box

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


# extended 3D soft-nms
def ext_soft_nms(np.ndarray[float, ndim=2] box, float sigma=0.5, float gamma=2,
                 float Nt=0.3, float threshold=0.001, unsigned int method=2):
    cdef unsigned int N = box.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1, x2, y1, y2, tx1, tx2, ty1, ty2, ts, area, weight, ov, tz, z, tx, x  # add depth and loc_x
    cdef float n, tn  # order number

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

                    w_dep = 1 - np.exp(-(z - tz)**2 / gamma) + 1e-7
                    weight = weight * w_dep

                    if abs(z - tz)>2.0 or abs(x-tx)>2.0:  # width
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
