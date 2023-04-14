""" Helper methods for loading and parsing KITTI data.
Author: Charles R. Qi, Kui Xu
Date: September 2017/2018
"""

import torch
import numpy as np
import cv2

cv2.setNumThreads(0)
import os, math
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from PIL import Image
import pdb

TOP_Y_MIN = -30
TOP_Y_MAX = +30
TOP_X_MIN = 0
TOP_X_MAX = 100
TOP_Z_MIN = -3.5
TOP_Z_MAX = 0.6

TOP_X_DIVISION = 0.05
TOP_Y_DIVISION = 0.05
TOP_Z_DIVISION = 0.02

cbox = np.array([[0, 70.4], [-40, 40], [-3, 2]])


def convertRot2Alpha(ry3d, z3d, x3d):
    alpha = ry3d - math.atan2(x3d, z3d)

    # equivalent
    # equ_alpha = ry3d - math.atan2(x3d, z3d)

    while alpha > math.pi: alpha -= math.pi * 2
    while alpha < (-math.pi): alpha += math.pi * 2

    return alpha


def convertAlpha2Rot(alpha, z3d, x3d):
    ry3d = alpha + math.atan2(x3d, z3d) + 0.5 * math.pi

    while ry3d > math.pi: ry3d -= math.pi * 2
    while ry3d < (-math.pi): ry3d += math.pi * 2

    return ry3d


def refresh_attributes(calib):
    # refresh all attributes when P2 varies

    calib.c_u = calib.P[0, 2]
    calib.c_v = calib.P[1, 2]
    calib.f_u = calib.P[0, 0]
    calib.f_v = calib.P[1, 1]
    calib.b_x = calib.P[0, 3] / (-calib.f_u)  # relative
    calib.b_y = calib.P[1, 3] / (-calib.f_v)


class Object3d(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax], dtype=np.float32)

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = np.array((float(data[11]), float(data[12]), float(data[13])), dtype=np.float32)
        # location (x,y,z) in camera coord.

        self.dis_to_cam = np.linalg.norm(self.t)

        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.real_alpha = data[3]
        self.ray = math.atan2(self.t[0], self.t[2])
        self.alpha = convertRot2Alpha(self.ry, self.t[2], self.t[0])

        # difficulty level
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])

        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.t

        return corners3d

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        print("Difficulty of estimation: {}".format(self.estimate_diffculty()))


class Calibration(object):
    """ Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    """

    def __init__(self, calib_filepath, from_video=False, use_right_cam=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)

        # Projection matrix from rect camera coord to image coord
        self.P = calibs["P3"] if use_right_cam else calibs["P2"]
        self.P = np.reshape(self.P, [3, 4])

        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)

        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs["R0_rect"]
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        """ Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        """
        data = {}
        cam2cam = self.read_calib_file(
            os.path.join(calib_root_dir, "calib_cam_to_cam.txt")
        )
        velo2cam = self.read_calib_file(
            os.path.join(calib_root_dir, "calib_velo_to_cam.txt")
        )
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam["R"], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam["T"]
        data["Tr_velo_to_cam"] = np.reshape(Tr_velo_to_cam, [12])
        data["R0_rect"] = cam2cam["R_rect_00"]
        data["P2"] = cam2cam["P_rect_02"]
        return data

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image_tensor(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = torch.cat((pts_3d_rect, torch.ones(pts_3d_rect.shape[0], 1).type_as(pts_3d_rect)), dim=1)
        pts_2d = torch.matmul(pts_3d_rect, torch.from_numpy(self.P).type_as(pts_3d_rect).t())  # nx3
        pts_2d_ = pts_2d.clone()
        pts_2d_[:, 0] /= pts_2d[:, 2]
        pts_2d_[:, 1] /= pts_2d[:, 2]

        return pts_2d_[:, 0:2], pts_2d_[:, 2]

    def project_rect_to_image(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2], pts_2d[:, 2]

    def project_velo_to_image(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_8p_to_4p(self, pts_2d):
        x0 = np.min(pts_2d[:, 0])
        x1 = np.max(pts_2d[:, 0])
        y0 = np.min(pts_2d[:, 1])
        y1 = np.max(pts_2d[:, 1])
        x0 = max(0, x0)
        # x1 = min(x1, proj.image_width)
        y0 = max(0, y0)
        # y1 = min(y1, proj.image_height)
        return np.array([x0, y0, x1, y1])

    def project_velo_to_4p(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: 4 points in image2 coord.
        """
        pts_2d_velo = self.project_velo_to_image(pts_3d_velo)
        return self.project_8p_to_4p(pts_2d_velo)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        """ Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        """
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y

        if isinstance(uv_depth, np.ndarray):
            pts_3d_rect = np.zeros((n, 3))
        else:
            # torch.Tensor or torch.cuda.Tensor
            pts_3d_rect = uv_depth.new(uv_depth.shape).zero_()

        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]

        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    def project_depth_to_velo(self, depth, constraint_box=True):
        depth_pt3d = get_depth_pt3d(depth)
        depth_UVDepth = np.zeros_like(depth_pt3d)
        depth_UVDepth[:, 0] = depth_pt3d[:, 1]
        depth_UVDepth[:, 1] = depth_pt3d[:, 0]
        depth_UVDepth[:, 2] = depth_pt3d[:, 2]
        # print("depth_pt3d:",depth_UVDepth.shape)
        depth_pc_velo = self.project_image_to_velo(depth_UVDepth)
        # print("dep_pc_velo:",depth_pc_velo.shape)
        if constraint_box:
            depth_box_fov_inds = (
                    (depth_pc_velo[:, 0] < cbox[0][1])
                    & (depth_pc_velo[:, 0] >= cbox[0][0])
                    & (depth_pc_velo[:, 1] < cbox[1][1])
                    & (depth_pc_velo[:, 1] >= cbox[1][0])
                    & (depth_pc_velo[:, 2] < cbox[2][1])
                    & (depth_pc_velo[:, 2] >= cbox[2][0])
            )
            depth_pc_velo = depth_pc_velo[depth_box_fov_inds]
        return depth_pc_velo


def get_depth_pt3d(depth):
    pt3d = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            pt3d.append([i, j, depth[i, j]])
    return np.array(pt3d)


def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def approx_proj_center(proj_center, surface_centers, img_size):
    # surface_inside
    img_w, img_h = img_size
    surface_center_inside_img = (surface_centers[:, 0] >= 0) & (surface_centers[:, 1] >= 0) & \
                                (surface_centers[:, 0] <= img_w - 1) & (surface_centers[:, 1] <= img_h - 1)

    if surface_center_inside_img.sum() > 0:
        target_surface_center = surface_centers[surface_center_inside_img.argmax()]
        # y = ax + b
        a, b = np.polyfit([proj_center[0], target_surface_center[0]],
                          [proj_center[1], target_surface_center[1]], 1)
        valid_intersects = []
        valid_edge = []

        left_y = b
        if (0 <= left_y <= img_h - 1):  # 是否与 left edge 相交
            valid_intersects.append(np.array([0, left_y]))  # intersection point
            valid_edge.append(0)

        right_y = (img_w - 1) * a + b  # x = img_w -1
        if (0 <= right_y <= img_h - 1):  # 是否与 right edge 相交
            valid_intersects.append(np.array([img_w - 1, right_y]))
            valid_edge.append(1)

        top_x = -b / a
        if (0 <= top_x <= img_w - 1):  # 是否与 top edge 相交
            valid_intersects.append(np.array([top_x, 0]))
            valid_edge.append(2)

        bottom_x = (img_h - 1 - b) / a
        if (0 <= bottom_x <= img_w - 1):  # 是否与 bottom edge 相交
            valid_intersects.append(np.array([bottom_x, img_h - 1]))
            valid_edge.append(3)

        valid_intersects = np.stack(valid_intersects)
        min_idx = np.argmin(np.linalg.norm(valid_intersects - proj_center.reshape(1, 2),
                                           axis=1))  # default: 2-norm

        return valid_intersects[min_idx], valid_edge[min_idx]
    else:
        return None


# -----------------------------
# show
# ------------------------------

def show_heatmap(img, heat_map, classes=['Car', 'Pedestrian', 'Cyclist'], index=None):
    colors = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ignored_color = np.array([1, 1, 1])

    resized_img = img.resize((heat_map.shape[2], heat_map.shape[1]))
    mix_img = np.array(resized_img) / 255
    all_heat_img = np.zeros_like(mix_img)

    for k in range(len(classes)):
        heat_map_k = heat_map[k, :, :]
        ignored_class_map = np.zeros_like(mix_img)
        ignored_class_map[heat_map_k == -1] = ignored_color
        heat_map_k[heat_map_k == -1] = 0
        class_map = heat_map_k[..., np.newaxis] * np.array(colors[k]).reshape(1, 1, 3) + ignored_class_map

        mix_img += class_map
        all_heat_img += class_map

    # import imageio
    # imageio.imsave('heatmap_img_{}.png'.format(index), mix_img)

    plt.figure(figsize=(10, 6))
    plt.subplot(311)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(312)
    plt.imshow(mix_img)
    plt.axis('off')
    plt.subplot(313)
    plt.imshow(all_heat_img)
    plt.axis('off')
    plt.show()


def show_image_with_boxes(image, cls_ids, target_center, box2d, corners_2d, reg_mask, offset_3D, down_ratio, pad_size,
                          encoded_alphas, vis=True, index=[]):
    image = np.array(image)
    img2 = np.copy(image)  # for 2d bbox
    img3 = np.copy(image)  # for 3d bbox
    font = cv2.FONT_HERSHEY_SIMPLEX

    from utils.visualizer import Visualizer

    img2_vis = Visualizer(img2)

    ori_target_center = (target_center + offset_3D) * down_ratio

    box2d[:, 0::2] = box2d[:, 0::2] * down_ratio
    box2d[:, 1::2] = box2d[:, 1::2] * down_ratio
    center_2d = (box2d[:, :2] + box2d[:, 2:]) / 2

    id_to_cls = ['Car', 'Pedestrian', 'Cyclist']

    for idx in range(target_center.shape[0]):
        if reg_mask[idx] == 0:
            continue

        img2_vis.draw_box(box_coord=box2d[idx])

        alpha_regress = encoded_alphas[idx]
        text = '{} {} {:.1f}'.format(alpha_regress[0], alpha_regress[1], alpha_regress[2] / np.pi * 180)
        # img2_vis.draw_text(text=text, position=(int(box2d[idx, 0]), int(box2d[idx, 1])))

        # 8 x 3
        corners_2d_i = (corners_2d[idx][:, :2] + target_center[idx].reshape(1, 2)) * down_ratio
        img3 = draw_projected_box3d(img3, corners_2d_i, cls=id_to_cls[cls_ids[idx]], draw_corner=index)

        # target center and 3D center
        cv2.circle(img3, tuple(center_2d[idx].astype(np.int)), 4, (255, 0, 0), -1)
        cv2.circle(img3, tuple(ori_target_center[idx].astype(np.int)), 4, (0, 255, 255), -1)

    img2 = img2_vis.output.get_image()
    stacked_img = np.vstack((img2, img3))

    if vis:
        plt.figure(figsize=(10, 6))
        plt.imshow(stacked_img)
        plt.show()

    return img3


def show_edge_heatmap(img, edge_heatmap, interp_edge_indices, output_size):
    # output_size: w, h
    resized_img = img.resize(output_size)
    interp_edge_indices_int = interp_edge_indices.round().astype(np.int)
    full_edgemap = np.zeros((output_size[1], output_size[0]), dtype=np.float32)
    full_edgemap[interp_edge_indices_int[:, 1], interp_edge_indices_int[:, 0]] = edge_heatmap[0]

    plt.figure()
    plt.subplot(211)
    plt.imshow(resized_img)
    plt.subplot(212)
    plt.imshow(full_edgemap)
    plt.show()


def draw_dotted_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=4):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def draw_projected_box3d(image, qs, color=None, cls=None, thickness=1, draw_orientation=False, draw_height_text=False,
                         draw_center_line=False, draw_corner=True, draw_number=False, corner_size=3,
                         draw_corner_line=False,
                         draw_orien_surface=True):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """

    if cls is not None and color is None:
        if cls == 'Car':
            # color = (55, 67, 225)
            color = (0, 255, 255)
        elif cls == 'Pedestrian':
            color = (255, 225, 255)
        elif cls == 'Cyclist':
            color = (139, 0, 0)

    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

    # draw the front: x = l / 2, link 0 <-> 4 and 1 <-> 5
    if draw_orientation:
        cv2.line(image, (qs[0, 0], qs[0, 1]), (qs[5, 0], qs[5, 1]), color, thickness, cv2.LINE_AA)
        cv2.line(image, (qs[1, 0], qs[1, 1]), (qs[4, 0], qs[4, 1]), color, thickness, cv2.LINE_AA)

    if draw_orien_surface:
        orien_mask = np.zeros((image.shape), dtype=np.uint8)
        # orien contours
        contours = qs[[0, 1, 5, 4], :].astype(np.int64)
        orien_mask = cv2.fillPoly(orien_mask, pts=[contours], color=color)

        image = cv2.addWeighted(image, 1, orien_mask, 0.3, 0)

    # draw colorful vertices, 0=4, 1=5, 2=6, 3=7, 9=10
    colors = [(43, 253, 63), (227, 34, 243), (254, 117, 34), (30, 255, 254),
              (43, 253, 63), (227, 34, 243), (254, 117, 34), (30, 255, 254), (255, 216, 49), (255, 216, 49)]

    # colors = [(0, 0, 205), (255, 20, 147), (255, 255, 0), (0, 255, 255),
    # (0, 0, 205), (255, 20, 147), (255, 255, 0), (0, 255, 255), (255, 48, 48), (255, 48, 48)]

    if draw_center_line:
        draw_dotted_line(image, (qs[9, 0], qs[9, 1]), (qs[8, 0], qs[8, 1]), colors[-1], 1, cv2.LINE_AA)

    if draw_height_text:
        height_text_size = 0.5
        height_pos = (qs[:4] + qs[4:8]) / 2
        height_pos_center = qs[8:].mean(axis=0)
        for i in range(height_pos.shape[0]):
            cv2.putText(image, 'h_{}'.format(i + 1), tuple(height_pos[i].astype(np.int)), cv2.FONT_HERSHEY_SIMPLEX,
                        height_text_size, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(image, 'h_c'.format(i + 1), tuple(height_pos_center.astype(np.int)), cv2.FONT_HERSHEY_SIMPLEX,
                    height_text_size, (255, 0, 0), 1, cv2.LINE_AA)

    if type(draw_corner) == bool:
        if draw_corner:
            for i in range(qs.shape[0]):
                cv2.circle(image, tuple(qs[i]), corner_size, colors[i], -1)

                if draw_corner_line:
                    if i in [0, 1, 2, 3]:
                        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[i + 4, 0], qs[i + 4, 1]), colors[i], thickness,
                                 cv2.LINE_AA)
                    elif i == 8:
                        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[i + 1, 0], qs[i + 1, 1]), colors[i], thickness,
                                 cv2.LINE_AA)

                if draw_number:
                    cv2.putText(image, str(i), tuple(qs[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    elif type(draw_corner) == list:
        for i in draw_corner:
            cv2.circle(image, tuple(qs[i]), corner_size, colors[i], -1)

            if draw_corner_line:
                if i in [0, 1, 2, 3]:
                    cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[i + 4, 0], qs[i + 4, 1]), colors[i], thickness,
                             cv2.LINE_AA)
                elif i == 8:
                    cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[i + 1, 0], qs[i + 1, 1]), colors[i], thickness,
                             cv2.LINE_AA)

            if draw_number:
                cv2.putText(image, str(i), tuple(qs[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    return image


def lidar_to_top_coords(x, y, z=None):
    if 0:
        return x, y
    else:
        # print("TOP_X_MAX-TOP_X_MIN:",TOP_X_MAX,TOP_X_MIN)
        X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
        Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
        xx = Yn - int((y - TOP_Y_MIN) // TOP_Y_DIVISION)
        yy = Xn - int((x - TOP_X_MIN) // TOP_X_DIVISION)

        return xx, yy


def draw_box3d_on_top(
        image,
        boxes3d,
        color=(255, 255, 255),
        thickness=1,
        scores=None,
        text_lables=[],
        is_gt=False,
):
    # if scores is not None and scores.shape[0] >0:
    # print(scores.shape)
    # scores=scores[:,0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image.copy()
    num = len(boxes3d)
    startx = 5
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0, 0]
        y0 = b[0, 1]
        x1 = b[1, 0]
        y1 = b[1, 1]
        x2 = b[2, 0]
        y2 = b[2, 1]
        x3 = b[3, 0]
        y3 = b[3, 1]
        u0, v0 = lidar_to_top_coords(x0, y0)
        u1, v1 = lidar_to_top_coords(x1, y1)
        u2, v2 = lidar_to_top_coords(x2, y2)
        u3, v3 = lidar_to_top_coords(x3, y3)
        # if is_gt:
        #     color = (0, 255, 0)
        #     startx = 5
        # else:
        #     color = heat_map_rgb(0.0, 1.0, scores[n]) if scores is not None else 255
        #     startx = 85
        cv2.line(img, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)

    # for n in range(len(text_lables)):
    #     text_pos = (startx, 25 * (n + 1))
    #     cv2.putText(img, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)

    return img


def init_bev_image(out_size=768):
    bird_view = np.ones((out_size, out_size, 3), dtype=np.uint8) * 230

    return bird_view


def draw_bev_box3d(image, boxes3d, color=(255, 255, 255), cls=None, thickness=1, scores=None, text_lables=[],
                   world_size=64, out_size=768):
    if color is None:
        if cls is not None:
            if cls == 'Car':
                # color = (55, 67, 225)
                color = (0, 255, 255)
            elif cls == 'Pedestrian':
                color = (255, 225, 255)
            elif cls == 'Cyclist':
                color = (139, 0, 0)
        else:
            color = (0, 0, 255)

    # import pdb
    # pdb.set_trace()

    for idx in range(boxes3d.shape[0]):
        pts = boxes3d[idx][[0, 1, 2, 3]][:, [0, 2]]
        pts[:, 0] += world_size / 2
        pts[:, 1] = world_size - pts[:, 1]
        pts = (pts * out_size / world_size).astype(np.int32)

        cv2.polylines(image, [pts.reshape(-1, 1, 2)], True, color, 2, lineType=cv2.LINE_AA)
        cv2.line(image, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), color, 4, lineType=cv2.LINE_AA)

    return image
