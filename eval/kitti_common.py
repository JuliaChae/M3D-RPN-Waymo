import concurrent.futures as futures
import os
import pathlib
import re
from collections import OrderedDict

import numpy as np
from skimage import io

import pdb


def get_image_index_str(img_idx):
    return "{:015d}".format(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path('training') / info_type / img_idx_str
    else:
        file_path = pathlib.Path('testing') / info_type / img_idx_str
    if not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'image_0', '.png', training,
                               relative_path)


def get_label_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'label_0', '.txt', training,
                               relative_path)


def get_velodyne_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path)


def get_calib_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path)


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_kitti_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    # image_infos = []
    root_path = pathlib.Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            image_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)
        image_info['img_path'] = get_image_path(idx, path, training,
                                                relative_path)
        if with_imageshape:
            img_path = image_info['img_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['img_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array(
                [float(info) for info in lines[0].split(' ')[1:13]]).reshape(
                    [3, 4])
            P1 = np.array(
                [float(info) for info in lines[1].split(' ')[1:13]]).reshape(
                    [3, 4])
            P2 = np.array(
                [float(info) for info in lines[2].split(' ')[1:13]]).reshape(
                    [3, 4])
            P3 = np.array(
                [float(info) for info in lines[3].split(' ')[1:13]]).reshape(
                    [3, 4])
            P4 = np.array(
                [float(info) for info in lines[4].split(' ')[1:13]]).reshape(
                    [3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
            image_info['calib/P0'] = P0
            image_info['calib/P1'] = P1
            image_info['calib/P2'] = P2
            image_info['calib/P3'] = P3
            image_info['calib/P4'] = P4
            R0_rect = np.array([
                float(info) for info in lines[5].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect
            image_info['calib/R0_rect'] = rect_4x4
            Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            image_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam
            image_info['calib/Tr_imu_to_velo'] = Tr_imu_to_velo
        if annotations is not None:
            image_info['annos'] = annotations
            add_difficulty_to_annos(image_info)
        return image_info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)


def filter_kitti_anno(image_anno,
                      used_classes,
                      used_difficulty=None,
                      dontcare_iou=None):
    if not isinstance(used_classes, (list, tuple)):
        used_classes = [used_classes]
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x in used_classes
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    if used_difficulty is not None:
        relevant_annotation_indices = [
            i for i, x in enumerate(img_filtered_annotations['difficulty'])
            if x in used_difficulty
        ]
        for key in image_anno.keys():
            img_filtered_annotations[key] = (
                img_filtered_annotations[key][relevant_annotation_indices])

    if 'DontCare' in used_classes and dontcare_iou is not None:
        dont_care_indices = [
            i for i, x in enumerate(img_filtered_annotations['name'])
            if x == 'DontCare'
        ]
        # bounding box format [y_min, x_min, y_max, x_max]
        all_boxes = img_filtered_annotations['bbox']
        ious = iou(all_boxes, all_boxes[dont_care_indices])

        # Remove all bounding boxes that overlap with a dontcare region.
        if ious.size > 0:
            boxes_to_remove = np.amax(ious, axis=1) > dontcare_iou
            for key in image_anno.keys():
                img_filtered_annotations[key] = (img_filtered_annotations[key][
                    np.logical_not(boxes_to_remove)])
    return img_filtered_annotations


def filter_annos_low_score(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno['score']) if s >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def kitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', None),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'difficulty': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'SIGN']

    annotations['name'] = np.array([x[0] for x in content], dtype=object)
    for i in range(0, len(annotations['name'])):
        if annotations['name'][i] in waymo_classes:
            kitti_class = ['Car', 'Pedestrian', 'Cyclist', 'DontCare']
            class_index = waymo_classes.index(annotations['name'][i])
            annotations['name'][i] = kitti_class[class_index]
    #annotations['truncated'] = np.array([float(x[1]) for x in content])
    #annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[round(float(info), 2) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[round(float(info), 2) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[round(float(info), 2) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [round(float(x[14]),2) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([round(float(x[15]),2) for x in content])
    else:
        annotations['score'] = np.full([len(annotations['bbox'])], 1.0)
    if len(content) != 0 and len(content[0]) == 17:  # waymo difficulty
        annotations['difficulty'] = np.array([get_difficulty(int(x[15]), int(x[16])) for x in content])
        #annotations = get_calib(annotations, label_path)
        #annotations['distance'] = rect_to_lidar(annotations['location'], annotations['R0'], annotations['Tr_velo_to_cam'])
    else:
        annotations['difficulty'] = np.zeros([len(annotations['bbox'])])
    return annotations

def get_calib(annotations, label_path):
    calib_path = str(label_path).replace(label_path.parts[-2],'calib')
    camera = label_path.parts[-2][-1]
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        obj = line.strip().split(' ')[1:]
        if line.strip().split(' ')[0] == 'P'+camera+':':
            P = np.array(obj, dtype=np.float32)
            annotations['P'] = P.reshape(3,4) 
        if line.strip().split(' ')[0] == 'R0_rect:':
            R0 = np.array(obj, dtype=np.float32)
            annotations['R0'] = R0.reshape(3,3)
        if line.strip().split(' ')[0] == 'Tr_velo_to_cam_'+camera+':':
            Tr_velo_to_cam = np.array(obj, dtype=np.float32)
            annotations['Tr_velo_to_cam'] = Tr_velo_to_cam.reshape(4,4)
    return annotations


def get_difficulty(num_pts, prepopulated_level):
    if prepopulated_level == 2:
        return 2
    elif num_pts <= 5 and num_pts > 0:
        return 2
    elif num_pts > 5:
        return 1
    elif num_pts == 0:
        return 0
    else:
        return -1


def get_label_annos(label_folder, image_ids=None):
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{15}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        image_idx = get_image_index_str(idx)
        label_filename = label_folder / (image_idx + '.txt')
        annos.append(get_label_anno(label_filename))
    return annos


def area(boxes, add1=False):
    """Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    if add1:
        return (boxes[:, 2] - boxes[:, 0] + 1.0) * (
            boxes[:, 3] - boxes[:, 1] + 1.0)
    else:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2, add1=False):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    if add1:
        all_pairs_min_ymax += 1.0
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    if add1:
        all_pairs_min_xmax += 1.0
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2, add1=False):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2, add1)
    area1 = area(boxes1, add1)
    area2 = area(boxes2, add1)
    union = np.expand_dims(
        area1, axis=1) + np.expand_dims(
            area2, axis=0) - intersect
    return intersect / union

def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom

def rect_to_lidar(pts_rect, R0, V2C):
    """
    :param pts_lidar: (N, 3)
    :return pts_rect: (N, 3)
    """
    pts_rect_hom = cart_to_hom(pts_rect)  # (N, 4)
    R0_ext = np.hstack((R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
    R0_ext[3, 3] = 1

    pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C).T))
    return pts_lidar[:, 0:3]
