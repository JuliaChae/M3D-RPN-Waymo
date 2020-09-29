""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import os
import sys
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))
import kitti_util as utils
import argparse

import pdb 
try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

cbox = np.array([[0, 70.4], [-40, 40], [-3, 1]])


class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="validation", args=None):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        print(root_dir, split)
        self.split_dir = os.path.join(root_dir, 'data/waymo_split/validation_left')
        self.output_dir = os.path.join(root_dir, 'output/eval/waymo_results_left/400000/left/vis_analysis')

        if split == "training":
            self.num_samples = 11000
        elif split == "testing":
            self.num_samples = 7518
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

        lidar_dir = "velodyne"
        depth_dir = "depth"
        pred_dir = "pred"
        if args is not None:
            pred_dir = args.preddir

        self.image_dir = os.path.join(self.split_dir, "image_3")
        #self.image_dir = os.path.join(root_dir, 'output/visualization/validation')
        self.label_dir = os.path.join(self.split_dir, "label_3")
        self.calib_dir = os.path.join(self.split_dir, "calib")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.pred_dir = os.path.join(root_dir, 'output/eval/waymo_results_left/400000/left/data')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, "%015d.png" % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.lidar_dir, "%015d.bin" % (idx))
        print(lidar_filename)
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, "%015d.txt" % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        label_filename = os.path.join(self.label_dir, "%015d.txt" % (idx))
        return utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        assert idx < self.num_samples
        pred_filename = os.path.join(self.pred_dir, "%015d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        pred_filename = os.path.join(self.pred_dir, "%015d.txt" % (idx))
        return os.path.exists(pred_filename)


class kitti_object_video(object):
    """ Load data for KITTI videos """

    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted(
            [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
        )
        self.lidar_filenames = sorted(
            [os.path.join(lidar_dir, filename) for filename in os.listdir(lidar_dir)]
        )
        print(len(self.img_filenames))
        print(len(self.lidar_filenames))
        # assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = self.img_filenames[idx]
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert idx < self.num_samples
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib


def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, "dataset/2011_09_26/")
    dataset = kitti_object_video(
        os.path.join(video_path, "2011_09_26_drive_0023_sync/image_00/data"),
        os.path.join(video_path, "2011_09_26_drive_0023_sync/velodyne_points/data"),
        video_path,
    )
    print(len(dataset))
    for _ in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        cv2.imshow("video", img)
        draw_lidar(pc)
        raw_input()
        pc[:, 0:3] = dataset.get_calibration().project_velo_to_rect(pc[:, 0:3])
        draw_lidar(pc)
        raw_input()
    return

def get_lidar_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo



def show_image_with_boxes(img, objects, calib, data_idx, savepath, ignores_count, show3d=True, depth=None):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    ignores = filter_objects(objects)
    for obj in ignores:
        if obj == True:
            ignores_count += 1
    index = 0
    for obj in objects:
        if obj.type == "DontCare":
            continue
        if ignores[index] == True:
            ignores_count +=1
            index += 1
            continue
        index +=1 
        cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 0),
            2,
        )
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)
    show3d = True
    cv2.imwrite(savepath + "/2d_ignores.png", img1)
    if show3d:
        cv2.imwrite(savepath + "/" + str(data_idx) + ".png", img2)
    if depth is not None:
        cv2.imwrite("imgs/depth" + ".png", depth)
    return img1, img2, ignores_count

def filter_objects(objects):
    objects.sort(key=lambda x: x.t[2], reverse=False)
    ignore_list = [False]*len(objects)
    for i in range(0, len(objects)):
        front_obj = objects[i]
        for j in range(i+1, len(objects)):
            back_obj = objects[j]
            if front_obj.box2d[0]<back_obj.box2d[0] and front_obj.box2d[1] < back_obj.box2d[1] and front_obj.box2d[2] > back_obj.box2d[2] and front_obj.box2d[3] > back_obj.box2d[3]:
                ignore_list[j] = True
    return ignore_list

def show_image_with_pred_boxes(img, objects, calib, data_idx, savepath, show3d=True, depth=None):
    """ Show image with 2D bounding boxes """
    #img1 = cv2.imread(savepath + "/" + str(data_idx) + ".png")
    #img2 = cv2.imread(savepath + "/" + str(data_idx) + ".png")
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == "DontCare":
            continue
        cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 0),
            2,
        )
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color = (0,0,255))
    show3d = True
    #cv2.imwrite(savepath + "/" + str(data_idx) + "2d.png", img1)
    if show3d:
        cv2.imwrite(savepath + "/" + str(data_idx) + "_pred.png", img2)
    if depth is not None:
        cv2.imwrite("imgs/depth" + ".png", depth)
    return img1, img2

def show_image_with_boxes_3type(img, objects, calib, objects2d, name, objects_pred):
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    type_list = ["Pedestrian", "Car", "Cyclist"]
    # draw Label
    color = (0, 255, 0)
    for obj in objects:
        if obj.type not in type_list:
            continue
        cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            color,
            3,
        )
    startx = 5
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [obj.type for obj in objects if obj.type in type_list]
    text_lables.insert(0, "Label:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    # draw 2D Pred
    color = (0, 0, 255)
    for obj in objects2d:
        cv2.rectangle(
            img1,
            (int(obj.box2d[0]), int(obj.box2d[1])),
            (int(obj.box2d[2]), int(obj.box2d[3])),
            color,
            2,
        )
    startx = 85
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [type_list[obj.typeid - 1] for obj in objects2d]
    text_lables.insert(0, "2D Pred:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    # draw 3D Pred
    if objects_pred is not None:
        color = (100, 0, 0)
        for obj in objects_pred:
            if obj.type not in type_list:
                continue
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                color,
                1,
            )
        startx = 165
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_lables = [obj.type for obj in objects_pred if obj.type in type_list]
        text_lables.insert(0, "3D Pred:")
        for n in range(len(text_lables)):
            text_pos = (startx, 25 * (n + 1))
            cv2.putText(
                img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA
            )

    cv2.imwrite("imgs/" + str(name) + ".png", img1)


def show_lidar_topview_with_boxes(pc_velo, objects, calib, savepath, objects_pred=None):
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)
    top_view = utils.lidar_to_top(pc_velo)
    top_image = utils.draw_top_image(top_view)
    print("top_image:", top_image.shape)
    # gt

    def bbox3d(obj):
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = utils.draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )

    cv2.imwrite(savepath + "0_top_img.png", top_image)
    return top_image

def show_topview_with_boxes(objects, calib, savepath, data_idx, objects_pred=None):
    """ top_view image"""
    top_image = np.zeros((500,500,3), np.uint8)

    def bbox3d(obj):
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = utils.draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )

    cv2.imwrite(savepath + "/" + str(data_idx) + "_topview.png", top_image)
    return top_image

def show_lidar_with_boxes(
    pc_velo,
    objects,
    calib,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
    data_idx = None,
    savepath = None
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        print(("FOV point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)

        # Draw depth
        if depth is not None:
            # import pdb; pdb.set_trace()
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            print("dep_pc_velo:", dep_pc_velo)

            draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))

        # Draw heading arrow
        _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig
            )
    mlab.savefig(savepath + '/{}_lidar.png'.format(data_idx), magnification=30)
    mlab.clf()
    mlab.close()


def box_min_max(box3d):
    box_min = np.min(box3d, axis=0)
    box_max = np.max(box3d, axis=0)
    return box_min, box_max


def get_velo_whl(box3d, pc):
    bmin, bmax = box_min_max(box3d)
    ind = np.where(
        (pc[:, 0] >= bmin[0])
        & (pc[:, 0] <= bmax[0])
        & (pc[:, 1] >= bmin[1])
        & (pc[:, 1] <= bmax[1])
        & (pc[:, 2] >= bmin[2])
        & (pc[:, 2] <= bmax[2])
    )[0]
    # print(pc[ind,:])
    if len(ind) > 0:
        vmin, vmax = box_min_max(pc[ind, :])
        return vmax - vmin
    else:
        return 0, 0, 0, 0


def dataset_viz(root_dir, args):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    # load 2d detection results
    #objects2ds = read_det_file("box2d.list")
    frame_count = 1
    if args.selected_frames == None: 
        data_idx_list = list(range(len(dataset)))
    else:
        data_idx_list = args.selected_frames[1:-1]
        data_idx_list = data_idx_list.split(',')
        data_idx_list = [int(x) for x in data_idx_list]
    ignores_count = 0
    class_count = 0
    for data_idx in data_idx_list:
        print(data_idx)
        if args.ind > 0:
            data_idx = args.ind
        if data_idx % frame_count == 0: 
            # Load data from dataset
            #data_idx = 1

            if args.split == "training":
                objects = dataset.get_label_objects(data_idx)
            else:
                objects = []
            #objects2d = objects2ds[data_idx]

            objects_pred = None
            if args.pred:
                # if not dataset.isexist_pred_objects(data_idx):
                #    continue
                objects_pred = dataset.get_pred_objects(data_idx)
                if objects_pred == None:
                    continue
            if objects_pred == None:
                print("no pred file")
                # objects_pred[0].print_object()

            n_vec = 4
            dtype = np.float32
            if args.dtype64:
                dtype = np.float64
            #pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
            calib = dataset.get_calibration(data_idx)
            img = dataset.get_image(data_idx)
            img_height, img_width, _ = img.shape
            print(data_idx, "image shape: ", img.shape)
            depth = None

            # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            # depth_height, depth_width, depth_channel = img.shape

            # print(('Image shape: ', img.shape))

            if args.stat:
                stat_lidar_with_boxes(pc_velo, objects, calib)
                continue
            print("======== Objects in Ground Truth ========")
            n_obj = 0
            for obj in objects:
                if obj.type != "DontCare":
                    print("=== {} object ===".format(n_obj + 1))
                    obj.print_object()
                    n_obj += 1
            if args.show_lidar_topview_with_boxes:
                # Draw lidar top view
                show_lidar_topview_with_boxes(pc_velo, objects, calib, dataset.output_dir, objects_pred)
                show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height,
                                      objects_pred, depth, img, data_idx, dataset.output_dir)
            class_count += len(objects)
            # show_image_with_boxes_3type(img, objects, calib, objects2d, data_idx, objects_pred)
            if args.show_topview_with_boxes:
                show_topview_with_boxes(objects, calib, dataset.output_dir, data_idx, objects_pred)
            if args.show_image_with_boxes and len(objects) >0:
                # Draw 2d and 3d boxes on image
                if args.pred:
                    show_image_with_pred_boxes(img, objects_pred, calib, data_idx, dataset.output_dir, True, depth)
                else:
                    _, _, ignores_count = show_image_with_boxes(img, objects, calib, data_idx, dataset.output_dir, ignores_count, True, depth)

def read_det_file(det_filename):
    """ Parse lines in 2D detection output files """
    #det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}
    objects = {}
    with open(det_filename, "r") as f:
        for line in f.readlines():
            obj = utils.Object2d(line.rstrip())
            if obj.img_name not in objects.keys():
                objects[obj.img_name] = []
            objects[obj.img_name].append(obj)
        # objects = [utils.Object2d(line.rstrip()) for line in f.readlines()]

    return objects


if __name__ == "__main__":
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    parser = argparse.ArgumentParser(description="KIITI Object Visualization")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="",
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-i",
        "--ind",
        type=int,
        default=0,
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-p", "--pred", action="store_true", help="show predict results"
    )
    parser.add_argument(
        "-s",
        "--stat",
        action="store_true",
        help=" stat the w/h/l of point cloud in gt bbox",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="use training split or testing split (default: training)",
    )
    parser.add_argument(
        "-r",
        "--preddir",
        type=str,
        default="pred",
        metavar="N",
        help="predicted boxes  (default: pred)",
    )
    parser.add_argument("--vis", action="store_true", help="show images")
    parser.add_argument("--img_fov", action="store_true", help="front view mapping")
    parser.add_argument("--const_box", action="store_true", help="constraint box")
    parser.add_argument(
        "--dtype64", action="store_true", help="for float64 datatype, default float64"
    )
    parser.add_argument(
        "--show_image_with_boxes", action="store_true", help="show lidar"
    )
    parser.add_argument(
        "--show_lidar_topview_with_boxes",
        action="store_true",
        help="show lidar topview",
    )
    parser.add_argument(
        "--show_topview_with_boxes",
        action="store_true",
        help="show lidar topview",
    )
    parser.add_argument(
        "--selected_frames",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    if args.vis:
        dataset_viz(args.dir, args)
