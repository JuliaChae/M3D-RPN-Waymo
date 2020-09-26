import copy
from eval.kitti_object_eval_python.eval import get_official_eval_result
import argparse
import pickle


def ignore_gt(gt_anno, index, difficulty):
    """
    Indicates whether to ignore GT sample
    Args:
        gt_anno [dict]: Ground truth annotation
        index [int]: GT sample index
        difficulty [int]: Difficulty index
    Returns
        ignore [bool]: Ignore flag
    """
    # Compute ignore
    ignore = False
    if (gt_anno["difficulty"][index] == 0):
        ignore = True
    elif (gt_anno["difficulty"][index] > difficulty):
        ignore = True
    return ignore


def ignore_dt(dt_anno, index, difficulty):
    """
    Indicates whether to ignore detection
    Args:
        dt_anno [dict]: Detection annotation
        index [int]: Detection index
        difficulty [int]: Difficulty index
    Returns
        ignore [bool]: Ignore flag
    """
    ignore = False
    return ignore


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'DontCare']

    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class]
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = ignore_gt(gt_anno=gt_anno, index=i, difficulty=difficulty)
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i] == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = ignore_dt(dt_anno=dt_anno, index=i, difficulty=difficulty)

        if ignore:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def evaluation(det_annos, gt_infos, class_names, class_to_name, **kwargs):
    if 'annos' not in gt_infos[0]:
        return 'None', {}

    eval_det_annos = copy.deepcopy(det_annos)
    eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_infos]
    ap_result_str, ap_dict = get_official_eval_result(gt_annos=eval_gt_annos,
                                                      dt_annos=eval_det_annos,
                                                      current_classes=class_names,
                                                      class_to_name=class_to_name,
                                                      clean_data_func=clean_data)

    return ap_result_str, ap_dict


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_names', type=list, default=['VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'SIGN'], help='')
    args = parser.parse_args()

    class_to_name = {0: 'VEHICLE',
                     1: 'PEDESTRIAN',
                     2: 'CYCLIST',
                     3: 'SIGN'}

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    print('Start to evaluate the Waymo format results...')
    print('CLASS AP @ IGNORE, LEVEL1, LEVEL2')
    ap_result_str, ap_dict = evaluation(
        pred_infos,
        gt_infos,
        class_names=args.class_names,
        class_to_name=class_to_name,
    )
    print(ap_result_str)


if __name__ == '__main__':
    main()
