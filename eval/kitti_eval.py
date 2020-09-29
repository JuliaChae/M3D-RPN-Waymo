# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
from model.config import cfg, get_output_dir
from shapely.geometry import Polygon
import pickle
import numpy as np
import utils.bbox as bbox_utils
from scipy.interpolate import InterpolatedUnivariateSpline
import sys
import operator
import json
import re
from scipy.spatial import ConvexHull
import utils.eval_utils as eval_utils
import matplotlib.pyplot as plt

import kitti_common as kitti
#Values    Name      Description
#----------------------------------------------------------------------------
#   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                     'Misc' or 'DontCare'
#   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                     truncated refers to the object leaving frame boundaries
#   1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                     0 = fully visible, 1 = partly occluded
#                     2 = largely occluded, 3 = unknown
#   1    alpha        Observation angle of object, ranging [-pi..pi]
#   4    bbox         2D bounding box of object in the frame (0-based index):
#                     contains left, top, right, bottom pixel coordinates
#   3    dimensions   3D object dimensions: height, width, length (in meters)
#   3    location     3D object location x,y,z in camera coordinates (in meters)
#   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#   1    score        Only for results: Float, indicating confidence in
#                     detection, needed for p/r curves, higher is better.
# https://github.com/rafaelpadilla/Object-Detection-Metrics
def kitti_eval(detpath,
               db,
               frameset,
               classname,
               cachedir,
               mode,
               ovthresh=0.5,
               eval_type='2d',
               d_levels=0):
    #Min overlap is 0.7 for cars, 0.5 for ped/bike
    """rec, prec, ap = waymo_eval(detpath,
                              annopath,
                              framesetfile,
                              classname,
                              [ovthresh])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(framename) should be the xml annotations file.
  framesetfile: Text file containing the list of frames, one frame per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """

    #Misc hardcoded variables
    idx = 0
    ovthresh_dc = 0.5
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(framename)
    # assumes framesetfile is a text file with each line an frame name
    # cachedir caches the annotations in a pickle file

    frame_path = get_frame_path(db, mode, eval_type)
    class_recs = load_recs(frameset, frame_path, db, mode, classname)
    # read dets
    detfile = detpath.format(classname)
    print('Opening det file: ' + detfile)

    gt_annos = kitti.get_label_annos(label_path)
    dt_annos = kitti.get_label_annos(result_path)

    #sys.exit('donezo')
    with open(detfile, 'r') as f:
        lines = f.readlines()
    #Extract detection file into array
    splitlines   = [x.strip().split(' ') for x in lines]
    #Many entries have the same idx & token
    frame_idx    = [x[0] for x in splitlines] #TODO: I dont like how this is along many frames
    frame_tokens = [x[1] for x in splitlines]
    confidence   = np.array([float(x[2]) for x in splitlines])
    #All detections for specific class
    bbox_elem  = cfg[cfg.NET_TYPE.upper()].NUM_BBOX_ELEM
    BB         = np.array([[float(z) for z in x[3:3+bbox_elem]] for x in splitlines])
    det_cnt    = np.zeros((cfg.KITTI.MAX_FRAME))
    _, uncertainties = eval_utils.extract_uncertainties(bbox_elem,splitlines)
    #Repeated for X detections along every frame presented
    idx = len(frame_idx)
    #DEPRECATED ---- 3 types, easy medium hard
    tp         = np.zeros((idx,d_levels))
    fp         = np.zeros((idx,d_levels))
    fn         = np.zeros((idx))
    tp_frame   = np.zeros(cfg.KITTI.MAX_FRAME)
    fp_frame   = np.zeros(cfg.KITTI.MAX_FRAME)
    npos_frame = np.zeros(cfg.KITTI.MAX_FRAME)
    npos       = np.zeros((len(class_recs),d_levels))
    #Count number of total labels in all frames
    count_npos(class_recs, npos, npos_frame)
    det_results         = []
    frame_uncertainties = []
    #Check if there are any dets at all
    if BB.shape[0] > 0:
        # sort by confidence (highest first)
        sorted_ind    = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        idx_sorted          = [int(frame_idx[x]) for x in sorted_ind]
        frame_tokens_sorted = [frame_tokens[x] for x in sorted_ind]
        #print(frame_ids)

        # go down dets and mark true positives and false positives
        #Zip together sorted_ind with frame tokens sorted. 
        #sorted_ind -> Needed to know which detection we are selecting next
        #frame_tokens_sorted -> Needed to know which set of GT's are for the same frame as the det
        print('num dets {}'.format(len(sorted_ind)))
        idx = 0
        for det_idx,token in zip(sorted_ind,frame_tokens_sorted):
            det_confidence = confidence[det_idx]
            #R is a subset of detections for a specific class
            #print('doing det for frame {}'.format(frame_idx[d]))
            #Need to find associated GT frame ID alongside its detection id 'd'
            #Only one such frame, why appending?
            #print(confidence[det_idx])
            R = None
            skip_iter = True
            R = eval_utils.find_rec(class_recs,token)
            if(R is None):
                continue
            #Deprecated
            #R = class_recs[frame_ids[d]]
            bb = BB[det_idx, :].astype(float)
            var = {}
            #Variance extraction, collect on a per scene basis
            for key,val in uncertainties.items():
                #uc_avg[key][int(R['idx'])] += val[det_idx, :]
                var[key] = val[det_idx, :]
            det_cnt[int(R['idx'])] += 1
            #Variance extraction, collect on a per scene basis
            ovmax = -np.inf
            #Multiple possible bounding boxes, perhaps for multi car detection
            BBGT = R['boxes'].astype(float)
            BBGT_dc = R['boxes_dc'].astype(float)
            #Preload all GT boxes and count number of true positive GT's
            #Not sure why we're setting ignore to false here if it were true
            #for i, BBGT_elem in enumerate(BBGT):
            #    BBGT_height = BBGT_elem[3] - BBGT_elem[1]
            ovmax_dc = 0
            if BBGT_dc.size > 0 and cfg.TEST.IGNORE_DC:
                overlaps_dc = eval_utils.iou(BBGT_dc,bb,eval_type)
                ovmax_dc = np.max(overlaps_dc)
            #Compute IoU
            if BBGT.size > 0:
                overlaps = eval_utils.iou(BBGT,bb,eval_type)
                ovmax = np.max(overlaps)
                #Index of max overlap between a BBGT and BB
                jmax = np.argmax(overlaps)
            else:
                jmax = 0
            # Minimum IoU Threshold for a true positive
            if ovmax > ovthresh and ovmax_dc < ovthresh_dc:
                #if ovmax > ovthresh:
                #ignore if not contained within easy, medium, hard
                if not R['ignore'][jmax]:
                    if not R['hit'][jmax]:
                        #print('TP')
                        if(R['difficulty'][jmax] <= 2):
                            tp[idx,2] += 1
                        if(R['difficulty'][jmax] <= 1):
                            tp[idx,1] += 1
                        if(R['difficulty'][jmax] <= 0):
                            tp[idx,0] += 1
                            #print('ez')
                        tp_frame[int(R['idx'])] += 1
                        R['hit'][jmax] = True
                        det_results.append(write_det(R,det_confidence,ovmax,bb,var,jmax))
                    else:
                        #print('FP-hit')
                        #If it already exists, cant double classify on same spot.
                        if(R['difficulty'][jmax] <= 2):
                            fp[idx,2] += 1
                        if(R['difficulty'][jmax] <= 1):
                            fp[idx,1] += 1
                        if(R['difficulty'][jmax] <= 0):
                            fp[idx,0] += 1
                        fp_frame[int(R['idx'])] += 1
                        det_results.append(write_det(R,det_confidence,ovmax,bb,var))
            #If your IoU is less than required, its simply a false positive.
            elif(BBGT.size > 0 and ovmax_dc < ovthresh_dc):
                #print('FP-else')
                #elif(BBGT.size > 0)
                #if(R['difficulty'][jmax] <= 2):
                #    fp[det_idx,2] += 1
                #if(R['difficulty'][jmax] <= 1):
                #    fp[det_idx,1] += 1
                #if(R['difficulty'][jmax] <= 0):
                #    fp[det_idx,0] += 1
                fp[idx,2] += 1
                fp[idx,1] += 1
                fp[idx,0] += 1
                fp_frame[int(R['idx'])] += 1
                det_results.append(write_det(R,det_confidence,ovmax,bb,var))
            idx = idx + 1
    else:
        print('waymo eval, no GT boxes detected')
    #for i in np.arange(cfg.KITTI.MAX_FRAME):
    #    frame_dets = np.sum(det_cnt[i])
    #    frame_uc = eval_utils.write_frame_uncertainty(uc_avg,frame_dets,i)
    #    if(frame_uc != '' and cfg.DEBUG.PRINT_SCENE_RESULT):
    #        print(frame_uc)
    #    frame_uncertainties.append(frame_uc)

    if(cfg.DEBUG.TEST_FRAME_PRINT):
        eval_utils.display_frame_counts(tp_frame,fp_frame,npos_frame)
    out_dir = get_output_dir(db,mode='test')
    out_file = '{}_detection_results.txt'.format(classname)
    eval_utils.save_detection_results(det_results, out_dir, out_file)
    #if(len(frame_uncertainties) != 0):
    #    uc_out_file = '{}_frame_uncertainty_results.txt'.format(classname)
    #    eval_utils.save_detection_results(frame_uncertainties, out_dir, uc_out_file)

    map = mrec = mprec = np.zeros((d_levels,))
    prec = 0
    rec  = 0
    fp_sum = np.cumsum(fp, axis=0)
    tp_sum = np.cumsum(tp, axis=0)
    #fn     = 1-fp
    #fn_sum = np.cumsum(fn, axis=0)
    npos_sum = np.sum(npos, axis=0)
    print(tp_sum)
    print(fp_sum)
    print(npos_sum)
    #print('Difficulty Level: {:d}, fp sum: {:f}, tp sum: {:f} npos: {:d}'.format(i, fp_sum[i], tp_sum[i], npos[i]))
    #recall
    #Per frame per class AP
    for i in range(0,d_levels):
        npos_sum_d = npos_sum[i]
        #Override to avoid NaN
        if(npos_sum_d == 0):
            npos_sum_d = np.sum([1])
        rec = tp_sum[:,i] / npos_sum_d.astype(float)
        prec = tp_sum[:,i] / np.maximum(tp_sum[:,i] + fp_sum[:,i], np.finfo(np.float64).eps)
        #print(rec)
        #print(prec)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth precision
        rec, prec = zip(*sorted(zip(rec, prec)))
        #plt.scatter(rec,prec)
        #plt.show()
        mprec[i]  = np.average(prec)
        mrec[i]   = np.average(rec)
        map[i]    = eval_utils.ap(rec, prec)
    return mrec, mprec, map


def count_npos(class_recs, npos, npos_frame):
    for i, rec in enumerate(class_recs):
        if(rec['ignore_frame'] is False):
            for j, ignore_elem in enumerate(rec['ignore']):
                if(not ignore_elem):
                    if(rec['difficulty'][j] <= 2):
                        npos[i,2] += 1
                    if(rec['difficulty'][j] <= 1):
                        npos[i,1] += 1
                    if(rec['difficulty'][j] <= 0):
                        npos[i,0] += 1
                    npos_frame[int(rec['idx'])] += 1

def get_frame_path(db, mode, eval_type):
    mode_sub_folder = db.subfolder_from_mode(mode)
    if(eval_type == 'bev' or eval_type == '3d' or eval_type == 'bev_aa'):
        frame_path = os.path.join(db._devkit_path, mode_sub_folder, 'velodyne')
    elif(eval_type == '2d'):
        frame_path = os.path.join(db._devkit_path, mode_sub_folder, 'images_2')
    return frame_path

def load_recs(frameset, frame_path, db, mode, classname):
    class_recs = []
    classes = (
            'DontCare',  # always index 0
            'Pedestrian',
            'Car',
            'Cyclist')
    num_classes = len(classes)
    class_to_ind = dict(
            list(zip(classes, list(range(num_classes)))))
    for tmp_rec in enumerate(gt_anno):
        if(len(tmp_rec['bbox']) == 0):
            tmp_rec['ignore_frame'] = True
        else:
            tmp_rec['ignore_frame'] = False
            if(len(tmp_rec['name']) > 0):
                gt_class_idx = np.where(tmp_rec['name'] == classname)[0]
            else:
                gt_class_idx = np.empty((0,))
            tmp_rec['gt_classes'] = tmp_rec['gt_classes'][gt_class_idx]
            tmp_rec['boxes'] = tmp_rec['boxes'][gt_class_idx]
            tmp_rec['gt_overlaps'] = tmp_rec['gt_overlaps'][gt_class_idx]
            tmp_rec['det'] = tmp_rec['det'][gt_class_idx]
            tmp_rec['ignore'] = tmp_rec['ignore'][gt_class_idx]
            tmp_rec['difficulty'] = tmp_rec['difficulty'][gt_class_idx]
            for i, elem in enumerate(tmp_rec['difficulty']):
                if elem != 0 and elem != 1 and elem != 2:
                    tmp_rec['ignore'][i] = True
        #tmp_rec['frame_idx']   = frame_idx
        #List of all frames with GT boxes for a specific class
        class_recs.append(tmp_rec)
        #Only print every hundredth annotation?
        if i % 10 == 0 and cfg.DEBUG.EN_TEST_MSG:
            #print(recs[idx_name])
            print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(frameset)))
    return class_recs

def write_det(R,confidence,ovmax,bb,var,jmax=None):
    frame    = R['idx']
    truncation     = -1
    occlusion      = -1
    distance       = -1
    difficulty     = -1
    iou            = ovmax
    class_t        = -1
    bbgt           = np.full((len(bb)),-1)
    #pts            = -1
    out_str  = ''
    out_str += 'frame_idx: {} '.format(frame)
    out_str += 'confidence: {} '.format(confidence)
    if(len(bb) > cfg.IMAGE.NUM_BBOX_ELEM):
        out_str += 'bbdet3d: '
    else:
        out_str += 'bbdet: '
    for bbox_elem in bb:
        out_str += '{:.5f} '.format(bbox_elem)
    for key,val in var.items():
        out_str += '{}: '.format(key)
        for var_elem in val:
            out_str += '{:.10f} '.format(var_elem)
    if(jmax is not None):
        #pts        = R['pts'][jmax]
        difficulty = R['difficulty'][jmax]
        #track_id   = R['ids'][jmax]
        class_t    = R['gt_classes'][jmax]
        bbgt       = R['boxes'][jmax]
        truncation = R['trunc'][jmax]
        occlusion  = R['occ'][jmax]
        distance   = R['distance'][jmax]
    #out_str   += 'track_idx: {} difficulty: {} pts: {} cls: {} '.format(track_id,
    #                                                                    difficulty,
    #                                                                    pts,
    #                                                                    class_t)
    out_str   += 'difficulty: {} cls: {} '.format(difficulty,
                                                  class_t)
    if(len(bbgt) > cfg.IMAGE.NUM_BBOX_ELEM):
        out_str += 'bbgt3d: '
    else:
        out_str += 'bbgt: '
    for i in range(len(bbgt)):
        out_str += '{:.3f} '.format(bbgt[i])
    out_str += 'occlusion: {:.5f} truncation: {:.3f}  distance: {:.3f} iou: {:.3f}'.format(occlusion,
                                                                                            truncation,
                                                                                            distance,
                                                                                            iou)
    #out_str += 'avg_intensity: {:.5f} avg_elongation: {:.5f} truncation: {:.3f} return_ratio: {:.5f} distance: {:.3f} iou: {:.3f}'.format(avg_intensity,
    #                                                                                                                                      avg_elongation,
    #                                                                                                                                      truncation,
    #                                                                                                                                      return_ratio,
    #                                                                                                                                      distance,
    #                                                                                                                                      iou)
    return out_str

#DEPRECATED
#def write_det(R,bb,confidence,var,jmax=None):
#    frame    = R['idx']
#    out_str  = ''
#    out_str += 'frame_idx: {} '.format(frame)
#    out_str += 'confidence: {} '.format(confidence)
#    out_str += 'bbdet: '
#    for bbox_elem in bb:
#        out_str += '{:.5f} '.format(bbox_elem)
#    for key,val in var.items():
#        out_str += '{}: '.format(key)
#        for var_elem in val:
#            out_str += '{:.10f} '.format(var_elem)
#    if(jmax is not None):
#        #pts        = R['pts'][jmax]
#        difficulty = R['difficulty'][jmax]
#        #track_id   = R['ids'][jmax]
#        class_t    = R['gt_classes'][jmax]
#        bbgt       = R['boxes'][jmax]
#        #out_str   += 'track_idx: {} '.format(track_id)
#        out_str   += 'difficulty: {} '.format(difficulty)
#        #out_str   += 'pts: {} '.format(pts)
#        out_str   += 'cls: {} '.format(class_t)
#        out_str   += 'bbgt: '
#        for bbox_elem in bbgt:
#           out_str += '{:4.3f} '.format(bbox_elem)
#    return out_str