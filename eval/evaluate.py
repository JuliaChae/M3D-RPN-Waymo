import time
import fire
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path,
             result_path,
             save_path,
             current_class=['Car', 'Pedestrian', 'Cyclist'],
             coco=False,
             score_thresh=-1):
    class_to_name = {0: 'Car',
                     1: 'Pedestrian',
                     2: 'Cyclist',
                     3: 'DontCare'}

    gt_annos = kitti.get_label_annos(label_path)
    dt_annos = kitti.get_label_annos(result_path)
    # visualize(gt_annos, dt_annos)
    print(len(dt_annos))
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)

    if coco:
        print(get_coco_eval_result(dt_annos, dt_annos, current_class))
    else:
        result_str, _ = get_official_eval_result(gt_annos, dt_annos, current_class, class_to_name)
        print(result_str)
        with open(save_path, 'w+') as f:
            f.write("\n")
            f.write(result_str)


def analyze(front_path, frontleft_path, left_path, save_path):

    gt_front = kitti.get_label_annos(front_path)
    gt_frontleft = kitti.get_label_annos(frontleft_path)
    gt_left = kitti.get_label_annos(left_path)
    gt_dataset = [gt_front, gt_frontleft, gt_left]

    annos_name = ["Front", "FrontLeft", "Left"]
    split = "left_pred_"
    for i in range(0, 3):
        count = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0, 'DontCare': 0}
        ranges = {'Car': [], 'Pedestrian': [], 'Cyclist': [], 'DontCare': []}
        rot_y = {'Car': [], 'Pedestrian': [], 'Cyclist': [], 'DontCare': []}
        for frame in gt_dataset[i]:
            for j in range(0, len(frame['name'])):
                cls = frame['name'][j]
                distance = frame['location'][j][2]
                ranges[cls].append(distance)
                ry = frame['rotation_y'][j] * (180 / np.pi)
                rot_y[cls].append(ry)
                count[cls] += 1
        print("Count of classes:" + str(count) + '\n')
        with open(save_path + split + "classes.txt", 'a+') as f:
            f.write(annos_name[i] + ' ' + str(count) + '\n')
        sns_plot = sns.distplot(ranges["Car"], color="skyblue", label="Car").set_title(
            annos_name[i] + " Car Distances")
        fig = sns_plot.get_figure()
        fig.savefig("output/eval/dataset_vis/" + split + annos_name[i] + "_car.png")
        plt.clf()

        sns_plot_ped = sns.distplot(ranges["Pedestrian"], color="red", label="Pedestrian").set_title(
            annos_name[i] + " Pedestrian Distances")
        fig_ped = sns_plot_ped.get_figure()
        fig_ped.savefig("output/eval/dataset_vis/" + split + annos_name[i] + "_ped.png")
        plt.clf()

        sns_plot_cyc = sns.distplot(ranges["Cyclist"], color="teal", label="Cyclist").set_title(
            annos_name[i] + " Cyclist Distances")
        fig_cyc = sns_plot_cyc.get_figure()
        fig_cyc.savefig("output/eval/dataset_vis/" + split + annos_name[i] + "_cyc.png")
        plt.clf()

        sns_plot = sns.distplot(rot_y["Car"], color="skyblue", label="Car").set_title(annos_name[i] + " Car Rotation")
        fig = sns_plot.get_figure()
        fig.savefig("output/eval/dataset_vis/" + split + "ry_" + annos_name[i] + "_car.png")
        plt.clf()

        sns_plot_ped = sns.distplot(rot_y["Pedestrian"], color="red", label="Pedestrian").set_title(
            annos_name[i] + " Pedestrian Rotation")
        fig_ped = sns_plot_ped.get_figure()
        fig_ped.savefig("output/eval/dataset_vis/" + split + "ry_" + annos_name[i] + "_ped.png")
        plt.clf()

        sns_plot_cyc = sns.distplot(rot_y["Cyclist"], color="teal", label="Cyclist").set_title(
            annos_name[i] + " Cyclist Rotation")
        fig_cyc = sns_plot_cyc.get_figure()
        fig_cyc.savefig("output/eval/dataset_vis/" + split + "ry_" + annos_name[i] + "_cyc.png")
        plt.clf()
    #print("Count of ranges:" + str(ranges)+'\n')
    #print("Count of ry:" + str(rot_y)+'\n')


def training_analyze(front_path, frontleft_path, left_path):

    gt_front = kitti.get_label_annos(front_path)
    gt_frontleft = kitti.get_label_annos(frontleft_path)
    gt_left = kitti.get_label_annos(left_path)
    gt_dataset = [gt_front, gt_frontleft, gt_left]

    x = ['Car', 'Pedestrian', 'Cyclist', 'DontCare']
    annos_name = ["Front", "FrontLeft", "Left"]
    count = {"Front": [0, 0, 0, 0], "FrontLeft": [0, 0, 0, 0], "Left": [0, 0, 0, 0]}

    for i in range(0, 3):
        view = annos_name[i]
        for frame in gt_dataset[i]:
            view_count = count[view]
            for j in range(0, len(frame['name'])):
                cls = frame['name'][j]
                view_count[x.index(cls)] += 1

    # set width of bar
    barWidth = 0.25

    # set height of bar
    bars1 = count["Front"]
    bars2 = count["FrontLeft"]
    bars3 = count["Left"]

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Front')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='FrontLeft')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Left')

    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Car', 'Pedestrian', 'Cyclist', 'Sign'])

    plt.title("Class Count Across Camera Views")
    # Create legend & Show graphic
    plt.legend()
    plt.savefig("output/eval/dataset_vis/train_class_count.png")

def training_density_analysis(front_path, frontleft_path, left_path):

    gt_front = kitti.get_label_annos(front_path)
    gt_frontleft = kitti.get_label_annos(frontleft_path)
    gt_left = kitti.get_label_annos(left_path)
    gt_dataset = [gt_front, gt_frontleft, gt_left]

    x = ['Car', 'Pedestrian', 'Cyclist', 'DontCare']
    annos_name = ["Front", "FrontLeft", "Left"]
    count = {"Front": [0, 0, 0, 0], "FrontLeft": [0, 0, 0, 0], "Left": [0, 0, 0, 0]}

    total_images = 0 
    total_car = 0
    for i in range(0, 1):
        view = annos_name[i]
        for frame in gt_dataset[i]:
            view_count = count[view]
            if len(frame['name']) != 0:
            	total_images += 1
            for j in range(0, len(frame['name'])):
                cls = frame['name'][j]
                if (cls) == 'Car' or (cls) == 'Pedestrian' or (cls) == 'Cyclist' :
                	total_car += 1
    print(str(total_car/total_images))

    # set width of bar
    barWidth = 0.25

    # set height of bar
    bars1 = count["Front"]
    bars2 = count["FrontLeft"]
    bars3 = count["Left"]

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Front')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='FrontLeft')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Left')

    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Car', 'Pedestrian', 'Cyclist', 'Sign'])

    plt.title("Class Count Across Camera Views")
    # Create legend & Show graphic
    plt.legend()
    plt.savefig("output/eval/dataset_vis/train_class_count.png")

def visualize(gt_annos, dt_annos):
    sns.set()
    ranges = {'Car': [], 'Pedestrian': [], 'Cyclist': [], 'DontCare': []}
    for frame in gt_annos:
        for i in range(0, len(frame['location'])):
            distance = frame['location'][i][2]
            cls = frame['name'][i]
            ranges[cls].append(distance)

    sns_plot = sns.distplot(ranges["Car"], color="skyblue", label="Car")
    fig = sns_plot.get_figure()
    fig.savefig("output/eval/dataset_vis/val_car.png")
    plt.clf()

    sns_plot_ped = sns.distplot(ranges["Pedestrian"], color="red", label="Pedestrian")
    fig_ped = sns_plot_ped.get_figure()
    fig_ped.savefig("output/eval/dataset_vis/val_ped.png")
    plt.clf()

    sns_plot_cyc = sns.distplot(ranges["Cyclist"], color="teal", label="Cyclist")
    fig_cyc = sns_plot_cyc.get_figure()
    fig_cyc.savefig("output/eval/dataset_vis/val_cyc.png")
    plt.clf()

    ranges = {'Car': [], 'Pedestrian': [], 'Cyclist': [], 'DontCare': []}
    for frame in dt_annos:
        for i in range(0, len(frame['location'])):
            distance = frame['location'][i][2]
            cls = frame['name'][i]
            ranges[cls].append(distance)

    sns_plot = sns.distplot(ranges["Car"], color="skyblue", label="Car")
    fig = sns_plot.get_figure()
    fig.savefig("output/eval/dataset_vis/pred_car.png")
    plt.clf()

    sns_plot_ped = sns.distplot(ranges["Pedestrian"], color="red", label="Pedestrian")
    fig_ped = sns_plot_ped.get_figure()
    fig_ped.savefig("output/eval/dataset_vis/pred_ped.png")
    plt.clf()

    sns_plot_cyc = sns.distplot(ranges["Cyclist"], color="teal", label="Cyclist")
    fig_cyc = sns_plot_cyc.get_figure()
    fig_cyc.savefig("output/eval/dataset_vis/pred_cyc.png")
    breakpoint()


def analyze_tp():
    training_views = ['front', 'frontleft', 'left']
    eval_views = ['front', 'frontleft', 'left']
    for train in training_views:
        for evalu in eval_views:
            name = 'output/eval/detections/' + train + '_' + evalu + '.pkl'
            with open(name, 'rb') as f:
                tp_annos = pickle.load(f)

                fig, axes = plt.subplots(ncols=2)
                ranges = [x[2] for x in tp_annos['locs']]
                sns.distplot(ranges, color="skyblue", ax=axes[0]).set_title("Distance from car")
                ry = [x * (180 / np.pi) for x in tp_annos["ry"]]
                sns.distplot(ry, color="red", ax=axes[1]).set_title("Rotation_y of detections")
                fig.savefig("output/eval/detections/" + train + '_' + evalu + '.png')
                plt.clf()

                ry = frame['rotation_y'][j] * (180 / np.pi)
                rot_y[cls].append(ry)
                count[cls] += 1
        print("Count of classes:" + str(count) + '\n')
        with open(save_path + split + "classes.txt", 'a+') as f:
            f.write(annos_name[i] + ' ' + str(count) + '\n')
        sns_plot = sns.distplot(ranges["Car"], color="skyblue", label="Car").set_title(
            annos_name[i] + " Car Distances")
        fig = sns_plot.get_figure()
        fig.savefig("output/eval/dataset_vis/" + split + annos_name[i] + "_car.png")
        plt.clf()

        sns_plot_ped = sns.distplot(ranges["Pedestrian"], color="red", label="Pedestrian").set_title(
            annos_name[i] + " Pedestrian Distances")
        fig_ped = sns_plot_ped.get_figure()
        fig_ped.savefig("output/eval/dataset_vis/" + split + annos_name[i] + "_ped.png")
        plt.clf()

        sns_plot_cyc = sns.distplot(ranges["Cyclist"], color="teal", label="Cyclist").set_title(
            annos_name[i] + " Cyclist Distances")
        fig_cyc = sns_plot_cyc.get_figure()
        fig_cyc.savefig("output/eval/dataset_vis/" + split + annos_name[i] + "_cyc.png")
        plt.clf()

        sns_plot = sns.distplot(rot_y["Car"], color="skyblue", label="Car").set_title(annos_name[i] + " Car Rotation")
        fig = sns_plot.get_figure()
        fig.savefig("output/eval/dataset_vis/" + split + "ry_" + annos_name[i] + "_car.png")
        plt.clf()

        sns_plot_ped = sns.distplot(rot_y["Pedestrian"], color="red", label="Pedestrian").set_title(
            annos_name[i] + " Pedestrian Rotation")
        fig_ped = sns_plot_ped.get_figure()
        fig_ped.savefig("output/eval/dataset_vis/" + split + "ry_" + annos_name[i] + "_ped.png")
        plt.clf()

        sns_plot_cyc = sns.distplot(rot_y["Cyclist"], color="teal", label="Cyclist").set_title(
            annos_name[i] + " Cyclist Rotation")
        fig_cyc = sns_plot_cyc.get_figure()
        fig_cyc.savefig("output/eval/dataset_vis/" + split + "ry_" + annos_name[i] + "_cyc.png")
        plt.clf()
    #print("Count of ranges:" + str(ranges)+'\n')
    #print("Count of ry:" + str(rot_y)+'\n')


def analyze_velocity():
    CALIB_PATH = "data/waymo/training/img_calib"
    path, dirs, files = next(os.walk(CALIB_PATH))
    files.sort()
    velocity = []
    angular_velocity = [] 
    hs_count = 0
    for file in files:
        f = open(CALIB_PATH + '/' + file)
        lines = f.readlines()
        v0 = lines[5].split(' ')[1:]
        cam0_vel = np.array([float(x) for x in v0])
        cam0_v = np.linalg.norm(cam0_vel[:3])
        cam0_w = np.linalg.norm(cam0_vel[3:])
        velocity.append(cam0_v*3.6)
        angular_velocity.append(cam0_w)
        if cam0_v*3.6 > 40:
        	hs_count +=1 
    
    print("Percentage is: " + str(hs_count/len(files)))
    sns_plot_cyc = sns.distplot(velocity, color="teal", label="Velocity").set_title("Velocity distribution in Training Set")
    fig_cyc = sns_plot_cyc.get_figure()
    fig_cyc.savefig("output/eval/dataset_vis/velocity.png")
    plt.clf()

    sns_plot_cyc = sns.distplot(angular_velocity, color="teal", label="Angular Velocity").set_title("Ang Velocity Distribution in Training Set")
    fig_cyc = sns_plot_cyc.get_figure()
    fig_cyc.savefig("output/eval/dataset_vis/ang_velocity.png")
    plt.clf()

if __name__ == '__main__':
    fire.Fire()
