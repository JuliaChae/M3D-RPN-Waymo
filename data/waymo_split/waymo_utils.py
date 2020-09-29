import numpy as np
import os
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit
import random 

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3
IMAGE_PATH = '/media/trail/harddrive/datasets/Waymo/waymo_10/training/image_0'
TRAIN_PATH = '/media/trail/harddrive/datasets/Waymo/waymo_split/training/label_3'
VAL_PATH = '/media/trail/harddrive/datasets/Waymo/waymo_split/validation/label_3'
import pdb

def cal_dir_stat():
    __, __, files = next(os.walk(IMAGE_PATH))
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for im_file in files:
        im = cv2.imread(os.path.join(IMAGE_PATH, im_file)) # image in M*N*CHANNEL_NUM shape, channel in BGR order
        im = im/255.0
        pixel_num += (im.size/CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    
    return rgb_mean, rgb_std

def generate_train_val_list(train_size, val_size):
    __, __, train_list = next(os.walk(TRAIN_PATH))
    #train_list = random.sample(train_list, k= train_size)
    train_list.sort()
    with open('train_left.txt', 'w') as f:
        for x in train_list:
            f.write(x.replace('.txt', '') + '\n')

    __, __, val_list = next(os.walk(VAL_PATH))
    #val_list = random.sample(val_list, k= val_size)
    val_list.sort()
    with open('val_left.txt', 'w') as f:
        for x in val_list:
            f.write(x.replace('.txt', '') + '\n')


if __name__ == '__main__':
    #mean, std = cal_dir_stat()
    #print('Mean is: ' + str(mean) + '\n')
    #print('Std is: ' + str(std) + '\n')
    generate_train_val_list(0, 0)