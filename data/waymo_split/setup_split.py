from importlib import import_module
from getopt import getopt
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import pprint
import sys
import os
import cv2
import math
import shutil
import re

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.util import *

split = 'waymo_split'

# base paths
base_data = os.path.join(os.getcwd(), 'data')

waymo_raw_tra = dict()
waymo_raw_tra['cal'] = os.path.join(base_data, 'waymo', 'training', 'calib')
waymo_raw_tra['ims'] = os.path.join(base_data, 'waymo', 'training', 'image_0')
waymo_raw_tra['lab'] = os.path.join(base_data, 'waymo', 'training', 'label_0')
waymo_raw_tra['pre'] = os.path.join(base_data, 'waymo', 'training', 'prev_0')

waymo_tra = dict()
waymo_tra['cal'] = os.path.join(base_data, split, 'training', 'calib')
waymo_tra['ims'] = os.path.join(base_data, split, 'training', 'image_0')
waymo_tra['lab'] = os.path.join(base_data, split, 'training', 'label_0')
waymo_tra['pre'] = os.path.join(base_data, split, 'training', 'prev_0')

waymo_raw_val = dict()
waymo_raw_val['cal'] = os.path.join(base_data, 'waymo', 'validation', 'calib')
waymo_raw_val['ims'] = os.path.join(base_data, 'waymo', 'validation', 'image_0')
waymo_raw_val['lab'] = os.path.join(base_data, 'waymo', 'validation', 'label_0')
waymo_raw_val['pre'] = os.path.join(base_data, 'waymo', 'validation', 'prev_20')

waymo_val = dict()
waymo_val['cal'] = os.path.join(base_data, split, 'validation', 'calib')
waymo_val['ims'] = os.path.join(base_data, split, 'validation', 'image_0')
waymo_val['lab'] = os.path.join(base_data, split, 'validation', 'label_0')
waymo_val['pre'] = os.path.join(base_data, split, 'validation', 'prev_0')

tra_file = os.path.join(base_data, split, 'train.txt')
val_file = os.path.join(base_data, split, 'val.txt')

# mkdirs
mkdir_if_missing(waymo_tra['cal'])
mkdir_if_missing(waymo_tra['ims'])
mkdir_if_missing(waymo_tra['lab'])
mkdir_if_missing(waymo_tra['pre'])
mkdir_if_missing(waymo_val['cal'])
mkdir_if_missing(waymo_val['ims'])
mkdir_if_missing(waymo_val['lab'])
mkdir_if_missing(waymo_val['pre'])


print('Linking train')
text_file = open(tra_file, 'r')

imind = 0

for line in text_file:

    parsed = re.search('(\d+)', line)

    if parsed is not None:

        id = str(parsed[0])
        new_id = '{:015d}'.format(imind)

        if not os.path.exists(os.path.join(waymo_tra['cal'], str(new_id) + '.txt')):
            os.symlink(os.path.join(waymo_raw_tra['cal'], str(id) + '.txt'), os.path.join(waymo_tra['cal'], str(new_id) + '.txt'))

        if not os.path.exists(os.path.join(waymo_tra['ims'], str(new_id) + '.png')):
            os.symlink(os.path.join(waymo_raw_tra['ims'], str(id) + '.png'), os.path.join(waymo_tra['ims'], str(new_id) + '.png'))

        if not os.path.exists(os.path.join(waymo_tra['pre'], str(new_id) + '_01.png')):
            os.symlink(os.path.join(waymo_raw_tra['pre'], str(id) + '_01.png'), os.path.join(waymo_tra['pre'], str(new_id) + '_01.png'))

        if not os.path.exists(os.path.join(waymo_tra['pre'], str(new_id) + '_02.png')):
            os.symlink(os.path.join(waymo_raw_tra['pre'], str(id) + '_02.png'), os.path.join(waymo_tra['pre'], str(new_id) + '_02.png'))

        if not os.path.exists(os.path.join(waymo_tra['pre'], str(new_id) + '_03.png')):
            os.symlink(os.path.join(waymo_raw_tra['pre'], str(id) + '_03.png'), os.path.join(waymo_tra['pre'], str(new_id) + '_03.png'))

        if not os.path.exists(os.path.join(waymo_tra['lab'], str(new_id) + '.txt')):
            os.symlink(os.path.join(waymo_raw_tra['lab'], str(id) + '.txt'), os.path.join(waymo_tra['lab'], str(new_id) + '.txt'))

        imind += 1

text_file.close()

print('Linking val')
text_file = open(val_file, 'r')

imind = 0

for line in text_file:

    parsed = re.search('(\d+)', line)

    if parsed is not None:

        id = str(parsed[0])
        new_id = '{:015d}'.format(imind)

        if not os.path.exists(os.path.join(waymo_val['cal'], str(new_id) + '.txt')):
            os.symlink(os.path.join(waymo_raw['cal'], str(id) + '.txt'), os.path.join(waymo_val['cal'], str(new_id) + '.txt'))

        if not os.path.exists(os.path.join(waymo_val['ims'], str(new_id) + '.png')):
            os.symlink(os.path.join(waymo_raw['ims'], str(id) + '.png'), os.path.join(waymo_val['ims'], str(new_id) + '.png'))

        if not os.path.exists(os.path.join(waymo_val['pre'], str(new_id) + '_01.png')):
            os.symlink(os.path.join(waymo_raw['pre'], str(id) + '_01.png'), os.path.join(waymo_val['pre'], str(new_id) + '_01.png'))
        
        if not os.path.exists(os.path.join(waymo_val['pre'], str(new_id) + '_02.png')):
            os.symlink(os.path.join(waymo_raw['pre'], str(id) + '_02.png'), os.path.join(waymo_val['pre'], str(new_id) + '_02.png'))
        
        if not os.path.exists(os.path.join(waymo_val['pre'], str(new_id) + '_03.png')):
            os.symlink(os.path.join(waymo_raw['pre'], str(id) + '_03.png'), os.path.join(waymo_val['pre'], str(new_id) + '_03.png'))

        if not os.path.exists(os.path.join(waymo_val['lab'], str(new_id) + '.txt')):
            os.symlink(os.path.join(waymo_raw['lab'], str(id) + '.txt'), os.path.join(waymo_val['lab'], str(new_id) + '.txt'))

        imind += 1

text_file.close()

print('Done')
