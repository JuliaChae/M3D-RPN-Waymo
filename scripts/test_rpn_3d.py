# -----------------------------------------
# python modules
# -----------------------------------------
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import os

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.waymo_imdb_util import *

conf_path = '/M3D-RPN/output/waymo_3d_multi_main_left/conf.pkl'
weights_path = '/M3D-RPN/output/waymo_3d_multi_main_left/weights/model_400000_pkl'
label_path = 

cam_num = 1
cam_train_view = "frontleft"
cam_test_num = 3
cam_test_view = "left"

# load config
conf = edict(pickle_read(conf_path))
conf.pretrained = None

data_path = os.path.join(os.getcwd(), 'data')
results_path = os.path.join('output', 'eval/waymo_results_' + cam_view, '400000', cam_test_view, 'data')
save_path = os.path.join('output', 'eval/waymo_results_' + cam_view, '400000', cam_test_view, 'eval.txt')

# make directory
mkdir_if_missing(results_path, delete_if_exist=True)

# -----------------------------------------
# torch defaults
# -----------------------------------------

# defaults
init_torch(conf.rng_seed, conf.cuda_seed)

# -----------------------------------------
# setup network
# -----------------------------------------

# net
net = import_module('models.' + conf.model).build(conf)

# load weights
load_weights(net, weights_path, remove_module=True)

# switch modes for evaluation
net.eval()

print(pretty_print('conf', conf))

# -----------------------------------------
# generate waymo test data
# -----------------------------------------

test_waymo_3d(conf.dataset_test, net, conf, results_path, data_path, use_log=False)

# -----------------------------------------
# evaluate generated test data
# -----------------------------------------
from eval.evaluate import *
evaluate(label_path, results_path, save_path):
