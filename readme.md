# M3D-RPN: Monocular 3D Region Proposal Network for Object Detection on Waymo Multiview Dataset

This repository is modified from Garrick Brazil and Xiaoming Liu's [Monocular 3D Region Proposal Network](https://github.com/garrickbrazil/M3D-RPN) based on their ICCV 2019 [arXiv report](https://arxiv.org/abs/1907.06038). Please see their [project page](http://cvlab.cse.msu.edu/project-m3d-rpn.html) for more information and their repository to train and evaluate on the default Kitti Dataset. 

## Overview

This project was used to train and evaluate the M3D-RPN Monocular 3D Object Detector on the various camera views of the [Waymo Open Dataset](https://waymo.com/open/). This was done as a part of a research project to investigate the effect of camera perspective on monocular object detection training and performance, in order to optimize the use of recent multiview datasets such as Waymo and nuScenes. 

This repository contains the following modifications to the original repository. 
*    Integration of Waymo-Kitt-Adapter in order to streamline the dataloading process 
*    Comprehensive Docker environment with all required dependencies to run the project on 
*    Set-up scripts to develop appropriate data split for training, validation, testing for all camera views 
*    New data object class for Waymo to handle image, label and calibration preparation, augmentation and loading
*    Modified comfiguration files to optimize training on the Waymo dataset
*    Embedded Waymo evaluation with 2D and 3D mAP and BEV metrics, modified from [kitti-object-eval](https://github.com/traveller59/kitti-object-eval-python) 

## Docker Setup
To use this repository in docker please make sure you have `nvidia-docker` installed.
```
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Install `nvidia-container-runtime`
```
sudo apt install nvidia-container-runtime
```

Edit/create `/etc/docker/daemon.json` with:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

Restart docker daemon
```
sudo systemctl restart docker
```


## Build Docker Image
```
./build.sh
```

## Run the Docker Container
```
./run.sh
```

## Data Set up
### WAYMO Dataset
Please download the official [Waymo Open Dataset](https://waymo.com/open/download/) and organize the downloaded files in any location as follows:

```
├── Waymo
│   ├── original
│   │   │──training
│   │   │   ├──training_0000
│   │   │   │   ├─.tfrecord files
│   │   │   ├──training_0001
│   │   │   │   ├─.tfrecord files
│   │   │   ├──...
│   │   │──validation
│   │   │   ├──validation_0000
│   │   │   │   ├─.tfrecord files
│   │   │   ├──validation_0001
│   │   │   │   ├─.tfrecord files
│   │   │   ├──...
│   │   │──testing
│   │   │   ├──testing_0000
│   │   │   │   ├─.tfrecord files
│   │   │   ├──testing_0001
│   │   │   │   ├─.tfrecord files
│   │   │   ├──...
```
Once you have extracted the dataset and organized the files, you need to run the Waymo-Kitti adapter to reformat the data appropriately. To do this, clone the repo https://github.com/JuliaChae/Waymo-Kitti-Adapter and follow the instructions in its README file. Convert all training, testing and validation files.

After runing the adapter, the Waymo data path should look something like with reformatted dataset in the "adapted" folder
```
...
├── Waymo
│   ├── original
│   │   │──training & testing & validation
│   ├── adapted
│   │   │──training
│   │   │   ├──calib & velodyne & label_0 & image_0
│   │   │──validation
│   │   │   ├──calib & velodyne & label_0 & image_0
│   │   │──testing
│   │   │   ├──calib & velodyne & label_0

```

Finally, symlink it using the following:
```shell
mkdir data/waymo
ln -s ${ADAPTED_WAYMO_DIR} data/waymo
```
Then use the following scripts to extract the data splits, which use softlinks to the above directory for efficient storage. Modify the camera view argument at the top of the code to correspond to the views that you are working with

    ```
    python data/waymo_split/setup_split.py
    ```

## Training

We use [visdom](https://github.com/facebookresearch/visdom) for visualization and graphs. Optionally, start the server by command line

```
python -m visdom.server -port 8100 -readonly
```
The port can be customized in *scripts/config* files. The training monitor can be viewed at [http://localhost:8100](http://localhost:8100)

Training is split into a warmup and main configurations. Review the configurations in *scripts/config* for details. 

``` 
// First train the warmup (without depth-aware)
python scripts/train_rpn_3d.py --config=waymo_3d_multi_warmup

// Then train the main experiment (with depth-aware)
python scripts/train_rpn_3d.py --config=waymo_3d_multi_main
```

If your training is accidentally stopped, you can resume at a checkpoint based on the snapshot with the *restore* flag. 
For example to resume training starting at iteration 10k, use the following command.

```
python scripts/train_rpn_3d.py --config=waymo_3d_multi_main --restore=10000
```

## Testing

Testing requires paths to the configuration file and model weights, exposed variables near the top *scripts/test_rpn_3d.py*. To test a configuration and model, simply update the variables and run the test file as below. 

```
python scripts/test_rpn_3d.py 
```
