CUR_DIR=$(pwd)
PROJ_DIR=$(dirname $CUR_DIR)

docker run -it \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$XAUTHORITY:/root/.Xauthority:rw" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --hostname="inside-DOCKER" \
        --name="m3drpn" \
        --volume $PROJ_DIR/data:/M3D-RPN/data \
        --volume $PROJ_DIR/lib:/M3D-RPN/lib \
        --volume $PROJ_DIR/models:/M3D-RPN/models \
        --volume $PROJ_DIR/scripts:/M3D-RPN/scripts \
        --volume $(readlink -f ../data/kitti/training):/M3D-RPN/data/kitti/training \
        --volume $(readlink -f ../data/kitti/testing):/M3D-RPN/data/kitti/testing \
        --rm \
    m3drpn-docker bash