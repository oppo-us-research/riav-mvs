#!/bin/bash
echo "Current user is : ${USER}"

USER_NAME=${1:-'ccj'}
GROUP=$(id -gn)
VER=${2:-1.0}
DOCKER_IMAGE=${USER_NAME}/riav-mvs:$VER

u=$(id -un)
g=$(id -gn)
echo $u $g
echo "DOCKER_IMAGE=$DOCKER_IMAGE"

docker run --gpus all --ipc=host \
    -e HOSTNAME=${hostname} \
    -u $USER_NAME:$GROUP \
    -v "/nfs:/nfs/" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -it $DOCKER_IMAGE /bin/bash

