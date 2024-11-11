#!/bin/bash

echo "------------------->"
echo "------------------->"
echo "----start building docker image ..."

PYTHON_VERSION=3.10

USER_NAME=${1:-'ccj'}
VER=${2:-1.0}
DOCKER_TAG=${USER_NAME}/riav-mvs:$VER
echo "Will build docker container $DOCKER_TAG ..."

docker build --tag $DOCKER_TAG \
    --force-rm \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    --build-arg USER_NAME=$USER_NAME \
    --build-arg GROUP_NAME=$(id -gn) \
    --build-arg python=${PYTHON_VERSION} \
    -f Dockerfile \
    .

#rm -r ./files
