#"""
#* Copyright (c) 2024 OPPO. All rights reserved.
#* Under license: MIT
#* For full license text, see LICENSE file in the repo root
#"""

# this Dockerfile is adapted from:
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Or you can use this conda docker
#FROM continuumio/miniconda3:latest


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

## > see this Dockerfile example at https://github.com/TRI-ML/dgp/blob/master/Dockerfile;
ARG python=3.10

ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive


# Using -y is convenient to automatically answer yes to all the questions;
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    git \
    build-essential \
    graphviz \
    cmake curl \
    libpng-dev \
    libjpeg-dev libjpeg-dev \
    libgl1-mesa-dev \
    ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    vim tmux \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    # required by OpenCV3
    libgtk2.0-0 libcanberra-gtk-module \
    python${PYTHON_VERSION}-tk \
    # eth3d dataset etc point cloud evaluation
    libboost-all-dev libeigen3-dev libpcl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py


COPY dev_requirements.txt /tmp/
COPY bashrc_extra /tmp/
# install PyTorch etc libraries;
RUN pip install -r /tmp/dev_requirements.txt

# Create a non-root user and switch to it
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME='dummy'
ARG GROUP_NAME='dummy'

## (Optional) if more groups to add, then uncomment the following;
#ARG GROUP2_ID=1001
#ARG GROUP3_ID=1002
#ARG GROUP2_NAME='dummy2'
#ARG GROUP3_NAME='dummy3'

RUN addgroup --gid $GROUP_ID $GROUP_NAME

## (Optional) if more groups to add, then uncomment the following;
# RUN addgroup --gid $GROUP_ID $GROUP_NAME && \
#     addgroup --gid $GROUP2_ID $GROUP2_NAME && \
#     addgroup --gid $GROUP3_ID $GROUP3_NAME

# Create a user 'xyz' under 'xyzgroup'
RUN useradd -rm -d /home/$USER_NAME --shell /bin/bash  \
    --uid $USER_ID --gid $GROUP_ID \
    -G $GROUP_NAME $USER_NAME

## (Optional) if more groups to add, then uncomment the following;
#RUN useradd -rm -d /home/$USER_NAME --shell /bin/bash  \
#    --uid $USER_ID --gid $GROUP_ID \
#    -G $GROUP_NAME,$GROUP2_NAME,$GROUP3_NAME $USER_NAME


EXPOSE 22
USER $USER_NAME:$GROUP_NAME
WORKDIR /home/$USER_NAME


##################################################
###  Environment variable setup
##################################################
ENV PATH=/home/$USER_NAME/.local/bin:$PATH
ENV PATH=/home/$USER_NAME/bin:$PATH

### color prompt ###
ENV TERM='xterm-color'

##################################################
###  Bashrc
##################################################
### (Optional) bashrc ###
COPY bashrc_extra /home/$USER_NAME/.bashrc_extra
RUN echo "source ~/.bashrc_extra" >> ~/.bashrc
