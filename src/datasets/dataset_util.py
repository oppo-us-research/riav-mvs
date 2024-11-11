"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from DeepVideoMVS (https://github.com/ardaduz/deep-video-mvs)
# MIT license.
# ------------------------------------------------------------------------------------

import copy
import random
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool

import cv2
import os
import numpy as np
from PIL import Image
from path import Path

import torch

""" load our own moduels """
from config.config import Config
from src.utils.utils import pose_distance

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_flow_based_on_gt_depth(
    ref_proj_mat, #[4,4]
    src_proj_mat, #[4,4]
    depth, # [H, W]
    valid_depth_mask, #[H,W]
    ):
    height, width = depth.shape[0], depth.shape[1]
    #print (depth.shape, type(depth))
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    yy = yy.reshape((-1)) #[H*W]
    xx = xx.reshape((-1)) 
    X = np.vstack((xx, yy, np.ones_like(xx))) #[3, H*W]
    depth = depth.reshape((-1))
    X = np.vstack((X * depth, np.ones_like(depth))) #[4, H*W]
    X = np.matmul(np.linalg.inv(ref_proj_mat), X)
    X = np.matmul(src_proj_mat, X)
    X /= X[2]
    X = X[:2]

    xx_new = X[0].reshape((height, width)).astype(np.float32)
    yy_new = X[1].reshape((height, width)).astype(np.float32)
    flow_x = xx_new - xx.reshape((height, width))
    flow_y = yy_new - yy.reshape((height, width))
    valid =  (0 < xx_new) & (xx_new < width) & (0 < yy_new) & (yy_new < height)
    valid = valid & valid_depth_mask
    flow_x[~valid] = .0 # or another value (e.g., np.nan)
    flow_y[~valid] = .0 # or another value (e.g., np.nan)
    mask = np.zeros_like(xx_new)
    mask[valid] = 1.0
    flow = np.concatenate((flow_x[None], flow_y[None], mask[None]), axis=0)
    #print ("[***] flow = ", flow.shape)
    return flow # 3xHxW


## numpy version:
# warp based on GT depth map or a scalar depth value (e.g., D= 8)
def warp_based_on_depth(
    src_img,
    ref_img,
    ref_proj_mat,
    src_proj_mat,
    depth,
    mask = None
    ):

    height = ref_img.shape[0]
    width = ref_img.shape[1]
    assert src_img.shape[0] == height and src_img.shape[1] == width
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    #print("yy", yy.max(), yy.min())
    #print("D = ", D)
    yy = yy.reshape([-1])
    xx = xx.reshape([-1])
    X = np.vstack((xx, yy, np.ones_like(xx))) #[3, H*W]
    if isinstance(depth, np.ndarray):
        print ("[???] depth =", depth.shape, "src = ", src_img.shape, "ref = ", ref_img.shape)
        assert depth.shape[0] == height and depth.shape[1] == width
        if mask is not None:
            assert mask.shape[0] == height and mask.shape[1] == width
        D = depth.reshape([-1])
        is_scalar_D = False
    elif isinstance(depth, float):
        D = depth
        is_scalar_D = True
        print("single value D = ", D)
    X = np.vstack((X * D, np.ones_like(xx)))
    X = np.matmul(np.linalg.inv(ref_proj_mat), X)
    X = np.matmul(src_proj_mat, X)
    X /= X[2]
    X = X[:2]

    yy = X[0].reshape([height, width]).astype(np.float32)
    xx = X[1].reshape([height, width]).astype(np.float32)

    warped = cv2.remap(src_img, yy, xx, interpolation = cv2.INTER_LINEAR )
    #print ("yy = ", yy)
    #print (mask)
    if mask is not None:
        tmp_idx = mask < 0.5
        warped[:,:,0][tmp_idx] = 0 # Red
        warped[:,:,1][tmp_idx] = 1 # Green
        warped[:,:,2][tmp_idx] = 0 # Blue
    return warped

def is_valid_pair(reference_pose, measurement_pose, 
    pose_dist_min, pose_dist_max,
    t_norm_threshold=0.05, return_measure=False):
    
    combined_measure, R_measure, t_measure = pose_distance(reference_pose, measurement_pose)

    if pose_dist_min <= combined_measure <= pose_dist_max and t_measure >= t_norm_threshold:
        result = True
    else:
        result = False

    if return_measure:
        return result, combined_measure
    else:
        return result


def gather_pairs_train(
    poses, used_pairs, is_backward, 
    initial_pose_dist_min,
    initial_pose_dist_max
    ):

    sequence_length = len(poses)
    while_range = range(0, sequence_length)

    pose_dist_min = copy.deepcopy(initial_pose_dist_min)
    pose_dist_max = copy.deepcopy(initial_pose_dist_max)

    used_measurement_indices = set()

    # Gather pairs
    check_future = False
    pairs = []

    if is_backward:
        i = sequence_length - 1
        step = -1
        first_limit = 5
        second_limit = sequence_length - 5
    else:
        i = 0
        step = 1
        first_limit = sequence_length - 5
        second_limit = 5

    loosening_counter = 0
    while i in while_range:
        pair = (i, -1)

        if check_future:
            for j in range(i + step, first_limit, step):
                if j not in used_measurement_indices and (i, j) not in used_pairs:
                    valid = is_valid_pair(poses[i], poses[j], pose_dist_min, pose_dist_max)

                    if valid:
                        pair = (i, j)
                        pairs.append(pair)
                        used_pairs.add(pair)
                        used_pairs.add((pair[1], pair[0]))
                        used_measurement_indices.add(j)
                        pose_dist_min = copy.deepcopy(initial_pose_dist_min)
                        pose_dist_max = copy.deepcopy(initial_pose_dist_max)
                        i += step
                        check_future = False
                        loosening_counter = 0
                        break
        else:
            for j in range(i - step, second_limit, -step):
                if j not in used_measurement_indices and (i, j) not in used_pairs:
                    valid = is_valid_pair(poses[i], poses[j], pose_dist_min, pose_dist_max)

                    if valid:
                        pair = (i, j)
                        pairs.append(pair)
                        used_pairs.add(pair)
                        used_pairs.add((pair[1], pair[0]))
                        used_measurement_indices.add(j)
                        pose_dist_min = copy.deepcopy(initial_pose_dist_min)
                        pose_dist_max = copy.deepcopy(initial_pose_dist_max)
                        i += step
                        check_future = False
                        loosening_counter = 0
                        break

        if pair[1] == -1:
            if check_future:
                pose_dist_min = pose_dist_min / 1.1
                pose_dist_max = pose_dist_max * 1.1
                check_future = False
                loosening_counter += 1
                if loosening_counter > 1:
                    i += step
                    loosening_counter = 0
            else:
                check_future = True
        else:
            check_future = False

    return pairs


def crawl_subprocess_short(scene, dataset_path, count, progress, 
        min_pose_distance, # e.g, == 0.125;
        max_pose_distance # e.g, == 0.325;
        ):
    scene_path = Path(dataset_path) / scene
    if os.path.isfile(scene_path / "poses.txt"):
        poses = np.reshape(np.loadtxt(scene_path / "poses.txt"), newshape=(-1, 4, 4))
    elif "seven-scenes" in dataset_path:
        # for 7-scenes
        pose_filenames = sorted(scene_path.files("*pose.txt"))
        poses = []
        for pose_filename in pose_filenames:
            pose = np.reshape(np.loadtxt(pose_filename), newshape=(1, 4, 4))
            poses.append(pose)
        poses = np.concatenate(poses, axis=0).astype(np.float32)
    else:
        raise FileNotFoundError(f'Cannot find poses.txt - make sure your '
                                f'--data_path is set correctly, or you are '
                                f' loading seven-scenes dataset.'
                                )

    samples = []
    used_pairs = set()

    for multiplier in [(1.0, False), (0.666, True), (1.5, False)]:
        pairs = gather_pairs_train(poses, used_pairs,
                                   is_backward=multiplier[1],
                                   initial_pose_dist_min=multiplier[0]*min_pose_distance, 
                                   initial_pose_dist_max=multiplier[0]*max_pose_distance 
                                   )

        for pair in pairs:
            i, j = pair
            sample = {'scene': scene,
                      'indices': [i, j]}
            samples.append(sample)

    progress.value += 1
    print(progress.value, "/", count, end='\r')

    return samples


def crawl_subprocess_long(scene, dataset_path, count, progress, subsequence_length, 
        train_crawl_step,
        min_pose_distance,
        max_pose_distance
        ):
    scene_path = Path(dataset_path) / scene
    if os.path.isfile(scene_path / "poses.txt"):
        poses = np.reshape(np.loadtxt(scene_path / "poses.txt"), newshape=(-1, 4, 4))
    elif "seven-scenes" in dataset_path:
        # for 7-scenes
        pose_filenames = sorted(scene_path.files("*pose.txt"))
        poses = []
        for pose_filename in pose_filenames:
            pose = np.reshape(np.loadtxt(pose_filename), newshape=(1, 4, 4))
            poses.append(pose)
        poses = np.concatenate(poses, axis=0).astype(np.float32)
    
    elif "eth3d_undistorted_ours" in dataset_path:
        # for 'eth3d_undistorted_ours/low-res'
        eth3d_lowres_path = scene_path / "cams_1"
        cam_filenames = sorted(eth3d_lowres_path.files("*_cam.txt"))
        poses = []
        for cam_filename in cam_filenames:
            with open(cam_filename) as f:
                lines = [line.rstrip() for line in f.readlines()]
            # extrinsics: line [1,5), 4x4 matrix
            extrinsics = np.fromstring(' '.join(lines[1:5]), 
                            dtype=np.float32, sep=' ').reshape((4, 4))
            pose = np.linalg.inv(extrinsics).reshape((1,4,4))
            poses.append(pose)
        # To follow the cam-to-world pose as in ScanNet;
        # change this eth3d extrinsics (i.e., world-to-cam) to cam-to-world 
        poses = np.concatenate(poses, axis=0).astype(np.float32)
    else:
        raise FileNotFoundError(f'Cannot find poses.txt - make sure your '
                                f'--data_path is set correctly, or you are '
                                f' loading seven-scenes dataset.'
                                ) 

    sequence_length = np.shape(poses)[0]

    used_pairs = set()

    usage_threshold = 1
    used_nodes = dict()
    for i in range(sequence_length):
        used_nodes[i] = 0

    calculated_step = train_crawl_step
    samples = []
    for offset, multiplier, is_backward in [(0 % calculated_step, 1.0, False),
                                            (1 % calculated_step, 0.666, True),
                                            (2 % calculated_step, 1.5, False),
                                            (3 % calculated_step, 0.8, True),
                                            (4 % calculated_step, 1.25, False),
                                            (5 % calculated_step, 1.0, True),
                                            (6 % calculated_step, 0.666, False),
                                            (7 % calculated_step, 1.5, True),
                                            (8 % calculated_step, 0.8, False),
                                            (9 % calculated_step, 1.25, True)]:

        if is_backward:
            start = sequence_length - 1 - offset
            step = -calculated_step
            limit = subsequence_length
        else:
            start = offset
            step = calculated_step
            limit = sequence_length - subsequence_length + 1

        for i in range(start, limit, step):
            if used_nodes[i] > usage_threshold:
                continue

            sample = {'scene': scene,
                      'indices': [i]}

            previous_index = i
            valid_counter = 1
            any_counter = 1
            reached_sequence_limit = False
            while valid_counter < subsequence_length:

                if is_backward:
                    j = i - any_counter
                    reached_sequence_limit = j < 0
                else:
                    j = i + any_counter
                    reached_sequence_limit = j >= sequence_length

                if not reached_sequence_limit:
                    current_index = j

                    check1 = used_nodes[current_index] <= usage_threshold
                    check2 = (previous_index, current_index) not in used_pairs
                    check3 = is_valid_pair(poses[previous_index],
                                           poses[current_index],
                                           multiplier * min_pose_distance,
                                           multiplier * max_pose_distance,
                                           t_norm_threshold = multiplier * min_pose_distance * 0.5)

                    if check1 and check2 and check3:
                        sample['indices'].append(current_index)
                        previous_index = copy.deepcopy(current_index)
                        valid_counter += 1
                    any_counter += 1
                else:
                    break

            if not reached_sequence_limit:
                previous_node = sample['indices'][0]
                used_nodes[previous_node] += 1
                for current_node in sample['indices'][1:]:
                    used_nodes[current_node] += 1
                    used_pairs.add((previous_node, current_node))
                    used_pairs.add((current_node, previous_node))
                    previous_node = copy.deepcopy(current_node)

                samples.append(sample)

    progress.value += 1
    print(progress.value, "/", count, end='\r')
    return samples


def crawl(dataset_path, scenes, subsequence_length, 
        train_crawl_step,
        min_pose_distance, 
        max_pose_distance, 
        num_workers=1):
    
    pool = Pool(num_workers)
    manager = Manager()

    count = len(scenes)
    progress = manager.Value('i', 0)

    samples = []

    if subsequence_length == 2:
        print ("[***] here subsequence_length = 2")
        for scene_samples in pool.imap_unordered(partial(crawl_subprocess_short,
                                                         dataset_path=dataset_path,
                                                         count=count,
                                                         progress=progress,
                                                         min_pose_distance=min_pose_distance,
                                                         max_pose_distance=max_pose_distance,
                                                        ), scenes):
            samples.extend(scene_samples)

    else:
        for scene_samples in pool.imap_unordered(partial(crawl_subprocess_long,
                                                        dataset_path=dataset_path,
                                                        count=count,
                                                        progress=progress,
                                                        subsequence_length=subsequence_length,
                                                        train_crawl_step=train_crawl_step,
                                                        min_pose_distance=min_pose_distance,
                                                        max_pose_distance=max_pose_distance
                                                        ), scenes):
            samples.extend(scene_samples)

    random.shuffle(samples)

    return samples

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_depth(path, scaling=1000.0):
    depth = np.load(path).astype(np.float32) / scaling
    return depth