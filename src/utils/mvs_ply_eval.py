"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from IterMVS (https://github.com/FangjinhuaWang/IterMVS)
# MIT license.
# ------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import argparse  # argument parser.
from datetime import datetime
import os
import sys
import numpy as np


import time
import cv2
from PIL import Image
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import json
from typing import Tuple, List

from struct import pack, unpack
import shutil
from plyfile import PlyData, PlyElement
from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
cudnn.benchmark = True

""" load our own moduels """
from src.utils import pfmutil as pfm
from src.dtu_ply_eval import DTU_ply_eval 
from src.evaluate_util import (
    compute_errors_v2, 
    manydepth_colormap, 
    error_colormap)
# to load yaml files
from src.utils.yaml_util import load_config



# mvs datasets evaluation
# code is adopted from https://github.com/FangjinhuaWang/IterMVS/blob/main/eval.py

parser = argparse.ArgumentParser(description='depth filter, fuse and point cloud evaluation')

parser.add_argument('--method', required=True, help='method name, to spefiy ply name')
parser.add_argument('--task', required=True, choices=[
                                'ply_fuse', 'ply_eval', 
                                'organize_dir', 'depth_eval', 
                                'plot_depth_eval_fig',
                                'plot_mvs_ply_eval_fig',
                                'all',
                                'organize_gipuma_ply',
                                'save_video',
                                ])
parser.add_argument('--dataset', default='dtu',
                    choices=['dtu', 'eth3d', 'tanks', 'scannet', 'eth3d_yao_eval'], 
                    help='select dataset')

parser.add_argument('--machine_name', help='machine_name')
parser.add_argument('--csv_dir', help='csv file dir')
parser.add_argument('--data_path', help='data path, to load image, camera etc')
parser.add_argument('--ply_data_path', help='data path, to load GT point cloud')
parser.add_argument('--idx_2_name_map_txt', help='txt file keeps the mapping '
                                                 'from index-base name to file path in disk')
parser.add_argument('--split', default='intermediate', help='select data')
parser.add_argument('--eth3d_reso', default='high-res', help='eth3d high-res or low-res sets')

parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', type=str, default='false', help='display depth images and masks')
parser.add_argument('--geo_pixel_thres', type=float, default=1, 
                    help='pixel threshold for geometric consistency filtering')
parser.add_argument('--geo_depth_thres', type=float, default=0.01, 
                    help='depth threshold for geometric consistency filtering')


parser.add_argument('--prob_threshold', type=float, default=0.9, 
                    help='prob confidence')


parser.add_argument('--num_workers', default=2, type=int, help='for multiprocessing.pool')

# newly added for Gipuma depth filtering
parser.add_argument('--disp_threshold', type=float, default = '0.25')
parser.add_argument('--num_consistent', type=float, default = '3')
parser.add_argument('--scans', 
             type=str, 
             default='', 
             help='specify scans')

# newly added for ply fusion to load yaml file for parameters per scene;
parser.add_argument('--config_file', help='yaml file to load hyperparameters '
                    'for ply fusion and evaluation')


def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# parse arguments and check
args = parser.parse_args()
if args.scans and args.scans != 'None':
    args.scans = [int(item) for item in args.scans.split(",")]

print("argv:", sys.argv[1:])
print("args.scans = ", args.scans)
print_args(args)

if args.dataset=="dtu":
    img_wh=(1600, 1152)
    MIN_DEPTH, MAX_DEPTH = 425, 935 # millimeter
    bad_x_thred=[0.125, 0.25, 0.5, 1.0] # in millimeters (mm)
elif args.dataset=="tanks":
    img_wh=(1920, 1024)
elif args.dataset=="eth3d":
    img_wh = (1920,1280)
elif args.dataset=="eth3d_yao_eval":
    img_wh = (-1,-1)
else:
    raise NotImplementedError

#----- global data -----
dtu_test_scan_idxs = [
    # 22 scenes in test;
    1, 4, 9, 10, 11, 12, 13, 15, 
    23, 24, 29, 
    32, 33, 34, 48, 49, 62, 75, 77, 
    110, 114, 118
    ]


eth3d_scans = {
    'high-res/test': ['botanical_garden', 'boulders', 'bridge', 'door',
            'exhibition_hall', 'lecture_room', 'living_room', 'lounge',
            'observatory', 'old_computer', 'statue', 'terrace_2'
            ], # 12
    
    'high-res/train': ['courtyard', 'delivery_area', 'electro', 'facade',
            'kicker', 'meadow', 'office', 'pipes', 'playground',
            'relief', 'relief_2', 'terrace', 'terrains'
            ], # 13
    
    # 5 scans
    'low-res/test': ['lakeside', 'sand_box', 'storage_room', 'storage_room_2', 'tunnel'],
    
    # 2 scans
    'low-res/val':['playground', 'terrains'],

    # 5 scans
    'low-res/train':['delivery_area', 'electro', 'forest', 'playground', 'terrains'],
}

eth3d_geo_mask_thres = {
    'high-res/test': {
            'botanical_garden':1,  # 30 images, outdoor
            'boulders':1, # 26 images, outdoor
            'bridge':2,  # 110 images, outdoor
            'door':2, # 6 images, indoor
            'exhibition_hall':2,  # 68 images, indoor
            'lecture_room':2, # 23 images, indoor
            'living_room':2, # 65 images, indoor
            'lounge':1,# 10 images, indoor
            'observatory':2, # 27 images, outdoor
            'old_computer':2, # 54 images, indoor
            'statue':2,  # 10 images, indoor
            'terrace_2':2 # 13 images, outdoor
            },
    
    'high-res/train': {
            'courtyard':1,  # 38 images, outdoor
            'delivery_area':2, # 44 images, indoor
            'electro':1,  # 45 images, outdoor
            'facade':2, # 76 images, outdoor
            'kicker':1,  # 31 images, indoor
            'meadow':1, # 15 images, outdoor
            'office':1, # 26 images, indoor
            'pipes':1,# 14 images, indoor
            'playground':1, # 38 images, outdoor
            'relief':1, # 31 images, indoor
            'relief_2':1, # 31 images, indoor
            'terrace':1,  # 23 images, outdoor
            'terrains':2 # 42 images, indoor
            },
    
    'low-res/test':{
        # (???) numbers added by CCJ:
        'lakeside': 2, 
        'sand_box': 2, 
        'storage_room': 2, 
        'storage_room_2': 2, 
        'tunnel': 2
        },

    'low-res/train':{
        'delivery_area': 2, 
        'electro': 1, 
        'forest': 2, 
        #'playground': 1, 
        #'terrains': 2
    },
    'low-res/val':{
        'playground': 1, 
        'terrains': 2
    },
 }
        

tanks_scans = {
        "intermediate": ['Family', 'Francis', 'Horse', 'Lighthouse',
                            'M60', 'Panther', 'Playground', 'Train'],
        "advanced":  ['Auditorium', 'Ballroom', 'Courtroom',
                    'Museum', 'Palace', 'Temple'],
    }

tanks_geo_mask_thres = {
        "intermediate": {
                            'Family': 5,
                            'Francis': 6,
                            'Horse': 5,
                            'Lighthouse': 6,
                            'M60': 5,
                            'Panther': 5,
                            'Playground': 5,
                            'Train': 5
                        },

        "advanced": { 'Auditorium': 3,
                            'Ballroom': 4,
                            'Courtroom': 4,
                            'Museum': 4,
                            'Palace': 5,
                            'Temple': 4
                    },

    }


# read intrinsics and extrinsics
def read_camera_parameters(filename, scale=None, crop=None, flag=None):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    
    if scale is not None:
        intrinsics[:2, :] *= scale
    if flag is not None:
        if (flag==0):
            intrinsics[0,2] -= crop
        else:
            intrinsics[1,2] -= crop
  
    return intrinsics, extrinsics

def read_mask(filename, img_wh):
    img = Image.open(filename)
    mask_hr = np.array(img, dtype=np.float32) # high resolution
    h, w = mask_hr.shape
    assert h == 1200 and w == 1600, "Loading wrong depth mask size"
    mask = cv2.resize(mask_hr, img_wh, interpolation=cv2.INTER_NEAREST)
    return mask


def load_image(filename: str, target_size: Tuple[int, int]) -> Tuple[np.ndarray, int, int]:
    """
    Reads and resizes an image from the given file.

    Args:
        filename: Path to the image file.
        target_size: Desired (width, height) to resize the image.

    Returns:
        A tuple containing the resized image as a NumPy array (scaled to [0, 1]), 
        and the original image dimensions (height, width).
    
    Raises:
        AssertionError: If the image's original size is not 1200x1600.
    """
    image = Image.open(filename)
    # Convert to NumPy array and scale pixel values to range [0, 1]
    np_image = np.array(image, dtype=np.float32) / 255.0
    original_height, original_width, _ = np_image.shape
    assert original_height == 1200 and original_width == 1600, "Invalid image size, expected 1200x1600."

    # Resize the image to the specified target size
    resized_image = cv2.resize(np_image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image, original_height, original_width

def save_binary_mask(filename: str, mask: np.ndarray) -> None:
    """
    Saves a binary mask to a file as an 8-bit grayscale image.

    Args:
        filename: Path where the mask image will be saved.
        mask: A boolean mask represented as a NumPy array.
    
    Raises:
        AssertionError: If the mask data type is not np.bool_.
    """
    assert mask.dtype == np.bool_, "Expected mask of type np.bool_."
    uint8_mask = mask.astype(np.uint8) * 255
    Image.fromarray(uint8_mask).save(filename)

def save_depth_image(filename: str, depth_map: np.ndarray) -> None:
    """
    Saves a depth map to a file as an 8-bit grayscale image.

    Args:
        filename: Path where the depth image will be saved.
        depth_map: A depth map represented as a NumPy array (float32 scaled to [0, 1]).
    """
    depth_uint8 = (depth_map.astype(np.float32) * 255).astype(np.uint8)
    Image.fromarray(depth_uint8).save(filename)

def read_pair_file(filename: str) -> List[Tuple[int, List[int]]]:
    """
    Reads a pair file that defines reference and source views for multi-view stereo.

    Args:
        filename: Path to the pair file.

    Returns:
        A list of tuples, each containing a reference view index and a list of 
        source view indices.
    """
    pairs = []
    with open(filename, 'r') as file:
        num_viewpoints = int(file.readline().strip())
        # 49 viewpoints
        for _ in range(num_viewpoints):
            ref_view_idx = int(file.readline().strip())
            src_view_indices = [int(x) for x in file.readline().strip().split()[1::2]]
            if src_view_indices:
                pairs.append((ref_view_idx, src_view_indices))
    return pairs


# project the reference point cloud into the source view, then project back
def reproject_with_depth(
        depth_ref, 
        intrinsics_ref, 
        extrinsics_ref, 
        depth_src, 
        intrinsics_src, 
        extrinsics_src):
    
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                    geo_pixel_thres, geo_depth_thres):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = \
                        reproject_with_depth(depth_ref, intrinsics_ref, 
                                extrinsics_ref, depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(
        dataset, 
        scan,
        scan_folder,
        out_folder, 
        plyfilename, 
        img_wh,
        geo_pixel_thres, 
        geo_depth_thres, # threshold for relative_depth_diff
        prob_threshold, # photometric loss;
        num_view_consistent # at least 3 source views matched
        ):
    
    # the pair file
    if str(dataset).upper() == "DTU":
        n_images = 49

    pair_file = os.path.join(scan_folder, "pair.txt") 
    cam_folder = Path(scan_folder)/'cams_1'
    img_folder = Path(scan_folder)/'images'
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)
    
    # for each reference view and the corresponding source views
    for i, (ref_view, src_views) in enumerate(pair_data):

        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            cam_folder/f'{ref_view:0>8}_cam.txt')
         
        ref_img, original_h, original_w = load_image(
            img_folder/f'{ref_view:0>8}.jpg', img_wh)
        
        ref_intrinsics[0] *= img_wh[0]/original_w
        ref_intrinsics[1] *= img_wh[1]/original_h
        
        # load the estimated depth of the reference view
        ref_depth_est = pfm.readPFM(Path(out_folder)/f'depth_est/{ref_view:0>8}.pfm')
        ref_dep_h, ref_dep_w = ref_depth_est.shape
        assert ref_dep_h == img_wh[1] and ref_dep_w == img_wh[0], "Ref depth map size is Wrong!"
        # load the photometric mask of the reference view
        confidence = pfm.readPFM(Path(out_folder)/f'confidence/{ref_view:0>8}.pfm')
        photo_mask = confidence > prob_threshold
        
        all_srcview_depth_ests = []
        #all_srcview_x = []
        #all_srcview_y = []
        #all_srcview_geomask = []
        
        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                cam_folder/f'{src_view:0>8}_cam.txt')
            _, original_h, original_w = load_image(
                img_folder/f'{src_view:0>8}.jpg', img_wh)
            src_intrinsics[0] *= img_wh[0]/original_w
            src_intrinsics[1] *= img_wh[1]/original_h
            
            # the estimated depth of the source view
            src_depth_est = pfm.readPFM(Path(out_folder)/f'depth_est/{src_view:0>8}.pfm')
            src_dep_h, src_dep_w = src_depth_est.shape
            assert src_dep_h == img_wh[1] and src_dep_w == img_wh[0], "Src depth map size is Wrong!"

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
                                                ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                src_depth_est, src_intrinsics, src_extrinsics,
                                                geo_pixel_thres, 
                                                geo_depth_thres)

            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            #all_srcview_x.append(x2d_src)
            #all_srcview_y.append(y2d_src)
            #all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= num_view_consistent
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_binary_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_binary_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_binary_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{:.4f}/{:.4f}/{:.4f}".format(
            scan, ref_view, photo_mask.mean(),geo_mask.mean(), final_mask.mean()))
        
        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print(" [!!!] valid_points", valid_points.mean())
        
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        color = ref_img[valid_points]
        
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))
    

    print("saving the final model to", plyfilename)
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)    
    print("   Done! Saved the final model to", plyfilename)
        


def print_metrics(mean_errors, bad_x_thred, mean_median_aligned_errors = None):
    if len(bad_x_thred) == 3:
        ELE_NUM = 12
        print("\n  " + ("{:>9} | " * 12).format( "abs", "abs_inv", "abs_rel", "sq_rel", "rmse", 
                                                "rmse_log", "a1", "a2", "a3", 
                                                "bad-%.1f"%(bad_x_thred[0]),
                                                "bad-%.1f"%(bad_x_thred[1]),
                                                "bad-%.1f"%(bad_x_thred[2]),
                                                ))
    elif len(bad_x_thred) == 4:
        ELE_NUM = 13
        print("\n  " + ("{:>9} | " * 13).format( "abs", "abs_inv", "abs_rel", "sq_rel", "rmse", 
                                                "rmse_log", "a1", "a2", "a3", 
                                                "bad-%.3f"%(bad_x_thred[0]),
                                                "bad-%.3f"%(bad_x_thred[1]),
                                                "bad-%.3f"%(bad_x_thred[2]),
                                                "bad-%.3f"%(bad_x_thred[3]),
                                                ))
    print( ("&{: 9.5f}  " * ELE_NUM).format( * mean_errors.tolist()) + "\\\\" )
    if mean_median_aligned_errors is not None:
        print (" ==> mean_median_aligned_errors")
        print( ("&{: 9.5f}  " * ELE_NUM).format( * mean_median_aligned_errors.tolist()) + "\\\\" )
    
    print( "\n-> Done!" )

def eval_depth(args):
    errors = []
    errors_mask = defaultdict(list)
    #errors_per_scan = defaultdict(list)
    errors_mask_per_scan = {}
    avg_err_mask_per_scan = {}
    
    def get_avg_err(
        errors_list: list,
        bad_x_thred: list,
        is_verbose: bool = False
        ):
        errors = np.array(errors_list)
        nan_num = 0
        for j in range(errors.shape[0]):
            if any(np.isnan(errors[j])):
                nan_num += 1
                print ("nan found at idx {} /{}, = {}".format(j, errors.shape[0], errors[j]))
        mean_errors = np.nanmean(errors, axis=0) # skip: the img_id at last dim;
        if is_verbose:
            print_metrics(mean_errors, bad_x_thred)
        
        error_names = ["abs", "abs_inv", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

        for i in range(len(bad_x_thred)):
            error_names.append("bad-%.3f"%(bad_x_thred[i]))
        
        mean_errors_dict = {}
        for i, err_nm in enumerate(error_names):
            mean_errors_dict[err_nm] = mean_errors[i]
        return mean_errors_dict

    if args.dataset=="dtu":
        for idx, scan_id in enumerate(dtu_test_scan_idxs):
            scan = f"scan{scan_id}"
            
            print (f"Processing scan {scan}: {idx+1}/{len(dtu_test_scan_idxs)}")
            scan_folder = os.path.join(args.data_path, scan)
            out_folder = os.path.join(args.outdir, scan) 
            gt_depth_mask_folder = os.path.join(args.data_path, "Depths_raw/"+scan) 
    
            # the pair file
            pair_file = os.path.join(scan_folder, "pair.txt")
            pair_data = read_pair_file(pair_file)
            nviews = len(pair_data)

            # for each reference view and the corresponding source views
            for i, (ref_view, src_views) in enumerate(pair_data):
                # load the estimated depth of the reference view
                ref_depth_est = pfm.readPFM(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))
                #TODO: using gt_depth for verification; ???
                ref_depth_gt = pfm.readPFM(os.path.join(gt_depth_mask_folder, 'depth_map_{:0>4}.pfm'.format(ref_view)))
                gt_depth_wh = ref_depth_gt.shape[1], ref_depth_gt.shape[0]
                assert ref_depth_gt.shape[0] == 1200 and ref_depth_gt.shape[1] == 1600, "Loading wrong GT depth size"
                #ref_depth_gt = cv2.resize(ref_depth_gt, img_wh, interpolation=cv2.INTER_NEAREST)
                ref_depth_est = cv2.resize(ref_depth_est, gt_depth_wh, interpolation=cv2.INTER_NEAREST)
                #print ("ref_depth_est: ", ref_depth_est.shape)
                
                #pred_dep = cv2.resize(pred_dep, (gt_width, gt_height), interpolation=cv2.INTER_NEAREST)
                mask = np.logical_and(ref_depth_gt >= MIN_DEPTH, ref_depth_gt <= MAX_DEPTH)

                
                pred_depth = ref_depth_est[mask]
                gt_depth = ref_depth_gt[mask]
                
                cur_error = compute_errors_v2(gt_depth, pred_depth, \
                    min_depth = MIN_DEPTH, max_depth = MAX_DEPTH, bad_x_thred=bad_x_thred)
                
                errors.append(cur_error)
                #errors_per_scan[scan].append(cur_error)
                
                #several masks
                for msk_name in ['geo', 'photo', 'final', 'no_mask']:
                    if msk_name == 'no_mask':
                        tmp_err = cur_error # from regular mask by depth_min/max;
                        errors_mask[msk_name].append(tmp_err)
                    else: # other masks
                        mask_fn = os.path.join(out_folder, f'mask/{ref_view:0>8}_{msk_name}.png')
                        if os.path.exists(mask_fn):
                            tmp_mask = np.array(Image.open(mask_fn), dtype=np.float32)
                            tmp_mask = cv2.resize(tmp_mask, gt_depth_wh, interpolation=cv2.INTER_NEAREST)
                            tmp_mask = tmp_mask > 2.0
                            tmp_err = compute_errors_v2(
                                        ref_depth_gt[tmp_mask],
                                        ref_depth_est[tmp_mask], 
                                        min_depth = MIN_DEPTH, 
                                        max_depth = MAX_DEPTH, 
                                        bad_x_thred = bad_x_thred
                                    )
                            errors_mask[msk_name].append(tmp_err)
                        
                    if i == 0:
                        errors_mask_per_scan[msk_name] = {}
                        errors_mask_per_scan[msk_name] = {scan: []}
                    else:
                        errors_mask_per_scan[msk_name][scan].append(tmp_err)

            
            # per_scan evaluation
            # save avg error per scan
            avg_err_mask_per_scan[scan] = {}
            for msk_name in ['geo', 'photo', 'final', 'no_mask']:
                print (f"[===>*******<===] scan={scan}, mask={msk_name}")
                mean_errors = get_avg_err(errors_mask_per_scan[msk_name][scan], bad_x_thred, is_verbose=False)
                avg_err_mask_per_scan[scan][msk_name] = mean_errors
            
            
    
    else:
        raise NotImplementedError
    
    # get averaged values;
    errors = np.array(errors)
    
    
    nan_num = 0
    for j in range(errors.shape[0]):
        if any(np.isnan(errors[j])):
            nan_num += 1
            print ("nan found at idx {} /{}, = {}".format(j, errors.shape[0], errors[j]))

        
    no_ratio_info = "We consider MIN_MAX_D={:0.3f}/{:0.3f} meters | No scaling ratio | valid frames: {:d}/{:d} | ratio=1.0 ".format(
        MIN_DEPTH, MAX_DEPTH, len(errors)-nan_num, len(errors))
    print (no_ratio_info)

    #mean_errors = np.nanmean(errors[:,:9+len(bad_x_thred)], axis=0) # skip: the img_id at dim=12;
    mean_errors = np.nanmean(errors, axis=0) # skip: the img_id at dim=12;
    print_metrics(mean_errors, bad_x_thred)
    
    for msk_name in ['geo', 'photo', 'final', 'no_mask']:
        if len(errors_mask[msk_name]) > 0:
            print (f"*****************\nProcessing the depth evaluation using {msk_name} mask ...")
            # get averaged values;
            errors = np.array(errors_mask[msk_name])
            nan_num = 0
            for j in range(errors.shape[0]):
                if any(np.isnan(errors[j])):
                    nan_num += 1
                    print ("nan found at idx {} /{}, = {}".format(j, errors.shape[0], errors[j]))

                
            no_ratio_info = "We consider MIN_MAX_D={:0.3f}/{:0.3f} meters | No scaling ratio | valid frames: {:d}/{:d} | ratio=1.0 ".format(
                MIN_DEPTH, MAX_DEPTH, len(errors)-nan_num, len(errors))
            print (no_ratio_info)

            mean_errors = np.nanmean(errors, axis=0) # skip: the img_id at dim=12;
            print_metrics(mean_errors, bad_x_thred)
    
    # save avg error per scan
    json_file = os.path.join(args.outdir, 'depth-pho-geo-masks-per-scan-eval.json')
    with open(json_file, 'w') as f:
        json.dump(avg_err_mask_per_scan, f, indent=2)
        print (f"Just saved depth metric to {json_file}")
    for msk_name in ['geo', 'photo', 'final', 'no_mask']:
        avg_epe = .0
        for idx, scan_id in enumerate(dtu_test_scan_idxs):
            scan = f"scan{scan_id}"
            avg_epe += avg_err_mask_per_scan[scan][msk_name]['abs']
        avg_epe /= len(dtu_test_scan_idxs)
        print (f"Averaged on all scans, mask={msk_name}, epe={avg_epe}")

def plot_depth_eval_fig(
    root_dir, 
    err_name = 'abs',
    json_files_dict = None
):
    def parse_json_file(depth_eval_json_file):
        with open(depth_eval_json_file, 'r') as f:
            res_dict = json.load(f)
        return res_dict
    
    # set width of bar
    barWidth = 0.25
    #Plot graph with 2 y axes
    #fig = plt.subplots(figsize =(16, 8))
    fig, ax1 = plt.subplots(figsize =(16, 8))
    ax1.set_ylim([0, 50])
    clr_lable = ['b', 'g', 'r']
    msk_markers = ['d', 'o', 'x', '*']
    ax2 = ax1.twinx()
    #ax1.set_ylim([0, 70])
    #ax2.set_ylim([0, 70])
    ax1.set_ylim([0, 30])
    ax2.set_ylim([0, 30])
    for idx, (method, filename) in enumerate(json_files_dict.items()):
        print ("method = ", method)

        avg_epe = parse_json_file(os.path.join(root_dir, filename))
        #print (avg_epe)
        #for j, msk_name in enumerate(['geo', 'photo', 'final', 'no_mask']):
        for j, msk_name in enumerate(['final', 'no_mask']):
            epes = []
            scans = []
            #print (avg_epe.keys())
            err_sum = .0
            tmp_num = 0
            
            for scan in avg_epe.keys():
                err = avg_epe[scan][msk_name][err_name]
                err_sum += err
                tmp_num += 1
                epes.append(err)
                scans.append('s'+scan[len('scan'):])
            # average among all scans
            avg_err = err_sum / tmp_num
            epes.append(avg_err)
            scans.append('avg')
            print (f"{method}, {msk_name}: {err_name}={avg_err}")
            
            
            br = [x + idx*barWidth for x in np.arange(len(scans))]
            #ax1.bar(br, epes, color = clr_lable[idx], width = barWidth, edgecolor ='grey', label = method)
            ## no fill color
            if msk_name == 'no_mask':
                ax1.bar(br, epes, edgecolor = clr_lable[idx], width = barWidth, fill = False, label = method + '(no-mask)')
                ax1.legend(loc='upper center')
            else:
                #Set up ax2 to be the second y axis with x shared
                #Plot a line
                ax2.scatter(br, epes, color = clr_lable[idx], marker=msk_markers[j], label= f"{msk_name}({method})")
                ax2.legend(loc='upper left')

        
        
    # Adding Xticks
    plt.xlabel('scans', fontweight ='bold', fontsize = 15)
    plt.ylabel(f'err-{err_name}', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(scans))], scans, rotation=45)
    
    #plt.show()
    tmp_dir = Path(root_dir)/"fig-plot"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    plt.savefig(tmp_dir/'dtu_depth_eval.png', bbox_inches='tight', dpi=300)

def plot_dtu_ply_eval_fig(
    root_dir, 
    json_files_dict = None
    ):

    def parse_json_file(mvs_ply_eval_json_file):
        with open(mvs_ply_eval_json_file, 'r') as f:
            res_dict = json.load(f)
        metrics = res_dict['metrics']
        acc, comp, overall = [],[],[]
        scans = []
        for s,v in metrics.items():
            scans.append(s)
            acc.append(v[0])
            comp.append(v[1])
            overall.append(v[2])
        #return metrics
        return scans, acc, comp, overall
    
    # set width of bar
    barWidth = 0.25
    #Plot graph with 2 y axes
    #fig = plt.subplots(figsize =(16, 8))
    fig, ax1 = plt.subplots(figsize =(16, 8))
    ax1.set_ylim([0.15, 0.8])
    clr_lable = ['b', 'g', 'r']
    err_markers = ['d', 'o', 'x', '*']
    ax2 = ax1.twinx()
    ax2.set_ylim([0.15, 0.8])
    for idx, (method, filename) in enumerate(json_files_dict.items()):
        print ("method = ", method)

        scans, accs, comps, overalls = parse_json_file(os.path.join(root_dir, filename))
        results = {'acc': accs, 'comps': comps, 'overall': overalls}

        #print (avg_epe)
        #for j, msk_name in enumerate(['geo', 'photo', 'final', 'no_mask']):
        for j, err_name in enumerate(['acc', 'comps', 'overall']):
            errs = []
            x_names = []
            for scan_idx, scan in enumerate(scans):
                tmp_err = results[err_name][scan_idx]
                errs.append(tmp_err)
                x_names.append('s'+scan[len('scan'):])
            # average among all scans
            avg_err = np.array(errs).mean()
            errs.append(avg_err)
            x_names.append('avg')
            print (f"{method}: {err_name}={avg_err}")
            
            
            br = [x + idx*barWidth for x in np.arange(len(x_names))]
            #ax1.bar(br, epes, color = clr_lable[idx], width = barWidth, edgecolor ='grey', label = method)
            ## no fill color
            if err_name == 'overall':
                ax1.bar(br, errs, edgecolor = clr_lable[idx], width = barWidth, fill = False, label = method + '(overall)')
                ax1.legend(loc='upper center')
            else:
                #Set up ax2 to be the second y axis with x shared
                #Plot a line
                ax2.scatter(br, errs, color = clr_lable[idx], marker=err_markers[j], label= f"{err_name}({method})")
                ax2.legend(loc='upper left')

        
        
    # Adding Xticks
    plt.xlabel('scans', fontweight ='bold', fontsize = 15)
    plt.ylabel(f'ply-eval', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(scans))], scans, rotation=45)
    
    #plt.show()
    tmp_dir = Path(root_dir)/"fig-plot"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    plt.savefig(tmp_dir/'dtu_ply_eval.png', bbox_inches='tight', dpi=300)
    return 1

def get_ply(scan, 
        geo_pixel_thres, # e.g., = 1;
        geo_depth_thres, # e.g., = 0.01, threshold for relative_depth_diff
        prob_threshold, # e.g., = 0.3;
        num_view_consistent,
        plyfilename,
        args,
        ):

    #method = args.method
    #print (f"Will generate point cloud for method={method}, on dataset={args.dataset}")
    

    if args.dataset=="dtu":
        #for scan_id in dtu_test_scan_idxs:
        #    scan = f"scan{scan_id}"
        #plyfilename = os.path.join(args.outdir, f'{scan_id:03d}.ply')
        
        
        filter_depth(
                dataset = args.dataset, 
                scan = scan,
                scan_folder = os.path.join(args.data_path, scan), 
                out_folder = os.path.join(args.outdir, scan), 
                plyfilename = plyfilename, 
                img_wh = img_wh, 
                geo_pixel_thres = geo_pixel_thres, # e.g., = 1;
                geo_depth_thres = geo_depth_thres, # e.g., = 0.01, threshold for relative_depth_diff
                #photo_thres = photo_thres, # e.g., = 0.3;
                prob_threshold = prob_threshold, # photometric loss;
                #num_view_consistent = 4 # at least 4 source views matched;
                num_view_consistent = num_view_consistent # at least 4 source views matched;
                ) 
    
    elif args.dataset=="tanks":
        # intermediate dataset
        #scans = tanks_scans[args.split]
        geo_mask_thres = tanks_geo_mask_thres[args.split]

        #for scan in scans:
        scan_folder = os.path.join(args.data_path, args.split, scan)
        out_folder = os.path.join(args.outdir, scan)

        plyfilename = os.path.join(args.outdir, f'{scan}.ply')
        raise NotImplementedError

    elif args.dataset in ["eth3d", "eth3d_yao_eval"]:
        #scans = eth3d_scans[f"{args.eth3d_reso}/{args.split}"]
        geo_mask_thres = eth3d_geo_mask_thres[f"{args.eth3d_reso}/{args.split}"]
        #for scan in scans:
        start_time = time.time()
        scan_folder = os.path.join(args.data_path, scan)
        out_folder = os.path.join(args.outdir, scan)
        plyfilename = os.path.join(args.outdir, f'{scan}.ply')
        raise NotImplementedError

        print('[@eth3d-{}] scan:  {}, time = {:3f}'.format( 
            args.split, scan, time.time() - start_time))

    else:
        raise NotImplementedError
    return scan

def eval_ply(scan_ids, args):
    method = args.method
    res_all = []
    print (f" ==> eval point cloud for method={method}, on dataset={args.dataset}")
    res_dict = {
        'method': args.method,
        'input_ply_dir': args.ply_data_path,
        'timeStamp': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
        "metric_info": f"list of accuracy / completeness / F-1 score, with each tolerance",
        "metrics": {}
        }
    scan_subdir = {
        "low-res": 'rig_scan_eval',
        'high-res': 'dslr_scan_eval',
        }[args.eth3d_reso]
    if args.dataset in ["eth3d", "eth3d_yao_eval" ]:
        # c++ eval lib
        import src.cpp.mvs_ply_eval.lib.libeth3d_mvs_eval as eth3d_mvs_eval
        #tolerances = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]).astype(np.float32) # meters
        #tolerances = np.array([0.01, 0.02, 0.05]).astype(np.float32) # meters
        tolerances = np.array([0.02, 0.05]).astype(np.float32) # meters
        tor_num = len(tolerances)
        res_dict['dataset'] = f"{args.dataset}/{args.eth3d_reso}/{args.split}"
        res_dict["eval_tolerance (cm)"] = [f"{100*tol:.1f}" for tol in tolerances]
        print (f"processing {args.dataset}, {args.eth3d_reso}/{args.split}")
         
        for scan in eth3d_scans[f"{args.eth3d_reso}/{args.split}"]:
            plyfilename = os.path.join(args.outdir, f'{scan}.ply')
            gt_mlp_path = os.path.join(args.ply_data_path, f"{scan}/{scan_subdir}/scan_alignment.mlp")
            #1) accuracy results: idx = 0:toler_nums
            #2) completeness results: idx = toler_nums : 2*toler_nums:
            #3) F-1 score results: idx = 2*toler_nums : 3*toler_nums 
            # accuracy results (toler_nums) + completeness (toler_nums) + F-1 score (toler_nums); 
            res = eth3d_mvs_eval.evaluate_eth3d_mvs(
                plyfilename,
                gt_mlp_path,
                tolerances,
                ## default args
                0.01, # voxel_size 
                0.5 * 0.00225,#beam_start_radius_meters
                0.011,#beam_divergence_halfangle_deg
                "",#completeness_cloud_output_path = ""
                "" #accuracy_cloud_output_path = ""
                )
            print ("acc./comp./F1", res)
            res_all.append(res)
            res_dict["metrics"][scan] = [
                tuple( (100*res[i], 100*res[tor_num+i], 100*res[2*tor_num+i]) ) for i in range(tor_num)
                ]
            #sys.exit()
        
        json_file = os.path.join(args.outdir, 'mvs-ply-eval.json')
        with open(json_file, 'w') as f:
            json.dump(res_dict, f, indent=2)
            print (f"Just saved mvs eval metric to {json_file}")

        res_avg = np.stack(res_all, axis=0).mean(axis=0)
        print ("[***] Averaged results:")
        for i in range(tor_num):
            tol_cm = tolerances[i] * 100 # meter to cm;
            acc = res_avg[i] * 100
            comp = res_avg[tor_num+i] * 100
            f1 = res_avg[2*tor_num+i] * 100
            print (f"\t@tolerance={tol_cm:.1f}cm : acc./comp./F1 = {acc:.2f} \& {comp:.2f} \& {f1:.2f}")
        
        csv_file = os.path.join(args.outdir, "mvs-ply-eval.csv")
        save_metrics_to_csv_file(res_avg, csv_file, \
                model_name = args.method, 
                dir_you_specifiy = args.outdir, 
                tolerances = tolerances)
        
        dst_csv_file = os.path.join(args.csv_dir, f'mvs-ply-eval-{args.machine_name}.csv')
        os.system(f'cat {csv_file} >> {dst_csv_file}')
        print (f"cat {csv_file} to {dst_csv_file}")

        
    elif args.dataset=="tanks":
        print (f"processing {args.dataset}")

    elif args.dataset=="dtu":
        print (f"processing {args.dataset}")
        my_dtu_eval = DTU_ply_eval(args, ply_eval_name=f'prob_thre_{args.prob_threshold:.2f}')
        #my_dtu_eval(scans_to_do = dtu_test_scan_idxs[0:1])
        my_dtu_eval(scans_to_do = scan_ids)

    else:
        raise NotImplementedError
    


def save_metrics_to_csv_file(res_avg, csv_file, model_name, dir_you_specifiy, tolerances):
    """ save as csv file, Excel file format """
    timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    tmp_dir = dir_you_specifiy
    tor_num = len(tolerances) # tolerances in meter
    messg = timeStamp + ",method={},resultDir,{}".format(model_name, tmp_dir)
    for i in range(tor_num):
        tol_cm = tolerances[i] * 100 # meter to cm;
        acc = res_avg[i] * 100
        comp = res_avg[tor_num+i] * 100
        f1 = res_avg[2*tor_num+i] * 100
        messg += f",tolerance={tol_cm:.1f}cm,acc,{acc:.4f},comp,{comp:.4f},f1,{f1:.4f}"
    with open( csv_file, 'w') as fwrite:
        fwrite.write(messg + "\n")
    print ("Done! Write ", csv_file, "\n")


def organize_est_depth(dataset_type, src_depth_dir, 
                    idx_2_name_map_txt, 
                    is_confidence_map = False,
                    scans = None
                    ):
    map_name_dict = {}
    print (f"calling func organize_est_depth() for {dataset_type}")
    if dataset_type in ["dtu", "eth3d", "eth3d_yao_eval"]:
        # map names
        # e.g. d_00000108.pfm : */dtu_patchmatchnet_test/scan1/images/00000000.jpg
        with open(idx_2_name_map_txt) as f:
            names = f.readlines()
            names = [line.rstrip() for line in names]
        
        for idx, nm in enumerate(names):
            src_dep_nm = os.path.join(src_depth_dir, f"d_{idx:08d}.pfm")
            img_name = nm.split('/')[-1]
            scan = nm.split('/')[-3]
            if scan not in scans:
                #print ("skipped scan = ", scan)
                continue
            if dataset_type == "eth3d_yao_eval":
                # e.g., */eth3d_mvsnet_yao/playground/images/images_rig_cam4_undistorted/1477833684658155598.png
                img_name = nm.split('/')[-2] + '-' + nm.split('/')[-1]
                scan = nm.split('/')[-4]
            #if scans is not None:
            #    assert scan in scans, f"Wrong scan={scan} decoded from txt"
            dst_dir = os.path.join(src_depth_dir, f"{scan}/depth_est")
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            dst_dep_nm = os.path.join(dst_dir, img_name[:-len(".jpg")] + ".pfm")
            to_do = f"mv {src_dep_nm} {dst_dep_nm}"
            #print (to_do)
            os.system(to_do)
            #sys.exit()
        
        if is_confidence_map:
            for idx, nm in enumerate(names):
                src_conf_nm = os.path.join(src_depth_dir, f"conf_{idx:08d}.pfm")
                img_name = nm.split('/')[-1]
                scan = nm.split('/')[-3]
                if scan not in scans:
                    #print ("skipped scan = ", scan)
                    continue
                if dataset_type == "eth3d_yao_eval":
                    img_name = nm.split('/')[-2] + '-' + nm.split('/')[-1]
                    scan = nm.split('/')[-4]
                #if scans is not None:
                #    assert scan in scans, f"Wrong scan={scan} decoded from txt"
                
                dst_dir = os.path.join(src_depth_dir, f"{scan}/confidence")
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                dst_conf_nm = os.path.join(dst_dir, img_name[:-len(".jpg")] + ".pfm")
                to_do = f"mv {src_conf_nm} {dst_conf_nm}"
                #print (to_do)
                os.system(to_do)
                #sys.exit()


    elif dataset_type == "tanks":
        raise NotImplementedError
    else:
        raise NotImplementedError


# see: https://github.com/kysucix/fusibile
# and the modified version by MVSNet: https://github.com/YoYo000/fusibile
class filter_depth_gipuma(object):
    def __init__(self, kwargs):
         
        #self.dense_folder = args.dense_folder
        self.fusibile_exe_path = kwargs.get(
            'fusibile_exe_path', 
            'src/cpp/fusibile-yao/bin/fusibile'
            )
        self.prob_threshold = kwargs.get('prob_threshold', 0.8)
        self.disp_threshold = kwargs.get('disp_threshold', 0.25)
        self.num_consistent = kwargs.get('num_consistent', 3)
        self.num_depth = kwargs.get('num_depth', 64) # 64 depth bins;
        img_extension = kwargs.get('img_extension', '.jpg')
        assert img_extension in [".jpg", ".png"], f"Wrong image extension {img_extension} found!"
        self.img_extension = img_extension
        
        # fusibile filter, hyper-parameters you can adjust
        self.fusibile_depth_min = 0.001
        self.fusibile_depth_max = 100000
        self.fusibile_normal_thresh = 360
    
    def probability_filter(self, scan_folder, dense_folder):
        image_folder = Path(scan_folder)/'images'
        depth_folder = Path(dense_folder)/'depth_est'
        prob_folder = Path(dense_folder)/'confidence'
        
        # convert cameras 
        image_names = os.listdir(image_folder)
        for image_name in image_names:
            image_prefix = os.path.splitext(image_name)[0]
            init_depth_map_path = depth_folder/image_prefix+'.pfm'
            prob_map_path = prob_folder / image_prefix+'.pfm'
            out_depth_map_path = depth_folder / image_prefix+'_prob_filtered.pfm'


            depth_map = pfm.readPFM(init_depth_map_path)
            prob_map = pfm.readPFM(prob_map_path)
            depth_map[prob_map < self.prob_threshold] = 0
            pfm.save(out_depth_map_path, depth_map)
    

    def load_cam(self, file, interval_scale=1):
        """ read camera txt file """
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        # not used, line 11: DEPTH_MIN DEPTH_INTERVAL (DEPTH_NUM DEPTH_MAX)
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

        return cam 
    

    def read_gipuma_dmb(self, path):
        '''read Gipuma .dmb format image'''

        with open(path, "rb") as fid:
            
            image_type = unpack('<i', fid.read(4))[0]
            height = unpack('<i', fid.read(4))[0]
            width = unpack('<i', fid.read(4))[0]
            channel = unpack('<i', fid.read(4))[0]
            
            array = np.fromfile(fid, np.float32)
        array = array.reshape((width, height, channel), order="F")
        return np.transpose(array, (1, 0, 2)).squeeze()

    
    def write_gipuma_dmb(self, path, image):
        '''write Gipuma .dmb format image'''
        
        image_shape = np.shape(image)
        width = image_shape[1]
        height = image_shape[0]
        if len(image_shape) == 3:
            channels = image_shape[2]
        else:
            channels = 1

        if len(image_shape) == 3:
            image = np.transpose(image, (2, 0, 1)).squeeze()

        with open(path, "wb") as fid:
            # fid.write(pack(1))
            fid.write(pack('<i', 1))
            fid.write(pack('<i', height))
            fid.write(pack('<i', width))
            fid.write(pack('<i', channels))
            image.tofile(fid)
        return 

    
    def mvsnet_to_gipuma_dmb(self, in_path, out_path):
        '''convert mvsnet .pfm output to Gipuma .dmb format'''
        
        image = pfm.readPFM(in_path)
        self.write_gipuma_dmb(out_path, image)
        return
    
    
    # change K and E, to projection matric K@E;
    def mvsnet_to_gipuma_cam(self, in_path, out_path):
        '''convert mvsnet camera to gipuma camera format'''

        cam = self.load_cam(open(in_path))
        #print ("loading cam from ", in_path, cam)

        #extrinsic = cam[0:4][0:4][0]
        #intrinsic = cam[0:4][0:4][1]
        extrinsic = cam[0]
        intrinsic = cam[1]
        #print (extrinsic, "\n", intrinsic)
        intrinsic[3][0] = 0
        intrinsic[3][1] = 0
        intrinsic[3][2] = 0
        intrinsic[3][3] = 0
        projection_matrix = np.matmul(intrinsic, extrinsic)
        projection_matrix = projection_matrix[0:3][:]
        
        #sys.exit()
        f = open(out_path, "w")
        for i in range(0, 3):
            for j in range(0, 4):
                f.write(str(projection_matrix[i][j]) + ' ')
            f.write('\n')
        f.write('\n')
        f.close()

        return

    
    def fake_gipuma_normal(self, in_depth_path, out_normal_path):
        depth_image = self.read_gipuma_dmb(in_depth_path)
        image_shape = np.shape(depth_image)

        normal_image = np.ones_like(depth_image)
        normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1))
        normal_image = np.tile(normal_image, [1, 1, 3])
        normal_image = normal_image / 1.732050808

        mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
        mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
        mask_image = np.tile(mask_image, [1, 1, 3])
        mask_image = np.float32(mask_image)

        normal_image = np.multiply(normal_image, mask_image)
        normal_image = np.float32(normal_image)

        self.write_gipuma_dmb(out_normal_path, normal_image)
        return 
    
    
    def mvsnet_to_gipuma(self, scan_folder, out_folder, gipuma_point_folder):

        # the pair file
        cam_folder = os.path.join(scan_folder, 'cams_1')
        
        image_folder = os.path.join(scan_folder, 'images')
        depth_folder = os.path.join(out_folder, 'depth_est')

        gipuma_cam_folder = os.path.join(gipuma_point_folder, 'cams')
        gipuma_image_folder = os.path.join(gipuma_point_folder, 'images')
        if not os.path.isdir(gipuma_point_folder):
            os.mkdir(gipuma_point_folder)
        if not os.path.isdir(gipuma_cam_folder):
            os.mkdir(gipuma_cam_folder)
        if not os.path.isdir(gipuma_image_folder):
            os.mkdir(gipuma_image_folder)

        # convert cameras 
        image_names = os.listdir(image_folder)
        image_names.sort()
        for image_name in image_names:
            image_prefix = os.path.splitext(image_name)[0]
            in_cam_file = os.path.join(cam_folder, image_prefix + '_cam.txt')
            #out_cam_file = os.path.join(gipuma_cam_folder, image_name+'.P')
            out_cam_file = os.path.join(gipuma_cam_folder, image_prefix+'.P')
            self.mvsnet_to_gipuma_cam(in_cam_file, out_cam_file)

        # copy images to gipuma image folder    
        for image_name in image_names:
            in_image_file = os.path.join(image_folder, image_name)
            out_image_file = os.path.join(gipuma_image_folder, image_name)
            shutil.copy(in_image_file, out_image_file)    

        # convert depth maps and fake normal maps
        gipuma_prefix = '2333__'
        for image_name in image_names:
            image_prefix = os.path.splitext(image_name)[0]
            sub_depth_folder = os.path.join(gipuma_point_folder, gipuma_prefix+image_prefix)
            if not os.path.isdir(sub_depth_folder):
                os.mkdir(sub_depth_folder)
            
            in_depth_pfm = os.path.join(depth_folder, image_prefix+'_prob_filtered.pfm')
            out_depth_dmb = os.path.join(sub_depth_folder, 'disp.dmb')
            
            fake_normal_dmb = os.path.join(sub_depth_folder, 'normals.dmb')
            self.mvsnet_to_gipuma_dmb(in_depth_pfm, out_depth_dmb)
            self.fake_gipuma_normal(out_depth_dmb, fake_normal_dmb)

     
    def depth_map_fusion(self, point_folder):

        cam_folder = os.path.join(point_folder, 'cams')
        image_folder = os.path.join(point_folder, 'images')

        cmd = self.fusibile_exe_path
        cmd = cmd + ' -img_ext ' + self.img_extension
        cmd = cmd + ' -input_folder ' + point_folder + '/'
        cmd = cmd + ' -p_folder ' + cam_folder + '/'
        cmd = cmd + ' -images_folder ' + image_folder + '/'
        cmd = cmd + ' --depth_min=' + str(self.fusibile_depth_min)
        cmd = cmd + ' --depth_max=' + str(self.fusibile_depth_max)
        cmd = cmd + ' --normal_thresh=' + str(self.fusibile_normal_thresh)
        cmd = cmd + ' --disp_thresh=' + str(self.disp_threshold)
        cmd = cmd + ' --num_consistent=' + str(self.num_consistent)
        print (cmd)
        os.system(cmd)

        return
    
    def __call__(self, scan_folder, dense_folder): 
        point_folder = os.path.join(dense_folder, 'points')
        if not os.path.isdir(point_folder):
            os.mkdir(point_folder)
    
        # probability filter
        print ('[Gipuma] filter depth map with probability map')
        self.probability_filter(scan_folder, dense_folder)

        # convert to gipuma format
        print ('[Gipuma] Convert mvsnet output to gipuma input')
        self.mvsnet_to_gipuma(
            scan_folder = scan_folder, 
            out_folder = dense_folder, 
            gipuma_point_folder = point_folder
        )

        # depth map fusion with gipuma 
        print ('[Gipuma] Run depth map fusion & filter')
        self.depth_map_fusion(point_folder)
                
     
def organize_gipuma_ply(out_dir, scans):
    dst_dir = os.path.join(out_dir, 'ply_gipuma')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for s in scans:
        scan_idx = s[len('scan'):]
        src_ply_dir = (Path(out_dir)/f'{s}/points/').dirs("consistencyCheck-*")
        assert len(src_ply_dir) == 1
        src_ply_dir = src_ply_dir[0]
        src_ply = src_ply_dir/"final3d_model.ply"
        #print (src_ply)
        dst_ply = os.path.join(dst_dir, f'{scan_idx:>03}.ply')
        to_do = f"mv {src_ply} {dst_ply}"
        #print (to_do)
        os.system(to_do)
        #sys.exit()

def save_to_mp4_video(
            video_file_name,
            plot_fig_name,
            method_names,
            per_scan_2_video_dict,
            is_no_color = False,
            maxval=-1, 
            is_inverse_depth=False,
            ):

    #fourcc = cv2.VideoWriter_fourcc(*'MP4V') # OpenCV: FFMPEG: tag 0x5634504d/'MP4V' could not be supported;
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Then fallback to use tag 0x7634706d/'mp4v'
    
    assert "ours" == method_names[0], "Our method is the first"
    method_num = len(method_names)
    assert method_num == 2
    h, w = cv2.imread(per_scan_2_video_dict['img_name'][0]).shape[0:2]
    print (f"img size = {w} x {h}")
    frame_nums  = len(per_scan_2_video_dict['img_name'])

    # in the format:
    # --------------
    #   img    | dep_method_1 | dep_method_2 | ...
    # ----------------------------------------------
    # gt-depth | dep_err_1    | depth_err_2  | ...
    board_wid =  board_hei = 8
    board1 = np.zeros((2*h+board_hei, board_wid, 3)).astype(np.uint8)
    board2 = np.zeros((board_hei, w, 3)).astype(np.uint8)
    
    hh = 2*h + board_hei  
    ww = w + (w + board_wid)* method_num
    print (f"final video dim = {ww} x {hh}")
    
    if is_inverse_depth:
        video_file_name += '_disp'
    if is_no_color:
        video_file_name += '_noclr'
    
    def put_txt(image, txt, x, y):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # org
        org = (x, y)
        # fontScale
        fontScale = 1
        # color in BGR
        color = (255, 255, 255)
        
        # Line thickness of 2 px
        thickness = 2
        
        # Using cv2.putText() method
        image = cv2.putText(image, txt, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        return image
    
    #fps = 20
    #fps = 30
    fps = 10
    abs_err_metric = { i : [] for i in method_names}
    #abs_rel_err_metric = { i : [] for i in method_names}

    out = cv2.VideoWriter(video_file_name + '_fps{}'.format(fps) + '.mp4', fourcc,
                fps,# fps
                (ww, hh) # frameSize
        )
    for idx in range(frame_nums):
        print (f"process frame {idx + 1} / {frame_nums}")
        img = cv2.imread(per_scan_2_video_dict['img_name'][idx])
        #img = put_txt(img, txt=f'frm{idx:02d}', x=w-20, y=10)
        depth_gt = pfm.readPFM(per_scan_2_video_dict['depth_gt_name'][idx])
        depth_gt_raw = depth_gt
        h, w = depth_gt.shape

        if is_inverse_depth:
            depth_gt = 1.0 / depth_gt
        if not is_no_color:
            depth_gt = manydepth_colormap(depth_gt, maxval, is_return_np = True)
        else:
            depth_gt = np.repeat(depth_gt, axis=-1)
        
        #print (img.shape, board2.shape, depth_gt.shape, type(depth_gt))
        depth_gt = put_txt(depth_gt, txt=f'Frame {idx:02d}', x=w-200, y=50)
        collage = np.concatenate((img[:,:,::-1], board2, depth_gt), 0)
        #print ("collage_1: ", collage.shape)

        for method in method_names:
            dep_pred = pfm.readPFM(per_scan_2_video_dict[method]['depth_est_name'][idx])
            dep_pred = cv2.resize(dep_pred, (w, h), interpolation=cv2.INTER_NEAREST)
            valid_mask = (depth_gt_raw > 0)
            dep_err = np.abs(dep_pred - depth_gt_raw)* valid_mask.astype(np.float32)
            l1_err = dep_err[valid_mask].mean()
            abs_err_metric[method].append(l1_err)
            print (f"{method}, L1-err = {l1_err}" )
            if not is_no_color:
                dep_pred = manydepth_colormap(dep_pred, maxval, is_return_np = True)
                dep_err = error_colormap(dep_err, maxval=20, is_return_np=True)
                    
            else:
                dep_pred = np.repeat(dep_pred, axis=-1)
                dep_err = np.repeat(dep_err, axis=-1)
            # write error
            dep_err = put_txt(dep_err, txt=f'L1={l1_err:.2f}', x=w-220, y=50)
            tmp_coll = np.concatenate((dep_pred, board2, dep_err), 0)
            collage = np.concatenate((collage, board1, tmp_coll), 1)
        #print ("to video ", collage.shape)
        out.write(collage[:,:,::-1].astype(np.uint8)) # for opencv GBR order;
        del collage
    
    out.release()
    print ("[***] saved video {}".format(video_file_name + ".mp4"))
    
    f, ax = plt.subplots(1)
    for method in method_names:
        print ("method = ", method)
        ax.plot(range(frame_nums), abs_err_metric[method], label=method)
    # set figures
    ax.legend()
    ax.set_title('l1-err (mm)')
    
    #plt.show()
    #tmp_dir = Path("results/tmp")
    #if not os.path.exists(tmp_dir):
    #    os.mkdir(tmp_dir)
    plt.savefig(plot_fig_name, bbox_inches='tight', dpi=100)



if __name__ == '__main__':

    if args.config_file:
        cfg = load_config(args.config_file)
        print (f"Just loaded cfg file from {args.config_file}")
    else:
        cfg = None
    
    if args.dataset=="dtu":
        scans = [f"scan{scan_id}" for scan_id in dtu_test_scan_idxs]
        scan_ids = dtu_test_scan_idxs
        if isinstance (args.scans, list):
            scans = [f"scan{scan_id}" for scan_id in args.scans]
            scan_ids = args.scans
    elif args.dataset=="tanks":
        scans = tanks_scans[args.split]
    elif args.dataset in ["eth3d", "eth3d_yao_eval"]:
        scans = eth3d_scans[f"{args.eth3d_reso}/{args.split}"]
    else:
        raise NotImplementedError
    
    if args.task == 'save_video':
        #scan = 'scan9'
        scan = 'scan1'
        tmp_dir = "results/tmp"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        video_file_name = os.path.join(tmp_dir, scan)
        plot_fig_name = os.path.join(tmp_dir, scan + "-fig-l1-err.png")
        method_names = ['ours', 'itermvs']

        exp_dirs = {
            'ours': "results/exp77D2-raftpsconf1a4G3Itr24pairspf-Dil1inv-dtu-Z0.42to0.935-D64-epo20-bs1-h1024xw1280-rtxa6000s3/depth-epo-003",
            'itermvs': 'results/itermvs/dtu'
        }

        per_scan_2_video_dict = {}
        is_no_color = False
        maxval = -1
        is_inverse_depth = False
        scan_folder = os.path.join(args.data_path, scan)
        gt_depth_folder = os.path.join(args.data_path, "Depths_raw/"+scan)
        per_scan_2_video_dict['img_name'] = []
        per_scan_2_video_dict['depth_gt_name'] = []
        for method in method_names:
            per_scan_2_video_dict[method] = {
                'depth_est_name': [],
                }
        frame_num = 49 
        #frame_num = 2 
        for view_idx in range(frame_num):
            gt_dep_nm = os.path.join(gt_depth_folder, 'depth_map_{:0>4}.pfm'.format(view_idx))
            img_nm = os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(view_idx))
            per_scan_2_video_dict['img_name'].append(img_nm)
            per_scan_2_video_dict['depth_gt_name'].append(gt_dep_nm)

            for method in method_names:
                per_scan_2_video_dict
                dep_nm = os.path.join(exp_dirs[method], scan, f'depth_est/{view_idx:0>8}.pfm')
                per_scan_2_video_dict[method]['depth_est_name'].append(dep_nm)
        
        save_to_mp4_video(
            video_file_name,
            plot_fig_name,
            method_names,
            per_scan_2_video_dict,
            is_no_color,
            maxval, 
            is_inverse_depth
            ) 
        sys.exit()
       
    if args.task == 'organize_gipuma_ply': 
        organize_gipuma_ply(
            out_dir = args.outdir,
            scans = scans)
        sys.exit(0)
    
    if args.task == 'organize_dir': 
        organize_est_depth(
            dataset_type = args.dataset, 
            src_depth_dir = args.outdir, 
            idx_2_name_map_txt = args.idx_2_name_map_txt,
            is_confidence_map = True,
            scans = scans
            )
        sys.exit()
    if args.task == 'depth_eval':
        eval_depth(args)
        sys.exit()
    
    if args.task == 'plot_depth_eval_fig':
        json_files_dict = {
            'iterMVS': "itermvs-official-bl_itermvs-Dil1inv-dtu-Z0.42to0.935-D32-epo20-bs4-h1152xw1600-rtxa6ks3/depth-epo-000/depth-pho-geo-masks-per-scan-eval.json",
            #'iterMVS-our-h640xw800': "exp66C-bl_itermvs-Dil1inv-dtu-Z0.42to0.935-D32-epo10-bs8-h640xw800-rtxa6ks3/depth-epo-001/depth-pho-geo-masks-per-scan-eval.json",
            #'exp72D2': 'exp77D2-raftpsconf1a4G3Itr24pairspf-Dil1inv-dtu-Z0.42to0.935-D64-epo20-bs1-h1024xw1280-rtxa6000s3/depth-epo-003/depth-pho-geo-masks-per-scan-eval.json',
            'exp77D6': "exp77D6-raftpsconf1a4G3Itr24pairspf-Dil1inv-dtu-Z0.42to0.935-D64-epo20-bs1-h1024xw1280-rtxa6ks8/depth-epo-019/depth-pho-geo-masks-per-scan-eval.json"
        }
        plot_depth_eval_fig(
            root_dir = '/nfs/STG/SemanticDenseMapping/changjiang/proj-riav-mvs/results/',
            err_name = 'abs',
            #err_name = 'rmse',
            #err_name = 'rmse_log',
            json_files_dict = json_files_dict
        )
        sys.exit()
    
    if args.task == 'plot_mvs_ply_eval_fig':
        json_files_dict = {
            'iterMVS': "itermvs-official-bl_itermvs-Dil1inv-dtu-Z0.42to0.935-D32-epo20-bs4-h1152xw1600-rtxa6ks3/depth-epo-000/ply_eval/mvs-ply-eval.json",
            'exp72D2': 'exp77D2-raftpsconf1a4G3Itr24pairspf-Dil1inv-dtu-Z0.42to0.935-D64-epo20-bs1-h1024xw1280-rtxa6000s3/depth-epo-003/ply_eval_conf/mvs-ply-eval.json',
        }
        root_dir = '/nfs/STG/SemanticDenseMapping/changjiang/proj-riav-mvs/results/'
        plot_dtu_ply_eval_fig(root_dir, json_files_dict)
        sys.exit()
    
    if args.task in ['ply_fuse', 'all']:
        print ("Generating point cloud ...")
        method = args.method
        print (f"Will generate point cloud for method={method}, on dataset={args.dataset}")
        #args.num_workers = 1
        print (f"Will process {len(scans)} scans, i.e., {scans}")
        
        # Code is adopted from GBi-Net;
        if cfg and not cfg.get("no_fusion", False):
            for para_id in range(cfg["fusion"]["xy_filter_per"]["para_num"]):
                if cfg["fusion"]["xy_filter_per"].get("para_tag", None) is not None:
                    para_tag = cfg["fusion"]["xy_filter_per"].get("para_tag")[para_id]
                else:
                    para_tag = para_id
                
                
                for scan in scans:
                    scan_id = int(scan[len('scan'):])
                    paras = cfg["fusion"]["xy_filter_per"][scan]
                    prob_threshold = paras["prob_threshold"][para_tag]
                    num_consistent = paras["num_consistent"][para_tag]
                    img_dist_thresh = paras["img_dist_thresh"][para_tag] if paras.get("img_dist_thresh", None) is not None else 1.0
                    depth_thresh = paras["depth_thresh"][para_tag] if paras.get("depth_thresh", None) is not None else 0.01
                    
                    point_dir = os.path.join(args.outdir, f'para_tag{para_tag}')
                    os.makedirs(point_dir, exist_ok=True)
                    
                    sub_dir = f'prb{prob_threshold:.2f}_' \
                              f'pix{img_dist_thresh:.2f}_' \
                              f'dep{depth_thresh:.2f}'
                    
                    os.makedirs(os.path.join(point_dir, sub_dir), exist_ok=True)
                    
                    
                    
                    plyfilename =  f'{scan_id:03d}.ply'
                            
                    print (f'New parameters for tuning: scan={scan}, ' \
                           f'geo_pixel_thres={img_dist_thresh:.3f}, ' \
                           f'geo_depth_thres={depth_thresh:.3f}, ' \
                           f'prob_thres={prob_threshold:.3f}, ' \
                           f'num_view_consistent={num_consistent}' \
                           )
                    get_ply(
                        scan = scan, 
                        geo_pixel_thres = img_dist_thresh, # e.g., = 1;
                        geo_depth_thres = depth_thresh, # e.g., = 0.01, threshold for relative_depth_diff
                        prob_threshold = prob_threshold, # e.g., = 0.3;
                        num_view_consistent = num_consistent,
                        plyfilename = os.path.join(point_dir, sub_dir, plyfilename),
                        args = args) 
                    
                    print ("Evaluating DTU point cloud ...")
                    # update ply dir and name
                    point_dir_old = args.outdir
                    args.outdir = point_dir
                    my_dtu_eval = DTU_ply_eval(args, ply_eval_name= sub_dir)
                    my_dtu_eval(scans_to_do = [scan_id])
                    # change dir back for next scan;
                    args.outdir = point_dir_old
        
        else:
            for scan in scans:
                scan_id = int(scan[len('scan'):])
                print ("scan = {}, scan_id = {}".format(scan, scan_id) )
                #plyfilename = f'{scan_id:03d}.ply'
                ply_fn_dir = os.path.join(args.outdir, f'prob_thre_{args.prob_threshold:.2f}')
                os.makedirs(ply_fn_dir, exist_ok=True)
                plyfilename = os.path.join(ply_fn_dir, f'{scan_id:03d}.ply')
                get_ply(
                    scan, 
                    geo_pixel_thres = args.geo_pixel_thres, # e.g., = 1;
                    geo_depth_thres = args.geo_depth_thres, # e.g., = 0.01, threshold for relative_depth_diff
                    prob_threshold = args.prob_threshold, # e.g., = 0.3;
                    num_view_consistent = args.num_consistent,
                    plyfilename = plyfilename,
                    args = args)
        
        
    
    if args.task in ['ply_eval', 'all']:
        print ("Evaluating point cloud ...")
        eval_ply(scan_ids, args)
