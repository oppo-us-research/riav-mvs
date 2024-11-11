"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import os
import sys
import re
from path import Path


""" load our own moduels """
from src.utils.utils import readlines, write_to_file


def build_scannet_train_val_list(
    data_dir,
    scenes_split_txt,
    samples_split_txt,
    interval = 2,
    r = 5
    ):
    dataset_index = []
    data_id = 0
    skip = r
    input_directory = Path(data_dir)
    # 'scene0018_00'
    scenes = readlines(scenes_split_txt)
    print (f"loading {len(scenes)} scenes, including: {scenes[0]}, .. {scenes[-1]}")
    lines = ['#scene_name frame_idxs (e.g., indices at time t-2, t-1, t, t+1, t+2)']
    sum_samples = 0
    for idx,scan in enumerate(scenes):
        if idx % 50 == 0:
            print(f"processing {idx+1}/{len(scenes)}", scan, "cur # samples = ", sum_samples)

        image_filenames = sorted(
            (input_directory /scan).files("*.npz")
            )
        img_names = [f.split('/')[-1] for f in image_filenames]
        sample_imgs = img_names[::interval]
        img_idx = [ int(f[:-len('.npz')]) for f in sample_imgs]
        #print ("image_filenames[0] = ", image_filenames[0:5])
        #print ("img_names = ", img_names[0:5])
        #print ("img_idx = ", img_idx[0:5], len(img_idx))
        tmp = range(r, len(img_idx) - r, skip)
        #print ("tmp = ", tmp)
        for i in range(r, len(img_idx) - r, skip):
            #print (f"i = {i}, r= {r}")
            frame_ids = img_idx[i-r : i+r+1]
            #print (frame_ids)
            line = f"{scan}" + (" {}"*len(frame_ids)).format(*frame_ids)
            #print ("line = ", line)
            #sys.exit()
            lines.append(line)
            sum_samples += 1
    
    write_to_file(samples_split_txt, lines)
        

def build_scannet_test_list(
    data_dir,
    scenes_split_txt,
    samples_split_txt,
    interval = 10,
    r = 5
    ):
    dataset_index = []
    data_id = 0
    skip = r
    input_directory = Path(data_dir)
    # 'scene0018_00'
    scenes = readlines(scenes_split_txt)
    print (f"loading {len(scenes)} scenes, including: {scenes[0]}, .. {scenes[-1]}")
    lines = ['#scene_name frame_idxs (e.g., indices at time t-2, t-1, t, t+1, t+2)']
    sum_samples = 0
    for idx,scan in enumerate(scenes):
        if idx % 50 == 0:
            print(f"processing {idx+1}/{len(scenes)}", scan, "cur # samples = ", sum_samples)

        image_filenames = sorted(
            (input_directory /scan / 'frames/color').files("*.png")
            )
        img_names = [f.split('/')[-1] for f in image_filenames]
        sample_imgs = img_names[::interval]
        img_idx = [ int(f[:-len('.png')]) for f in sample_imgs]
        #print ("image_filenames[0] = ", image_filenames[0:5])
        #print ("img_names = ", img_names[0:5])
        #print ("img_idx = ", img_idx[0:5], len(img_idx))
        tmp = range(r, len(img_idx) - r, skip)
        #print ("tmp = ", tmp)
        for i in range(r, len(img_idx) - r, skip):
            #print (f"i = {i}, r= {r}")
            frame_ids = img_idx[i-r : i+r+1]
            #print (frame_ids)
            line = f"{scan}" + (" {}"*len(frame_ids)).format(*frame_ids)
            #print ("line = ", line)
            #sys.exit()
            lines.append(line)
            sum_samples += 1
    
    write_to_file(samples_split_txt, lines)
        

""" 
How to run this file:
- cd ~/PROJ_ROOT/src/models/bl_estdepth/
- python get_scannet_train_list.py
"""
if __name__ == '__main__':
    proj_dir = '/nfs/STG/SemanticDenseMapping/changjiang/proj-raft-mvs'
    data_dir = "/nfs/flash/STG/project/SemanticDenseMapping/Scannet/other_exported_train/deepvideomvs"
    data_dir = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/other_exported_train/deepvideomvs"
    #trainlist = Path(proj_dir)/'splits/scannet_benchmark/train_small_files.txt'
    split = 'train'
    split = 'val'
    split = 'test'
    txtlist = Path(proj_dir)/f'splits/scannet_benchmark/{split}_files.txt'
    
    
    seq_len = 5
    #txt_fn = Path(proj_dir)/f"splits/scannetv2/scannet_simple10_npz/{split}_files_sml.txt"
    txt_fn = Path(proj_dir)/f"splits/scannetv2/scannet_simple10_npz/{split}_files.txt"
    if split == 'test':
        data_dir = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_test"
        txt_fn = Path(proj_dir)/f"splits/scannetv2/test/test_estdepth_simple10-nmeas4/{split}_files.txt"

        interval = 10 
        # due to frame_skip = 1 if not is_train in 
        # src/datasets/scannet_export.py;
        # so we set sampling rate = 10;
        build_scannet_test_list(
            data_dir,
            scenes_split_txt = txtlist,
            samples_split_txt = txt_fn, 
            interval = 10,
            r = seq_len // 2
            )
    else:    
        interval = 2 
        # due to frame_skip = 4 if is_train in 
        # src/datasets/scannet_export.py;
        # so actually sampling rate = 2*4=8;
        build_scannet_train_val_list(
            data_dir,
            scenes_split_txt = txtlist,
            samples_split_txt = txt_fn, 
            interval = interval,
            r = seq_len // 2
            )
    