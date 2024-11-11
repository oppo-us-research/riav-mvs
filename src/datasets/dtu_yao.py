"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from IterMVS (https://github.com/FangjinhuaWang/IterMVS)
# MIT license.
# ------------------------------------------------------------------------------------

import os
from path import Path
import numpy as np
import cv2
from PIL import Image

import torch
from torchvision import transforms

""" load our own moduels """
from src.utils import pfmutil as pfm
from .mvs_dataset import MVSDatasetBase

def half_and_crop_depth(hr_depth, target_h=512, target_w=640):
    #downsample
    h0, w0 = hr_depth.shape
    # original w,h: 1600, 1200; downsample -> 800, 600 ; crop -> 640, 512
    # depth value unit: in mm
    depth = cv2.resize(hr_depth, (w0//2, h0//2), interpolation=cv2.INTER_NEAREST)
    #crop
    h, w = depth.shape
    assert target_h < h and target_w < w, "too large target w and h"
    start_h, start_w = (h - target_h)//2, (w - target_w)//2
    depth_crop = depth[start_h: start_h + target_h, start_w: start_w + target_w]
    return depth_crop

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(MVSDatasetBase):
    def __init__(self, *args, **kwargs):
        super(MVSDataset, self).__init__(*args, **kwargs)

        # specific to DTU-train dataset;
        self.num_scales = 4 # multi image scales: 1/2^i, i=0,1,2,3
        #assert self.num_scales == 4, f"Got self.num_scales={self.num_scales}"

        #assert (self.width, self.height) == (640, 512), "DTU_train requires img size (640, 512)"

        self.data_mode = kwargs.get('data_mode', 'train')
        self.interval_scale = kwargs.get('depth_interval_scale', 1.06)
        assert self.data_mode in ["train", "val"] # "for test, we need dtu_yap_eval.py";
        print ("[***] DTU_yao: self.data_mode=", self.data_mode)

        assert self.height%32==0 and self.width%32==0, \
                'img_w and img_h must both be multiples of 32!'
        self.mm_to_meter_factor = 0.001 # change mm value to m (meter);
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=self.interp)

        self.metas = self.build_metas()

    def build_metas(self):
        metas = []
        with open(self.filenames) as f:
            scans = [line.rstrip() for line in f.readlines()]

        pair_file = "Cameras_1/pair.txt"
        data_idx = 0 # global index among the metas;
        for scan in scans:
            # read the pair file
            with open(os.path.join(self.data_path, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views, data_idx))
                        data_idx += 1

        print(f"Have built metas: {self.data_mode}, smaples # = {len(metas)}")
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, scan, view_idx):
        cam_filename = Path(self.data_path)/'Cameras_1/dtu_cam_train/{}_train/{:0>8}_cam.txt'.format(scan, view_idx)
        with open(cam_filename) as f:
            lines = [line.rstrip() for line in f.readlines()]

        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        #depth_interval = float(lines[11].split()[1]) * self.interval_scale
        depth_max = float(lines[11].split()[-1])
        assert depth_max > depth_min, "Probably you read wrong cam.txt. Check that you do not read depth_interval!!"

        """
        # this intrinsics K is designed for 1/4 scale img (160 x 128);
        """
        img_w, img_h = 160, 128
        # NOTE: Make sure your depth intrinsics matrix is *normalized* by depth dimension;
        K = np.eye(4, dtype=np.float32)
        K[:2,:3] = intrinsics[:2,:3]
        K[0,:] /= img_w # normalized by 1/W
        K[1,:] /= img_h # normalized by 1/H
        #print ("normalized K = \n", K, depth_min, depth_max)
        #return K, extrinsics, depth_min, depth_max, depth_interval
        return K, extrinsics, depth_min, depth_max

    def get_color(self, scan, view_idx, light_idx):
        # NOTE that the id in image file names is from 1 to 49 (not 0~48)
        img_filename = self.get_image_path(scan, view_idx, light_idx)
        color = self.loader(img_filename)
        #if do_flip:
        #    color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def prepare_depth(self, hr_depth):
        depth_crop = half_and_crop_depth(hr_depth, target_h=512, target_w=640)

        if (self.height, self.width) != (512, 640):
            # depth value unit: in mm
            depth_crop = cv2.resize(depth_crop, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return depth_crop

    def read_mask(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        return np_img

    def get_image_path(self, scan, view_idx, light_idx):
        # NOTE that the id in image file names is from 1 to 49 (not 0~48)
        img_filename = Path(self.data_path)/'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(
            scan, view_idx + 1, light_idx)
        return img_filename


    def get_depth_path(self, scan, view_idx):
        depth_filename = Path(self.data_path)/'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, view_idx)
        mask_filename = Path(self.data_path)/'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, view_idx)
        return depth_filename, mask_filename

    def get_depth(self, scan, view_idx, scale):
        # read pfm depth file
        depth_filename, mask_filename = self.get_depth_path(scan, view_idx)
        depth_hr = pfm.readPFM(depth_filename)*scale # high resolution
        depth = self.prepare_depth(depth_hr)
        mask_hr = self.read_mask(mask_filename)
        mask = self.prepare_depth(mask_hr)
        return depth, mask


"""
How to run this file:
- cd ~/manydepth-study/
- python -m src.datasets.dtu_yao
"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.tools.utils import readlines, kitti_colormap
    from src.tools import pfmutil as pfm
    import time
    import sys
    from src.datasets.dataset_util import warp_based_on_depth


    def check_valid_pose(inputs):
        is_valid_pose = False
        for key, ipt in inputs.items():
            if isinstance(key, tuple) and 'pose' in key[0]:
                if torch.isnan(ipt).any():
                    return False
        return True

    ## sanity_check
    def sanity_check_train(my_dataset, data_loader_bs):
        data_loader = torch.utils.data.DataLoader(
            my_dataset, data_loader_bs,
            shuffle=False, # set True for the iter() defined later;
            num_workers= 16,
            pin_memory=True,
            drop_last=False)

        N = len(data_loader)
        for batch_idx, inputs in enumerate(data_loader):
            before_op_time = time.time()
            if not check_valid_pose(inputs):
                print ("Found invalid pose")
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        print(key, value.shape)
                    else:
                        print(key, value)
                return
            if batch_idx % 200 == 0:
                print ("processing batch idx %d out of total %d" %(batch_idx, N))
        print ("Congrates! Sanity check passed!!!")




    # Data loading code
    dataset_name = 'dtu_yao'
    data_path = "/home/ccj/datasets/DTU/dtu_patchmatchnet/"
    data_mode = 'train'
    #data_mode = 'val'
    train_filenames = './splits/dtu/{}.txt'.format(data_mode)
    #train_filenames = './splits/dtu/val.txt'

    #data_mode = 'train'
    #data_mode = 'val'

    height = 512
    width = 640

    frames_to_load = [0, 1, 2, 3, 4]
    nviews = len(frames_to_load)
    num_scales = 4
    #num_scales = 2

    kwargs = {
        'data_mode': data_mode
    }
    dataset = MVSDataset(
        data_path,
        train_filenames,
        height,
        width,
        nviews,
        num_scales,
        #is_train= True,
        #robust_train = True,
        is_train= data_mode == 'train',
        robust_train = False,
        load_depth=True,
        depth_min= 0.425,
        depth_max= 0.935,
        **kwargs
        )


    if 0:
        sanity_check_train(dataset, data_loader_bs = 2)
        sys.exit()
    value_scale = 1.0
    #_DEPTH_MIN = value_scale* 425 # mm to m
    #_DEPTH_MAX = value_scale* 935 # mm to m

    print("Number of samples:", len(dataset))

    batch_idx = 4
    inputs = dataset[batch_idx]

    #for key, value in inputs.items():
    #    print(key, type(value))


    if 1:
        """ numpy version """
        f, ax = plt.subplots(4, nviews)
        #scale = 2
        scale = 0
        ref_img = inputs[('color', 0, scale)].numpy().transpose([1, 2, 0])
        ref_idx = 0
        #dep =  inputs["depth_gt"].squeeze(0).numpy()
        dep = value_scale*inputs[(f"dep_gt_level_{scale}", ref_idx)].squeeze(0).numpy()
        dep_min = value_scale*inputs['depth_min'].item()
        dep_max = value_scale*inputs['depth_max'].item()
        print (f"depth_min={dep_min}, max={dep_max}")
        #dep[dep > dep_max] = dep_max
        dep[dep < dep_min] = dep_min

        mask = inputs[(f"dep_mask_level_{scale}", ref_idx)].squeeze(0).numpy()
        print ("???? dep = ", dep)
        
        K = inputs[("K", ref_idx, scale)].numpy() # 4 x 4
        print ("K ", K.shape)
        ref_Extrins = inputs[("pose", 0)].numpy() # 4 x 4
        print ("E ", ref_Extrins.shape)
        ref_proj_mat = np.matmul(K, ref_Extrins) # 4 x 4
        for i, frame_id in enumerate(frames_to_load):
            img = inputs[('color_aug', frame_id, scale)].numpy().transpose([1, 2, 0])
            ax[0, i].imshow(img)
            if frame_id == 0:
                n = 'scale 1/%d: Ref frm %s'% (2**scale, frame_id)
                n0 = n
            else:
                n = 'scale 1/%d: Src frm %s'% (2**scale, frame_id)
            ax[0, i].set_title('%s'%n)
            ax[1, i].imshow(dep, cmap='gray')
            ax[1, i].set_title('depth for %s'%n0)
            ax[2, i].imshow( kitti_colormap(1.0/dep)) # 1/depth to mimic the kitti color for disparity map;
            ax[2, i].set_title('depth (kitti clr) for %s'%n0)
            cur_E = inputs[("pose", frame_id)].numpy() # 4 x 4
            warped_gt_depth = warp_based_on_depth(
                src_img = img,
                ref_img = ref_img,
                ref_proj_mat = ref_proj_mat,
                src_proj_mat = np.matmul(K, cur_E),
                depth = dep/value_scale,
                #depth = 5.0,
                mask= mask
                )
            ax[3, i].imshow( warped_gt_depth)
            ax[3, i].set_title('warped for %s to ref'%n)
        #plt.savefig("./results/tmp/plt_bt%d.png"%(batch_idx))
        plt.show()
    #sys.exit()
