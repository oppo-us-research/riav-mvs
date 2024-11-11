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
import PIL.Image as pil
from PIL import Image

import torch
from torchvision import transforms

""" load our own moduels """
from .mvs_dataset import MVSDatasetBase
from src.utils import pfmutil as pfm


# the DTU dataset preprocessed by Yao Yao (only for training)

"""
test_sets_list = [
    1, 4, 9, 10, 11, 12, 13, 15, 
    23, 24, 29, 
    32, 33, 34, 48, 49, 62, 75, 77, 
    110, 114, 118
    ]
dtu_test_sets = [ "scan%d"%i for i in test_sets_list]
"""

class MVSDataset(MVSDatasetBase):
    def __init__(self, *args, **kwargs):
        super(MVSDataset, self).__init__(*args, **kwargs)

        # specific to DTU-train dataset;
        self.num_scales = 4 # multi image scales: 1/2^i, i=0,1,2,3
        #assert self.num_scales == 4, f"Got self.num_scales={self.num_scales}"

        self.eval_width =  kwargs.get('eval_width', 1600)
        self.eval_height =  kwargs.get('eval_height', 1152)
        self.data_mode = kwargs.get('data_mode')
        self.interval_scale = kwargs.get('depth_interval_scale', 1.06)
        assert self.data_mode == "test", "dtu_yao_eval.py only for test"

        assert self.height%32==0 and self.width%32==0, \
                'img_w and img_h must both be multiples of 32!'
        print (f"[***] DTU_yao_eval: self.data_mode={self.data_mode}\n" + \
               f"      input img size={self.width}x{self.height}\n" + \
               f"      gt depth for evaluation size = {self.eval_width}x{self.eval_height}")

        self.mm_to_meter_factor = 0.001


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
        # dummy light condition
        dummy_light_idx = 0
        data_idx = 0 # global index among the metas;
        # use this data_idx to save the predicted depth 
        # in paralle, without worrying about overwrite; 
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.data_path, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, dummy_light_idx, ref_view, src_views, data_idx))
                    data_idx += 1

        print(f"Have built metas: {self.data_mode}, smaples # = {len(metas)}")
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, scan, view_idx):
        # 1) with fix depth_min and depth_max in the cam.txt;
        cam_filename = Path(self.data_path)/f'{scan}/cams_1/{view_idx:0>8}_cam.txt'
        # Or 2) with more accurate depth_min and depth_max in the cam.txt;
        #cam_filename = Path(self.data_path)/'dtu_cam_test/{scan}/cams_1/{view_idx:0>8}_cam.txt'
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

        """
        # this intrinsics K is designed for img (1600 x 1200);
        """
        img_w, img_h = 1600, 1200
        # NOTE: Make sure your depth intrinsics matrix 
        # is *normalized* by depth dimension;
        K = np.eye(4, dtype=np.float32)
        K[:2,:3] = intrinsics[:2,:3]
        K[0,:] /= img_w # normalized by 1/W
        K[1,:] /= img_h # normalized by 1/H
        #print ("normalized K = \n", K)
        #return K, extrinsics, depth_min, depth_max, depth_interval
        return K, extrinsics, depth_min, depth_max

    def get_color(self, scan, view_idx, dummy_light_idx):
        # NOTE that the id in image file names is from 1 to 49 (not 0~48)
        img_filename = self.get_image_path(scan, view_idx, dummy_light_idx)
        color = self.loader(img_filename)
        # (1600, 1200) is resized to (1600, 1152)
        #color = color.resize((self.width, self.height), Image.ANTIALIAS)
        color = color.resize((self.width, self.height), Image.LANCZOS)
        return color

    def get_image_path(self, scan, view_idx, dummy_light_idx):
        # no light_idx for testset;
        img_filename = Path(self.data_path)/'{}/images/{:0>8}.jpg'.format(scan, view_idx)
        return img_filename

    def get_depth_path(self, scan, view_idx):
        depth_filename = Path(self.data_path)/'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, view_idx)
        mask_filename = Path(self.data_path)/'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, view_idx)
        return depth_filename, mask_filename
    
    def read_mask(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        return np_img
    
    def get_depth(self, scan, view_idx, scale):
        # read pfm depth file
        depth_filename, mask_filename = self.get_depth_path(scan, view_idx)
        # high resolution
        depth = pfm.readPFM(depth_filename)*scale 
        mask = self.read_mask(mask_filename)
        h, w = depth.shape
        assert h == 1200 and w == 1600, "Loading wrong depth size"
        if self.eval_width != w or self.eval_height != h:
            depth = cv2.resize(depth, (self.eval_width, self.eval_height), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (self.eval_width, self.eval_height), interpolation=cv2.INTER_NEAREST)
        return depth, mask

"""
How to run this file:
- cd ~/manydepth-study/
- python -m src.datasets.dtu_yao_eval
"""
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from src.utils.utils import readlines, kitti_colormap
    from src.utils import pfmutil as pfm
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
            num_workers= 4,
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
    dataset_name = 'dtu_yao_eval'
    data_path = "/home/ccj/datasets/DTU/dtu_patchmatchnet_test/"
    data_mode = 'test'
    test_filenames = './splits/dtu/test.txt'
    test_filenames = './splits/dtu/test_small.txt'

    height = 1152
    width = 1600

    frames_to_load = [0, 1, 2, 3, 4, 5]
    nviews = len(frames_to_load)
    num_scales = 4
    #num_scales = 2

    kwargs = {
        'data_mode': data_mode,
        'load_depth_path': True,
        'load_image_path': True,
        'eval_width': width,
        'eval_height' : height
    }
    dataset = MVSDataset(
        data_path,
        test_filenames,
        height,
        width,
        nviews,
        num_scales,
        is_train= False,
        robust_train = False,
        load_depth = True,
        depth_min= 0.425, # mm
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

    batch_idx = 2
    inputs = dataset[batch_idx]

    #for key, value in inputs.items():
    #    print(key, type(value))


    if 1:
        """ numpy version """
        f, ax = plt.subplots(4, nviews)
        scale = 2
        #scale = 0
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
        dep_path = inputs[("depth_gt_path", ref_idx)]
        mask_path = inputs[("mask_gt_path", ref_idx)]
        print (f"depth_path = {dep_path}, mask_path = {mask_path}")

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
