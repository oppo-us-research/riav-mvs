"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import sys
from os.path import join as pjoin
import random

import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import InterpolationMode

""" load our own moduels """
from .dataset_util import pil_loader


class MVSDatasetBase(data.Dataset):
    """Superclass for monocular dataloaders
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 nviews, # ref img + source imgs
                 num_scales,
                 is_train,
                 robust_train,
                 load_depth,
                 depth_min,
                 depth_max,
                 **kwargs
                 ):
        super(MVSDatasetBase, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.width, self.height = width, height
        self.robust_train = robust_train
        self.num_scales = num_scales

        self.interp = InterpolationMode.BILINEAR

        self.load_depth = load_depth
        self.load_depth_path = kwargs.get('load_depth_path', False)
        self.load_image_path = kwargs.get('load_image_path', False)
        self.ndepths = kwargs.get('ndepths', 192)

        self.nviews = nviews

        self.is_train = is_train
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        
        self.depth_min = depth_min
        self.depth_max = depth_max

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            self.color_augment = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
            self.color_augment = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)


    def get_view_ids(self, ref_view, src_views, robust_train):
        if robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            #aug_scale = random.uniform(0.8, 1.25) # augmentation scale
            aug_scale = 1.0

        else:
            # use only the reference view and first nviews-1 source views
            view_ids = [ref_view] + src_views[:self.nviews - 1]
            aug_scale = 1 # augmentation scale
        return view_ids, aug_scale

    def preprocess(self, inputs, color_aug):
        """
        Resize color images to the required scales and apply augmentation as specified.

        The color_aug object is created once and applied uniformly to all images 
        in the item. This ensures that all images input to the pose network 
        receive consistent augmentation.
        """

        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                # check it isn't a blank frame - keep _aug as zeros so we can check for it
                if inputs[(n, im, i)].sum() == 0:
                    inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                else:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        raise NotImplementedError

    def build_metas(self):
        raise NotImplementedError

    def __getitem__(self, index):
        #-------------------------------------------
        # to follow the API in manydepth
        #-------------------------------------------
        """
        Returns a single training item from the dataset as a dictionary.
        The dictionary contains torch tensors with keys that are 
        either strings or tuples, which indicate the type and configuration of the data:

        Keys:
            - ("color", <frame_id>, <scale>): Raw color images.
            - ("color_aug", <frame_id>, <scale>): Augmented color images.
            - ("K", scale) : Camera intrinsics.
            - ("inv_K", scale): Inverse camera intrinsics.
            - "depth_gt": Ground truth depth maps.

        Parameters:
            - <frame_id>: An integer representing the temporal step relative 
                          to the current index (e.g., 0, -1, 1).
            - <scale>: An integer indicating the image scale relative to 
                           the full-size image:
                - "-1": Native resolution as loaded from disk.
                -  "i=0, 1, 2 and 3": Resized to (self.width//2^i, self.height//2^i).
        """


        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5


        meta = self.metas[index]
        scan, light_idx, ref_view, src_views, global_id = meta
        # if robust_train=True, randomly sample src_views
        view_ids, aug_scale = self.get_view_ids(ref_view, src_views, self.robust_train)

        inputs["global_id"] = torch.Tensor([global_id])
        if not self.is_train:
            inputs["global_id_str"] = f"{global_id:08d}"

        # NOTE: ~~~
        # change mm value to m: depth and translate in pose;
        aug_scale *= self.mm_to_meter_factor

        for i, vid in enumerate(view_ids):

            #intrinsics: 4 x 4
            #extrinsics: world_to_cam pose
            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(scan, vid)
            extrinsics[:3,3] *= aug_scale # adjust translate "t";

            # get_color must after read_cam_file, for ETH3D dataset;
            inputs[("color", i, -1)] = self.get_color(scan, vid, light_idx)
            
            # adjusting intrinsics to match each scale in the pyramid
            for s in range(self.num_scales):
                K = intrinsics.copy()
                K[0, :] *= self.width // (2 ** s)
                K[1, :] *= self.height // (2 ** s)
                inputs[("numpy_K", i, s)] = K # added for flow by warping depth;
                inputs[("K", i, s)] = torch.from_numpy(K)
                inv_K = np.linalg.pinv(K)
                inputs[("inv_K", i, s)] = torch.from_numpy(inv_K)
                # multiply intrinsics and extrinsics to get projection matrix
                proj_mat = np.matmul(K, extrinsics.copy())
                proj_mat_inv = np.linalg.inv(proj_mat)
                inputs[("proj_mat", i, s)] = torch.from_numpy(proj_mat)
                inputs[("proj_mat_inv", i, s)] = torch.from_numpy(proj_mat_inv)

            # following mono_data API;
            pose = extrinsics.copy()
            pose_inv = np.linalg.inv(pose)
            inputs[("numpy_pose", i)] = pose
            inputs[("pose", i)] = torch.from_numpy(pose)
            inputs[("pose_inv", i)] = torch.from_numpy(pose_inv)

            # save depth for each view, might do depth prediction for each view;
            inputs["depth_min"] = torch.tensor([self.depth_min]) # scalar
            inputs["depth_max"] = torch.tensor([self.depth_max]) # scalar
            
            depth_min = depth_min_ * aug_scale
            depth_max = depth_max_ * aug_scale
            inputs["min_depth_tracker"] = torch.Tensor([depth_min])
            inputs["max_depth_tracker"] = torch.Tensor([depth_max])
            
            linear_depth_values_np = np.linspace(depth_min, depth_max, self.ndepths, dtype=np.float32)
            #print ("??? depth_values_np shape = ", linear_depth_values_np.shape, linear_depth_values_np)
            inputs["linear_depth_values"] = torch.from_numpy(linear_depth_values_np)
                
            
            if self.load_depth:
                depth, mask = self.get_depth(scan, vid, aug_scale)
                inputs[("depth_gt", i)] = torch.from_numpy(depth[None].astype(np.float32)) #[1,H,W]
                inputs[("depth_mask", i)] = torch.from_numpy(mask[None]) #[1, H, W]

                #if self.is_train:
                h, w = depth.shape
                for s in range(self.num_scales):
                    depth_cur = cv2.resize(depth, (w//(2**s), h//(2**s)), interpolation=cv2.INTER_NEAREST)
                    mask_cur = cv2.resize(mask, (w//(2**s), h//(2**s)), interpolation=cv2.INTER_NEAREST)
                    inputs[(f"dep_gt_level_{s}", i)] = torch.from_numpy(depth_cur[None])
                    inputs[(f"dep_mask_level_{s}", i)] = torch.from_numpy(mask_cur[None])

            if self.load_depth_path:
                dep_path, mask_path = self.get_depth_path(scan, vid) 
                inputs[("depth_gt_path", i)] = dep_path
                inputs[("mask_gt_path", i)] = mask_path 
            if self.load_image_path:
                inputs[("image_path", i)] = self.get_image_path(scan, vid, light_idx)

        """
        # since different view has slightly different K,
        # it is not meaningful to hold this K;
        # instead, we use the proj_mat above;
        """
        # adjusting intrinsics to match each scale in the pyramid
        #for s in range(self.num_scales):
        #    K = intrinsics.copy()
        #    K[0, :] *= self.width // (2 ** s)
        #    K[1, :] *= self.height // (2 ** s)

        #    inv_K = np.linalg.pinv(K)
        #    inputs[("numpy_K", s)] = K # added for flow by warping depth;
        #    inputs[("K", s)] = torch.from_numpy(K)
        #    inputs[("inv_K", s)] = torch.from_numpy(inv_K)


        if do_color_aug:
            color_aug = self.color_augment
        else:
            color_aug = lambda x: x

        self.preprocess(inputs, color_aug)

        # delete raw image;
        for i, vid in enumerate(view_ids):
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        return inputs
