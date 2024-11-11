"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""
# ------------------------------------------------------------------------------------
# Modified from IterMVS (https://github.com/FangjinhuaWang/IterMVS)
# MIT license.
# ------------------------------------------------------------------------------------

import numpy as np
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

""" load modules from third_parties/IterMVS """
from third_parties.IterMVS.utils import (
    AbsDepthError_metrics, 
    Thres_metrics, 
)
from third_parties.IterMVS.models.net import full_loss

""" load our own moduels """
from .sub_nets import Pipeline_atten
from src.utils.comm import (is_main_process, print0)

#------------------------------------
#-- backbone itermvs + attention ----
#------------------------------------
class baseline_itermvs(nn.Module):
    def __init__(self, options):
        super(baseline_itermvs, self).__init__()
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.mode = str(options.mode).lower() # 'train', 'resume', 'val' or 'test'
        self.is_train = self.mode in ['train', 'resume']
        # for train and val, model saves last step depth into a list, 
        # and returns this list (even only with 1 element);
        # but still different stntaxs for train/val vs test; 
        # here we use train/val, not test;
        # fot test, we will use iter_MVS_eval() defined below;
        self.is_test = self.mode in ['test', 'test_small'] 
        self.depth_thres_scale = 1.0e-3 # change meter to mm for thres_1mm metric;
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.num_scales = len(self.opt.scales)
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        # used to identify principal GPU to print some info;
        if is_main_process():
            self.is_print_info_gpu0 = True
        else:
            self.is_print_info_gpu0 = False
        

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = frames_to_load
        
        self.is_adaptive_bins = self.opt.adaptive_bins

        if self.is_print_info_gpu0:
            print ("[***] self.is_test = ", self.is_test)
            print ("[***] self.is_train = ", self.is_train)
            print ("[***] self.mode = ", self.mode)
            print ("[***] is_adaptive_bins = ", self.is_adaptive_bins)
            print ("[***] IterMVS GRU iters = %d" %self.opt.itermvs_iters )
            print('Matching_ids : {}'.format(self.matching_ids))
            print('frames_to_load : {}'.format(frames_to_load))
        
        # small num
        self.epsilon = 1.0e-8
        # MODEL SETUP
        self.encoder = None

        my_kwargs = {}
        my_kwargs['gma_weights_path'] = self.opt.gma_pretrained_path
        my_kwargs['atten_num_heads'] = 4
        self.depth = Pipeline_atten(iteration=self.opt.itermvs_iters, test= self.is_test, **my_kwargs)
        self.mono_encoder = None
        self.mono_depth = None
        self.pose_encoder = None
        self.pose = None

        # TODO: now just try single scale
        assert len(self.opt.scales) == 4, "IterMVS neeeds 4 scales: 1/2^{0,1,2,3}!!!"
    
    def forward(self, inputs, is_train, is_verbose, 
                is_regress_loss = True, # e.g., warmup training, disable it;
                do_tb_summary = False,
                do_tb_summary_image = False,
                val_avg_meter = None # only for validation, to accumulate the evluation metrics;
                ):
        
        outputs = {}
        losses = {}
        
        # prepare data API;
        ref_frame_idx = 0
        depth_gt = inputs["depth_gt", ref_frame_idx]
        
        for s in self.opt.scales:
            if (f'dep_gt_level_{s}', ref_frame_idx) not in inputs:
                #h_scaled, w_scaled = h // (2**s), w // (2**s)
                groundtruth_scaled = F.interpolate(
                    depth_gt,
                    scale_factor= 1.0/(2**s),
                    mode='nearest'
                )
                inputs[(f'dep_gt_level_{s}', ref_frame_idx)] = groundtruth_scaled
                inputs[(f"dep_mask_level_{s}", ref_frame_idx)] = (groundtruth_scaled > 0).float()

        depth_gt_0 = inputs[('dep_gt_level_0', ref_frame_idx)]
        mask_0 = inputs[('dep_mask_level_0', ref_frame_idx)]
        depth_gt_1 = inputs[('dep_gt_level_2', ref_frame_idx)]
        mask_1 = inputs[('dep_mask_level_2', ref_frame_idx)]
        
        current_image = inputs["color_aug", ref_frame_idx, 0]
        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x V_frames x 3 x h x w
        proj_matrices = {}
        gt_masks = {}
        gt_depths = {}
        
        for s in self.opt.scales:
            # projection matrix
            projs_tmp = [ inputs[("proj_mat", idx, s)]  for idx in self.matching_ids]
            proj_matrices[f'level_{s}'] = torch.stack(projs_tmp, dim=1)  # batch x (1+V) x 3 x h x w
            # depth mask
            gt_masks[f'level_{s}'] = inputs[(f'dep_mask_level_{s}', ref_frame_idx)]
            gt_depths[f'level_{s}'] = inputs[(f'dep_gt_level_{s}', ref_frame_idx)]
        
        # if dynamic depth bin max/min provided by the dataloader;
        if self.is_adaptive_bins:
            assert ("min_depth_tracker" in inputs) and ("max_depth_tracker" in inputs)
            min_depth_bin = inputs["min_depth_tracker"] #[N, 1]
            max_depth_bin = inputs["max_depth_tracker"] #[N, 1]
        
        else:
            min_depth_bin = inputs["depth_min"] #[N, 1]
            max_depth_bin = inputs["depth_max"] # [N, 1]
        

        if is_verbose:
            print ("[????] min_depth_bin/max_depth_bin = ", \
                        min_depth_bin.min().item(), max_depth_bin.max().item())
        
        ## Updated to nn.Parameters;
        outputs['min_depth_bin'] = min_depth_bin
        outputs['max_depth_bin'] = max_depth_bin
        
        outputs.update(
            self.depth(current_image, lookup_frames, proj_matrices, min_depth_bin, max_depth_bin)
            )

        depth_est = outputs["depths"]
        confidences_est = outputs["confidences"]
        depth_upsampled_est = outputs["depths_upsampled"]
        # save to output to follow our code API;
        scale = 0
        outputs[("depth", ref_frame_idx, scale)] = depth_upsampled_est[-1]
        outputs[("disp", scale)] = 1.0/(self.epsilon + depth_upsampled_est[-1])

        # optimizer loss 
        optimizer_loss = full_loss(depth_est, depth_upsampled_est, confidences_est, 
                        gt_depths, gt_masks, 
                        min_depth_bin, 
                        max_depth_bin, 
                        is_regress_loss
                    )
        losses['loss'] = optimizer_loss
        
        """ other metrics and images, saved for logging to tensorboard """
        with torch.no_grad():
            if do_tb_summary_image:
                outputs.update(
                    {
                        "itermvs/depth_gt": depth_gt_1 * mask_1,
                        "itermvs/depth_initial": depth_est["combine"][0] * mask_1,
                        "itermvs/depth_final_full": depth_upsampled_est[-1] * mask_0
                    }
                )
                outputs["errormap/initial"] = (depth_est["combine"][0] - depth_gt_1).abs() * mask_1
                outputs["errormap/final_full"] = (depth_upsampled_est[-1] - depth_gt_0).abs() * mask_0

            if do_tb_summary or val_avg_meter is not None:
                losses["loss/L1/initial"] = AbsDepthError_metrics(depth_est["combine"][0], depth_gt_1, mask_1 > 0.5)
                losses["loss/thres1mm/initial"] = Thres_metrics(depth_est["combine"][0], depth_gt_1, mask_1 > 0.5, 1)
                losses["loss/L1/final_full"] = AbsDepthError_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5)
                # threshold = 1mm
                losses["loss/thres1mm/final_full"] = Thres_metrics(
                    depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 1*self.depth_thres_scale)
                # threshold = 2mm
                losses["loss/thres2mm/final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 2*self.depth_thres_scale)
                # threshold = 4mm
                #losses["loss/thres4mm/final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 4*self.depth_thres_scale)
                # threshold = 8mm
                losses["loss/thres8mm/final_full"] = Thres_metrics(depth_upsampled_est[-1], depth_gt_0, mask_0 > 0.5, 8*self.depth_thres_scale)

                iters = self.opt.itermvs_iters + 1
                for j in range(1, iters):
                    losses[f"loss/thres1mm/gru_itr/{j}"] = Thres_metrics(depth_est["combine"][j], depth_gt_1, mask_1 > 0.5, 1)
                    losses[f"loss/L1/gru_itr/{j}"] = AbsDepthError_metrics(depth_est["combine"][j], depth_gt_1, mask_1 > 0.5)
                # accumulate metrics to validation DictAverageMeter();
                #val_avg_meter.update(tensor2float(losses))
                if val_avg_meter is not None:
                    val_avg_meter.update(losses)
        
        return outputs, losses


""" Evaluation """
class baseline_itermvs_eval(baseline_itermvs):
    def __init__(self, options):
        ## updated for evaluation mode
        print ("[!!!] supervised_MVSModel_PSS_eval : Reset class attributes !!!")
        options.mode = 'test'# for test, model only return last step depth, as a tensor; 
        # make sure to set is_test = True, before initializing the depth model;
        super(baseline_itermvs_eval, self).__init__(options)

    def forward(self, inputs, frames_to_load, is_verbose):
        my_res = {}

        # prepare data API;
        ref_frame_idx = 0 
        depth_gt = inputs["depth_gt", ref_frame_idx]
        for s in self.opt.scales:
            if (f'dep_gt_level_{s}', ref_frame_idx) not in inputs:
                #h_scaled, w_scaled = h // (2**s), w // (2**s)
                groundtruth_scaled = F.interpolate(
                    depth_gt,
                    scale_factor= 1.0/(2**s),
                    mode='nearest'
                )
                inputs[(f'dep_gt_level_{s}', ref_frame_idx)] = groundtruth_scaled
                inputs[(f"dep_mask_level_{s}", ref_frame_idx)] = (groundtruth_scaled > 0).float()
        
        
        # for test, use 'color' not 'color_aug'
        current_image = inputs["color", ref_frame_idx, 0]
        lookup_frames = [inputs[('color', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x V_frames x 3 x h x w
        proj_matrices = {}
        gt_masks = {}
        gt_depths = {}
        
        for s in self.opt.scales:
            # projection matrix
            projs_tmp = [ inputs[("proj_mat", idx, s)]  for idx in self.matching_ids ]
            proj_matrices[f'level_{s}'] = torch.stack(projs_tmp, dim=1)  # batch x (1+V) x 3 x h x w
            # depth mask
            gt_masks[f'level_{s}'] = inputs[(f'dep_mask_level_{s}', ref_frame_idx)]
            gt_depths[f'level_{s}'] = inputs[(f'dep_gt_level_{s}', ref_frame_idx)]

        # if dynamic depth bin max/min provided by the dataloader;
        if self.is_adaptive_bins:
            assert ("min_depth_tracker" in inputs) and ("max_depth_tracker" in inputs)
            min_depth_bin = inputs["min_depth_tracker"] #[N, 1]
            max_depth_bin = inputs["max_depth_tracker"] #[N, 1]
        
        else:
            min_depth_bin = inputs["depth_min"] #[N, 1]
            max_depth_bin = inputs["depth_max"] # [N, 1]
 
        #---------------------
        #-- MVS Part ---
        #---------------------
        outputs = self.depth(current_image, lookup_frames, proj_matrices, 
                            min_depth_bin, max_depth_bin)
        pred_depth = outputs["depths_upsampled"]
        pred_confidence = outputs["confidence_upsampled"]
        # save to output
        my_res['confidence'] = pred_confidence
        my_res['depth'] = pred_depth
            

        ## for print
        if is_verbose:
            _min_depth = pred_depth.min(-1)[0].min(-1)[0]
            _max_depth = pred_depth.max(-1)[0].max(-1)[0]
            print ("  -- pred-depth: min_depth=%f, max_depth=%f; ++ tracker: min_depth_bin = %f, max_depth_bin = %f" %(
                _min_depth.mean(), _max_depth.mean(), 
                min_depth_bin.min().item(), max_depth_bin.max().item()
                ))

        return my_res
