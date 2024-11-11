"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from DeepVideoMVS (https://github.com/ardaduz/deep-video-mvs)
# MIT license.
# ------------------------------------------------------------------------------------

import numpy as np
import sys
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

""" load our own moduels """
from src.models.bl_pairnet.dvmvs_pairnet import (
    DVMVS_PairNet_EncoderMatching, 
    DVMVS_PairNet_DepthDecoder_2D
)

from src.layers import (
    BackprojectDepth, Project3D, disp_to_depth
)
from src.loss_utils import  (
    update_losses, 
    LossMeter
)

from src.utils.utils import check_dict_k_v
from src.utils.comm import (get_rank, is_main_process, print0)

#-----------------------
#-- baseline pairnet ---
#-----------------------
class baseline_pairnet(nn.Module):
    def __init__(self, options):
        super(baseline_pairnet, self).__init__()
        self.opt = options
        self.mode = options.mode # 'train', 'resume', 'val' or 'test'
        self.is_train = str(options.mode).lower() in ['train', 'resume']
        
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.num_scales = len(self.opt.scales)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        # used to identify principal GPU to print some info;
        self.is_print_info_gpu0 = is_main_process()
        
        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = frames_to_load
        
        self.min_depth_tracker = self.opt.min_depth
        self.max_depth_tracker = self.opt.max_depth
         
        self.is_pairnet_depth_returned = True
        
        if self.is_print_info_gpu0:
            print('Matching_ids : {}'.format(self.matching_ids))
            print('frames_to_load : {}'.format(frames_to_load))
            print ("[***] is_adaptive_bins = ", self.opt.adaptive_bins)

        # MODEL SETUP
        assert self.opt.depth_binning == 'inverse', "PairNet requires inverse depth binning"
        assert self.opt.num_depth_bins == 64, "PairNet uses M=64 plane hypotheses"
        device = torch.device(f"cuda:{get_rank()}")
        self.encoder = DVMVS_PairNet_EncoderMatching(
                input_height=self.opt.height,
                input_width=self.opt.width,
                mode = self.mode,
                min_depth_bin=self.opt.min_depth,
                max_depth_bin=self.opt.max_depth,
                adaptive_bins = self.opt.adaptive_bins, # e.g., DTU dataset, has dep min/max per image;
                depth_binning=self.opt.depth_binning,
                num_depth_bins=self.opt.num_depth_bins,
                device = device,
                is_print_info_gpu0 = self.is_print_info_gpu0,
                pretrain_weights_dir = self.opt.pretrain_dvmvs_pairnet_dir
            )

        self.depth = DVMVS_PairNet_DepthDecoder_2D(
                input_channels = self.opt.num_depth_bins,
                opt_min_depth = self.opt.min_depth,
                opt_max_depth = self.opt.max_depth,
                is_print_info_gpu0 = self.is_print_info_gpu0,
                pretrain_weights_dir = self.opt.pretrain_dvmvs_pairnet_dir,
                use_cost_augmentation = self.opt.use_cost_augmentation
            )

        self.loss_type = self.opt.loss_type
        assert self.loss_type == 'L1-inv', "PairNet uses L1-inv loss type"

        # No teacher_net !!!"
        self.mono_encoder = None
        self.mono_depth = None
        self.pose_encoder = None
        self.pose = None

        # TODO: now just try single scale
        assert len(self.opt.scales) == 1, "baseline PairNet: now just try single scale!!!"
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            setattr(self, 'backproject_depth_{}'.format(scale),
                    BackprojectDepth(h, w))

            setattr(self, 'project_3d_{}'.format(scale), Project3D(h, w))

    def forward(self, inputs, is_train, is_verbose, min_depth_bin=None, max_depth_bin=None, is_freeze_fnet=False):
        """
        ## for multi-gpu parallel efficiency,
        ## here we put warping, loss etc time consuming functions
        ## inside the forward()

        Pass a minibatch through the network and generate images and losses
        """
        
        outputs = {}
        losses = {}
        img_key = "color_aug" if self.is_train else "color"
        
        #The reference frame
        ref_frame = inputs[img_key, 0, 0]
        # The source frames
        lookup_frames = [inputs[(img_key, idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w
        
        if is_train:
            # batch loss
            losses['loss/L1'] = .0
            losses['loss/L1-inv'] = .0
            losses['loss/L1-rel'] = .0
        else:
            # keep the sum and count for accumulation;
            losses['batch_l1_meter_sum'] = .0
            losses['batch_l1_inv_meter_sum'] = .0
            losses['batch_l1_rel_meter_sum'] = .0
            losses['batch_l1_meter_count'] = .0
        
        # load GT pose from dataset;
        pose_pred = self.cal_poses(inputs)

        outputs.update(pose_pred)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        lookup_frame_ids = self.matching_ids[1:]

        batch_size, num_frames, _, height, width = lookup_frames.shape

        scale_int = 1 # in half resopution, 1/2^1;
        ref_idx = 0
        if ('inv_K', ref_idx, scale_int) in inputs:
            invK_ref = inputs[('inv_K', ref_idx, scale_int)]
        else:
            invK_ref = inputs[('inv_K', scale_int)]
        # Updated: assuming distinctive K per frame;
        Ks_src = []
        for f_i in self.matching_ids[1:]:
            if ('K', f_i, scale_int) in inputs:
                K_src = inputs[('K', f_i, scale_int)]
            else:
                K_src = inputs[('K', scale_int)]
            Ks_src.append(K_src)
        Ks_src = torch.stack(Ks_src, 1)  # batch x frames x 4 x 4
        #print ("????///////////////", invK_ref.shape, Ks_src.shape)
        
        #---------------------
        # multi frame path
        #---------------------
        encoder_output = self.encoder(
                                ref_frame,
                                lookup_frames,
                                relative_poses,
                                Ks_src, # in half scale, i.e., 2^1=2
                                invK_ref,
                                min_depth_bin=None,
                                max_depth_bin=None,
                                is_freeze_fnet = is_freeze_fnet
                            )

        #import pdb
        #pdb.set_trace()

        lowest_cost = encoder_output['lowest_cost']
        outputs.update(
            self.depth(encoder_output, is_train = self.is_train))
        
        if is_verbose and (min_depth_bin and max_depth_bin):
            print ("  [**dep_min/dep_max]: tracker %f/%f" %(min_depth_bin, max_depth_bin))

        # already be 1/depth for visualization;
        outputs["lowest_cost"] = F.interpolate(lowest_cost,
                                               [self.opt.height, self.opt.width],
                                               mode="nearest")

        self.generate_images_pred(inputs, outputs,
                    is_depth_returned = self.is_pairnet_depth_returned,
                    is_multi=True)

        l1_meter, l1_inv_meter, l1_rel_meter, optimizer_loss = \
            self.compute_pairnet_losses(
                is_train,
                inputs,
                outputs,
                min_depth = self.opt.min_depth,
                max_depth = self.opt.max_depth,
                loss_type = self.loss_type
            )
        if is_train: 
            # we will calculate the running loss in run_epoch() function;
            losses['loss/L1'] = losses['loss/L1'] + l1_meter.avg
            losses['loss/L1-inv'] = losses['loss/L1-inv'] + l1_inv_meter.avg
            losses['loss/L1-rel'] = losses['loss/L1-rel'] + l1_rel_meter.avg
        else:
            # for validation, return the meter loss, to accumulate batch loss;
            losses['batch_l1_meter_sum'] += l1_meter.sum
            losses['batch_l1_inv_meter_sum'] += l1_inv_meter.sum
            losses['batch_l1_rel_meter_sum'] += l1_rel_meter.sum
            losses['batch_l1_meter_count'] += l1_meter.count
            #print ("l1_meter", l1_meter)
        
        losses['loss'] = optimizer_loss
        # debuging
        #check_dict_k_v(outputs, 'outputs')
        #check_dict_k_v(losses, 'losses')

        return outputs, losses

    def cal_poses(self, inputs):
        """
        calculate relative poses using absolute poses, for example, from KITTI Raw;
        """
        outputs = {}
        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # get relative pose from ref to src view
                # T^src_ref = T^src_w * T^w_ref = T^src_w * inv(T^ref_w),
                # i.e., = Extrinsic_src * inv(Extrinsic_ref);
                rel_pose = torch.matmul(inputs[("pose", f_i)], inputs[("pose_inv", 0)])
                ## CCJ: Syntax: ("cam_T_cam", reference frame, source frame):
                # backward-map the coordinates from target view to source view;
                # then the backward-warped coordinates (aka grid) is used in
                # `torch.NN.F.grid_sample()` func, to generate the synthesized view
                # of source frame (i.e., by warping it to target view);
                outputs[("cam_T_cam", 0, f_i)] = rel_pose.float()

                # this is used for consistency loss if needed
                #rel_pose = torch.matmul(inputs[("pose_inv", 0)], inputs[("pose", f_i)])
                #outputs[("cam_T_cam", f_i, 0)] = rel_pose.float()

        # now we need poses for matching
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.matching_ids[1:]:
            rel_pose = torch.matmul(inputs[("pose", f_i)], inputs[("pose_inv", 0)])
            # set missing images to 0 pose
            for batch_idx, feat in enumerate(pose_feats[f_i]):
                if feat.sum() == 0:
                    rel_pose[batch_idx] *= 0
            # save the relative_pose to inputs dict;
            inputs[('relative_pose', f_i)] = rel_pose
        return outputs


    def generate_images_pred(self, inputs, outputs, is_depth_returned = False, is_multi=True):
        """
        Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            source_scale = 0
            if not is_depth_returned: # if depth decoder network returns disparity;
                disp = outputs[("disp", scale)]
                # change disparity to depth
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", 0, scale)] = depth # in full-res;
            else: # load depth
                depth = outputs[("depth", 0, scale)] # already in full-res;
                outputs[("disp", scale)] = 1.0/(depth + 1e-8) # for disparity smoothness loss;

            
            ## since we use supervised loss,
            # just save those results for tensorboard summary;
            with torch.no_grad():
                for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                    T = outputs[("cam_T_cam", 0, frame_id)]
                    
                    if ('K', frame_id, source_scale) in inputs:
                        K_src = inputs[('K', frame_id, source_scale)]
                    else:
                        K_src = inputs[('K', source_scale)]

                    if ('inv_K', source_scale, 0) in inputs:
                        invK_ref = inputs[('inv_K', 0, source_scale)]
                    else:
                        invK_ref = inputs[('inv_K', source_scale)]

                    cam_points = getattr(self, 'backproject_depth_{}'.format(source_scale))(
                        depth, invK_ref)
                    pix_coords = getattr(self, 'project_3d_{}'.format(source_scale))(
                        cam_points, K_src, T)

                    #outputs[("sample", frame_id, scale)] = pix_coords
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        pix_coords,
                        padding_mode="border", align_corners=True)

                        
                    if 'depth_gt' in inputs:
                        depth_gt = inputs["depth_gt"] # [N,1,H,W]
                        cam_points = getattr(self, 'backproject_depth_{}'.format(source_scale))(
                            depth_gt, inputs[("inv_K", source_scale)])
                        pix_coords = getattr(self, 'project_3d_{}'.format(source_scale))(
                            cam_points, inputs[("K", source_scale)], T)

                        outputs[("color_gtdepth", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords,
                            padding_mode="border", align_corners=True)


                    # for each iteration in RAFT
                    if is_multi:
                        if not is_depth_returned:
                            num_iters = len(outputs[('disp_iters', scale)])
                            itr_list = range(1, num_iters) # skip the disp_init
                        elif ('depth_iters', scale) in outputs: # add loss to each iteration flow;
                            num_iters = len(outputs[('depth_iters', scale)])
                            itr_list = range(0, num_iters)
                        else: # just use loss on the flow at last iteration step;
                            itr_list = [-1]

                        for itr in itr_list:
                            if ('disp_iters', scale) in outputs: # add loss to each iteration flow;
                                # change disparity to depth
                                disp_cur = outputs[('disp_iters', scale)][itr] #Nx1xHxW
                                _, depth_cur = disp_to_depth(disp_cur, self.opt.min_depth, self.opt.max_depth)
                            elif ('depth_iters', scale) in outputs:
                                depth_cur = outputs[('depth_iters', scale)][itr] #Nx1xHxW

                            else:
                                depth_cur = outputs[('depth', 0, 0)]

                            cam_points = getattr(self, 'backproject_depth_{}'.format(source_scale))(
                                depth_cur, inputs[("inv_K", source_scale)])
                            pix_coords = getattr(self, 'project_3d_{}'.format(source_scale))(
                                cam_points, inputs[("K", source_scale)], T)

                            outputs[("color_iters", frame_id, scale, itr)] = F.grid_sample(
                                inputs[("color", frame_id, source_scale)],
                                pix_coords,
                                padding_mode="border",
                                #padding_mode="zeros",
                                align_corners=True)



    def compute_pairnet_losses(self, is_train, inputs, outputs, min_depth, max_depth, loss_type):
        # loss accumulator
        l1_meter = LossMeter()
        l1_inv_meter = LossMeter()
        l1_rel_meter = LossMeter()
        if ('depth_gt', 0) in inputs:
            depth_gt = inputs[("depth_gt", 0)] # [N,1,H,W]
        else:
            depth_gt = inputs["depth_gt"] # [N,1,H,W]
        
        if ("depth_mask", 0) in inputs:
            mask = inputs[("depth_mask", 0)] # float type;
        else:
            mask = (depth_gt >= min_depth) & (depth_gt <= max_depth)
            mask = mask.float()
        #print ("[???] depth_gt = ", depth_gt.shape)
        weights = [1, 1, 1, 1, 1]
        pairnet_scales = [4, 3, 2, 1, 0] # 1/2^i scale;
        predictions = [outputs[("depth", 0, scale)] for scale in pairnet_scales]
        optimizer_loss = update_losses(
                            predictions=predictions,
                            weights=weights,
                            groundtruth= depth_gt,
                            valid_mask = mask,
                            is_training= is_train,
                            l1_meter = l1_meter,
                            l1_inv_meter = l1_inv_meter,
                            l1_rel_meter = l1_rel_meter,
                            loss_type = loss_type)
        #print ("after updated_losses: l1_meter ", l1_meter)
        return l1_meter, l1_inv_meter, l1_rel_meter, optimizer_loss



""" Evaluation """
class baseline_pairnet_eval(baseline_pairnet):
    def __init__(self, options):
        ## updated for evaluation mode
        print ("[!!!] baseline_pairnet_eval : Reset class attributes !!!")
        options.mode = 'test'

        super(baseline_pairnet_eval, self).__init__(options)

    # modified for eval/test mode
    def cal_poses(self, data, frames_to_load):
        """
        predict the poses and save them to `data` dictionary;
        """
        for fi in frames_to_load[1:]:
            rel_pose = torch.matmul(data[("pose", fi)], data[("pose_inv", 0)])
            data[('relative_pose', fi)] = rel_pose


    def forward(self, data, frames_to_load, is_verbose):

        # for test, use 'color' not 'color_aug'
        input_color = data[('color', 0, 0)]
        my_res = {}
        
        ## use 'color' not 'color_aug'
        lookup_frames = [data[('color', idx, 0)] for idx in frames_to_load[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w
        #lookup_frame_ids = frames_to_load[1:]
        #batch_size, num_frames, _, height, width = lookup_frames.shape

        ## mvs frames and poses
        # predict poses for all frames
        self.cal_poses(data, frames_to_load)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        if self.opt.adaptive_bins:
            assert ("min_depth_tracker" in data) and ("max_depth_tracker" in data)
            min_depth_bin = data["min_depth_tracker"] #[N, 1]
            max_depth_bin = data["max_depth_tracker"] #[N, 1]
        
        else:
            min_depth_bin = data["depth_min"] #[N, 1]
            max_depth_bin = data["depth_max"] # [N, 1]
        
        if torch.cuda.is_available():
            min_depth_bin = min_depth_bin.cuda()
            max_depth_bin = max_depth_bin.cuda()
        
        ref_idx = 0
        scale_int = 1 # half scale;
        if ('inv_K', ref_idx, scale_int) in data:
            invK_ref = data[('inv_K', ref_idx, scale_int)]
        else:
            invK_ref = data[('inv_K', scale_int)]
        
        # Updated: assuming distinctive K per frame;
        Ks_src = []
        for f_i in frames_to_load[1:]:
            if ('K', f_i, scale_int) in data:
                K_src = data[('K', f_i, scale_int)]
            else:
                K_src = data[('K', scale_int)]
            Ks_src.append(K_src)
        Ks_src = torch.stack(Ks_src, 1)  # batch x frames x 4 x 4
        

        #print ("????///////////////", invK_ref.shape, Ks_src.shape)
        #print ("????///////////////", invK_ref, Ks_src)

        if torch.cuda.is_available():
            input_color = input_color.cuda()
            lookup_frames = lookup_frames.cuda()
            relative_poses = relative_poses.cuda()

            Ks_src = Ks_src.cuda()
            invK_ref = invK_ref.cuda()

        #---------------------
        #-- MVS Part ---
        #---------------------
        encoder_output = self.encoder(
                input_color,
                lookup_frames,
                relative_poses,
                Ks_src, # in half scale, i.e., 2^1=2
                invK_ref,
                None,
                None
            )
        encoder_output['cost_volume_raft'] = None
        output = self.depth(encoder_output)

        if not self.is_pairnet_depth_returned:
            # get scaled disparity;
            pred_disp, pred_depth = disp_to_depth(output[("disp", 0)], \
                self.opt.min_depth, self.opt.max_depth)

        else: # load depth
            pred_depth = output[("depth", 0, 0)]
            pred_disp = 1.0/(pred_depth + 1e-8)
            output[("disp", 0)] = pred_disp


        my_res['disp'] = pred_disp
        my_res['depth'] = pred_depth

        ## for print
        if is_verbose:
            _min_depth = pred_depth.min(-1)[0].min(-1)[0]
            _max_depth = pred_depth.max(-1)[0].max(-1)[0]
            print ("  -- pred-depth: min_depth=%f, max_depth=%f" %(
                _min_depth.mean(), _max_depth.mean()))

        return my_res