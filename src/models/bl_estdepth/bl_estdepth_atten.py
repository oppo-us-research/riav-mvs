"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

""" load modules from third_parties/ESTDepth """
#from third_parties.ESTDepth.utils.misc_utils import *
#from third_parties.ESTDepth.utils.utils import *

""" load our own moduels """
from src.models.bl_estdepth.model_hybrid_atten import DepthNetHybrid_atten
from src.layers import BackprojectDepth, Project3D, disp_to_depth
from src.utils.comm import (is_main_process, print0)

#---------------------------------------------
#-- baseline estdepth + our self-attention ---
#---------------------------------------------
class baseline_estdepth(nn.Module):
    def __init__(self, options):
        super(baseline_estdepth, self).__init__() 
        self.opt = options
        self.mode = options.mode # 'train', 'resume', 'val' or 'test'
        self.is_train = str(options.mode).lower() in ['train', 'resume']
        
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.num_scales = len(self.opt.scales)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        #assert self.opt.frame_ids == [0, 1, 2] , "frame_ids must have 3 frames specified"
        assert self.opt.frame_ids == [0, 1, 2, 3, 4], "frame_ids must have 5 frames specified"
        #assert len(self.opt.frame_ids) in [3,5], "frame_ids must have 3 or 5 frames specified"

        # used to identify principal GPU to print some info;
        if is_main_process():
            self.is_print_info_gpu0 = True
        else:
            self.is_print_info_gpu0 = False
        
        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = frames_to_load
        
        self.min_depth_tracker = self.opt.min_depth
        self.max_depth_tracker = self.opt.max_depth
         
        self.is_estd_depth_returned = True
 
        
        if self.is_print_info_gpu0:
            print('Matching_ids : {}'.format(self.matching_ids))
            print('frames_to_load : {}'.format(frames_to_load))
            print ("[***] is_adaptive_bins = ", self.opt.adaptive_bins)

        # MODEL SETUP
        assert self.opt.depth_binning == 'linear', "ESTD requires linear depth binning"
        assert self.opt.num_depth_bins == 64, "ESTD uses M=64 plane hypotheses"

        # small num
        self.epsilon = 1.0e-8
        # MODEL SETUP
        self.encoder = None
        # No teacher_net !!!"
        self.mono_encoder = None
        self.mono_depth = None
        self.pose_encoder = None
        self.pose = None
        
        my_kwargs = {}
        my_kwargs['gma_weights_path'] = self.opt.gma_pretrained_path
        my_kwargs['atten_num_heads'] = 4
        self.depth = DepthNetHybrid_atten(
            ndepths=self.opt.num_depth_bins, 
            depth_min= self.opt.min_depth,
            depth_max= self.opt.max_depth, 
            resnet= 50, # Or 18;
            IF_EST_transformer = True,
            **my_kwargs
            )
        
        # after the depth model initialization;
        pretrain_ckpt = self.opt.pretrain_estdepth_ckpt
        if pretrain_ckpt is not None and pretrain_ckpt != '':
            self.load_pretrained_ckpt(pretrain_ckpt)

        self.loss_type = self.opt.loss_type
        assert self.loss_type == 'L1', "ESTD uses L1 loss type"
        
        # TODO: now just try single scale
        assert len(self.opt.scales) == 1, "baseline ESTD: now just try single scale!!!"
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            setattr(self, 'backproject_depth_{}'.format(scale),
                    BackprojectDepth(self.opt.batch_size, h, w, device=self.device)
                    )

            setattr(self, 'project_3d_{}'.format(scale),
                    Project3D(self.opt.batch_size, h, w)
                    )
    
    
    def load_pretrained_ckpt(self, loadckpt):
        # load pretrained checkpoint
        state_dict = torch.load(loadckpt, map_location=self.device)
        self.depth.load_state_dict(state_dict['model'])
        if self.is_print_info_gpu0:
            print("Loaded weights for model DepthNetHybrid from ckpt {}".format(loadckpt))
    
    def forward(self, inputs, is_train, is_verbose, 
                #min_depth_bin=None, max_depth_bin=None,
                pre_costs=None,
                pre_cam_poses=None
                ):
        """
        ## for multi-gpu parallel efficiency,
        ## here we put warping, loss etc time consuming functions
        ## inside the forward()

        Pass a minibatch through the network and generate images and losses
        """
        outputs = {}
        losses = {}
        img_key = "color_aug" if self.is_train else "color"
        
        # prepare data
        sample_cuda = {}
        # grab poses + frames and stack for input to the multi frame network
        sample_cuda['cam_intr'] = inputs[('K', 0)][:,:3,:3].contiguous().cuda()
        
        assert self.opt.frame_ids == [0, 1, 2, 3, 4]
        #this_orders = [1, 0, 2] #[t-1, t, t+1]
        this_orders = [1, 2, 0, 3, 4] #[t-2, t-1, t, t+1, t+2]
        imgs = [ inputs[('color', i, 0)] for i in this_orders]
        cam_poses = [inputs[("pose_inv", i)] for i in this_orders ]
        dmaps = [ inputs["depth_gt", i] for i in this_orders ]
        sample_cuda['imgs'] = torch.stack(imgs, dim=1).contiguous().cuda()
        #print ("??? sample_cuda['imgs']:", sample_cuda['imgs'].shape)
        sample_cuda['cam_poses'] = torch.stack(cam_poses, dim=1).contiguous().cuda()
        sample_cuda['dmaps'] = torch.stack(dmaps, dim=1).contiguous().cuda()
        sample_cuda['dmasks'] = (sample_cuda['dmaps'] >= self.opt.min_depth) & (sample_cuda['dmaps'] <= self.opt.max_depth)
        views_num_for_loss = len(this_orders) - 2
        
        if len(self.opt.frame_ids) == 3:
            ref_idx = 0
        elif len(self.opt.frame_ids) == 5:
            ref_idx = 1
        else:
            raise NotImplementedError
         
        # load GT pose from dataset;
        pose_pred = self.cal_poses(inputs)

        outputs.update(pose_pred)
        

        if is_train:
            model_outputs, model_losses = self.depth(
                        sample_cuda["imgs"],
                        sample_cuda["cam_poses"],
                        sample_cuda["cam_intr"],
                        sample_cuda,
                        mode='train',
                        pre_costs= pre_costs,
                        pre_cam_poses= pre_cam_poses
                    )
            scale = 0
            losses['loss/L1'] = model_losses[f'loss_{scale}'] # l1: abs loss at scale 0;
            losses['loss/L1-rel'] = model_losses[f'abs_rel_{scale}']
            losses['loss'] = model_losses['loss'] # total loss of l1 at scales 0, 1, 2, 3;

        else:
            model_outputs, model_metrics = self.depth(
                        sample_cuda["imgs"],
                        sample_cuda["cam_poses"],
                        sample_cuda["cam_intr"],
                        sample_cuda,
                        mode='test',
                        pre_costs=None,
                        pre_cam_poses=None
                    )        
            scale = 0
            # four scales in total, we only use the final output 
            # (i.e. scale=0, for full resolution)
            losses['loss/L1'] = model_metrics[f'abs_diff_{scale}']
            losses['loss/L1-rel'] = model_metrics[f'abs_rel_{scale}']
        
        #depth_scales = [0, 1, 2, 3]
        # four scales in total, we only use the final output 
        depth_scales = [0]
        # save to output to follow our code API;
        ref_idx_our = 0
        for img_idx in [ref_idx]:
            for scale in depth_scales:
                tmp_depth = model_outputs[("depth", img_idx, scale)]
                #print ("?? tmp_depth = ", tmp_depth.shape)
                outputs[("depth", ref_idx_our, scale)] = tmp_depth
                outputs[("disp", scale)] = 1.0/(self.epsilon + tmp_depth)

                outputs[("init_prob", ref_idx_our, scale)] = model_outputs[("init_prob", img_idx)]
                outputs[("fused_prob", ref_idx_our, scale)] = model_outputs[("fused_prob", img_idx)]

        self.generate_images_pred(inputs, outputs,
                    is_depth_returned = self.is_estd_depth_returned,
                    is_multi=True)
        
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

                    cam_points = getattr(self, 'backproject_depth_{}'.format(source_scale))(
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = getattr(self, 'project_3d_{}'.format(source_scale))(
                        cam_points, inputs[("K", source_scale)], T)

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




#-----------------------
#-- baseline estdepth ---
#-----------------------
class baseline_estdepth_eval(baseline_estdepth):
    def __init__(self, options):
        super(baseline_estdepth_eval, self).__init__(options)
        print ("[!!!] baseline_estdepth_eval : Reset class attributes !!!")
        options.mode = 'test'
        
    # > see: src/models/bl_estdepth/eval_hybrid.py
    # Joint mode, not the sequential mode;
    def forward(self, data, frames_to_load, is_verbose,
        pre_costs = None,
        pre_cam_poses = None
        ):
        # for test, use 'color' not 'color_aug'
        my_res = {}
        # prepare
        sample_cuda = {}
        # map: {0, 1, 2, 3, 4} 
        #      [2, 1, 0, 3, 4]
        assert frames_to_load == [0, 1, 2, 3, 4]
        # grab poses + frames and stack for input to the multi frame network
        sample_cuda['cam_intr'] = data[('K', 0)][:,:3,:3].contiguous().cuda()
        this_orders = [2, 1, 0, 3, 4]
        imgs = [ data[('color', i, 0)] for i in this_orders]
        cam_poses = [data[("pose_inv", i)] for i in this_orders ]
        dmaps = [ data["depth_gt", i] for i in this_orders ]
        sample_cuda['imgs'] = torch.stack(imgs, dim=1).contiguous().cuda()
        #print (sample_cuda['imgs'].shape)
        sample_cuda['cam_poses'] = torch.stack(cam_poses, dim=1).contiguous().cuda()
        sample_cuda['dmaps'] = torch.stack(dmaps, dim=1).contiguous().cuda()
        sample_cuda['dmasks'] = (sample_cuda['dmaps'] >= self.opt.min_depth) & (sample_cuda['dmaps'] <= self.opt.max_depth)


        outputs, costs, cam_poses = self.depth(
                                    sample_cuda["imgs"],
                                    sample_cuda["cam_poses"],
                                    sample_cuda["cam_intr"],
                                    sample_cuda,
                                    pre_costs= pre_costs,
                                    pre_cam_poses= pre_cam_poses,
                                    mode='val'
                                    )        

        ref_idx = 1
        pred_depth = outputs[("depth", ref_idx, 0)]
        my_res['depth'] = pred_depth
        my_res['pre_costs'] = costs
        my_res['pre_cam_poses'] = cam_poses
        #pred_disp = 1.0/(pred_depth + 1e-8)
        #my_res[("disp", 0)] = pred_disp
        
        ## for print
        if is_verbose:
            _min_depth = pred_depth.min(-1)[0].min(-1)[0]
            _max_depth = pred_depth.max(-1)[0].max(-1)[0]
            print ("  -- pred-depth: min_depth=%f, max_depth=%f" %(
                _min_depth.mean(), _max_depth.mean()))
        
        return my_res