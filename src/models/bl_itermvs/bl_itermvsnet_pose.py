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
from src import models
from src.layers import (
        BackprojectDepth, 
        Project3D,
        transformation_from_parameters,
        warp_frame_depth,
        SSIM
        )
from .bl_itermvsnet import baseline_itermvs as my_baseline_itermvs

#-------------------------------
#-- backbone itermvs + pose ----
#-------------------------------
class baseline_itermvs(my_baseline_itermvs):
    def __init__(self, options):
        super(baseline_itermvs, self).__init__(options)
        
        # now just try single scale
        assert len(self.opt.scales) == 4, "IterMVS neeeds 4 scales: 1/2^{0,1,2,3}!!!"
        
        # now just try single scale
        for scale in [0]: # only in original resolution;
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            setattr(self, 'backproject_depth_{}'.format(scale),
                    BackprojectDepth(h, w))

            setattr(self, 'project_3d_{}'.format(scale), Project3D(h, w))
 
        # pose_encoder Param #: 11698920;
        # pose_decoder Param #:  1314572;
        ## only R,G,B
        self.num_pose_frames = 2
        self.num_channels = None
        ## (R, G, B, x, y)
        #self.num_pose_frames = 2
        #self.num_channels= self.num_pose_frames*5 + 1
        self.pose_encoder = models.ResnetEncoder(
                                18, True,
                                num_input_images=self.num_pose_frames,
                                num_channels=self.num_channels
                                )
        self.pose = models.PoseDecoder(
                        self.pose_encoder.num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=1 # updated by CCJ;
                        )
        
        our_pretrain_itermvs_path = self.opt.our_pretrain_itermvs_path
        if our_pretrain_itermvs_path and self.is_train:
            self.load_our_pretrained_ckpt(our_pretrain_itermvs_path)
        
        pretrain_pose_path = self.opt.pretrain_residual_pose_path
        if pretrain_pose_path and self.is_train:
            self.load_pretrained_pose_ckpt(pretrain_pose_path)
        if not self.opt.no_ssim:
            self.ssim = SSIM()

    
    def load_pretrained_pose_ckpt(self, pretrain_weights_path):
        old_ckpt = torch.load(pretrain_weights_path)
        old_dict = old_ckpt['state_dict'] if 'state_dict' in old_ckpt else old_ckpt['model']
        #for k, v in old_dict.items():
        #    print (f"{k}: {v.shape}")
        new_dict = {'pose_encoder': {}, 'pose': {}}
        for k, v in old_dict.items():
            if 'module.pose_encoder.' in k:
                new_k = k[len('module.pose_encoder.'):]
                new_dict['pose_encoder'][new_k] = v
            elif 'module.pose.' in k:
                new_k = k[len('module.pose.'):]
                new_dict['pose'][new_k] = v
            else:
                continue

            #if self.is_print_info_gpu0:
            #    print (f"{k} --> {new_k}")
        
        for k in ['pose_encoder', 'pose']:
            try:
                getattr(self, k).load_state_dict(new_dict[k])
                if self.is_print_info_gpu0:
                    print(f"Loaded weights for model {k} from ckpt {pretrain_weights_path}")
            except Exception as e:
                if self.is_print_info_gpu0:
                    print(e)
                    print(f"[XXXXXXXXXXXXX] Skipping ... model {k}")
                #sys.exit()
    

    def forward(self, inputs, is_train, is_verbose, 
                is_regress_loss = True, # e.g., warmup training, disable it;
                do_tb_summary = False,
                do_tb_summary_image = False,
                val_avg_meter = None # only for validation, to accumulate the evluation metrics;
                ):
        
        outputs = {}
        losses = {}
        losses['loss'] = 0.0
        
        # prepare data API;
        ref_frame_idx = 0
        depth_gt = inputs["depth_gt", ref_frame_idx]
        #h, w = depth_gt.size()[2], depth_gt.size()[3]
        #print (h, w)
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
        
        #---------------------------------------
        # 1) If pose-net exists, we do two-round forward;
        #   first: inference depth without pose-net
        #---------------------------------------
        do_warp_with_pred_depth = random.random() > 0.4
        
        pred_depth_no_resi_pose, loss_no_resi_pose, _ = self.forward_no_posenet(
            current_image, lookup_frames, 
            proj_matrices, 
            gt_depths,
            gt_masks,
            min_depth_bin, max_depth_bin,
            is_regress_loss)
        # save to output to follow our code API;
        scale = 0
        outputs[("disp/no_resi_pose", scale)] = 1.0/(self.epsilon + pred_depth_no_resi_pose)
        losses['loss/no_resi_pose']  = loss_no_resi_pose
        losses['loss'] = losses['loss'] + 0.5*loss_no_resi_pose
        

        if do_warp_with_pred_depth: 
            depth_for_warp = pred_depth_no_resi_pose.detach()
        else:
            depth_for_warp = depth_gt


        # rectify the pose if possible
        residual_poses = self.rectify_poses_warp(inputs, depth_for_warp)
        
        # update mask
        for key in list(residual_poses.keys()):
            _key = list(key)
            if _key[0] in ['reproj_mask']:
                _key = tuple(_key)
                outputs[_key] = residual_poses[key]

        #---------------------------------------
        # 2) another round of inference depth with pose-net
        #---------------------------------------
        # save relative pose to inputs;
        pose_pred = self.cal_poses(inputs, residual_poses)
        outputs.update(pose_pred)

        # Updated proj_matrix;
        for s in self.opt.scales:
            # projection matrix
            projs_tmp = [ inputs[("proj_mat", idx, s)]  for idx in self.matching_ids]
            proj_matrices[f'level_{s}'] = torch.stack(projs_tmp, dim=1)  # batch x (1+V) x 3 x h x w

        outputs.update(
            self.depth(current_image, lookup_frames, proj_matrices, min_depth_bin, max_depth_bin)
            )

        if is_verbose:
            print ("  [**dep_min/dep_max]: tracker %f/%f" %(
                min_depth_bin.min().item(), max_depth_bin.max().item(),
            ))
         
        depth_est = outputs["depths"]
        confidences_est = outputs["confidences"]
        depth_upsampled_est = outputs["depths_upsampled"]
        # save to output to follow our code API;
        scale = 0
        outputs[("depth", ref_frame_idx, scale)] = depth_upsampled_est[-1]
        outputs[("disp", scale)] = 1.0/(self.epsilon + depth_upsampled_est[-1])
        
        # generate the predicted images before loss calculation;
        self.generate_images_pred(inputs, outputs)

        # optimizer loss 
        optimizer_loss = full_loss(
                        depth_est, 
                        depth_upsampled_est, 
                        confidences_est, 
                        gt_depths, gt_masks, 
                        min_depth_bin, 
                        max_depth_bin, 
                        is_regress_loss
                    )
        #----------------------
        # Added for pose net;
        #----------------------
        scale = 0 # actually we only have 1 scale;
        photo_loss_weight = 10.0
        
        if self.pose and self.pose_encoder:
            source_scale = 0
            reprojection_losses = []
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                edge_mask = outputs[("reproj_mask", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred*edge_mask, target*edge_mask))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
            losses['loss/reproj_loss/{}'.format(scale)] = reprojection_loss
            if is_verbose:
                print (f"reproj_loss (not multiplied by {photo_loss_weight}) = {reprojection_loss.mean()}")
            optimizer_loss = optimizer_loss + photo_loss_weight *reprojection_loss.mean()
        
        losses['loss'] = losses['loss'] + optimizer_loss
        
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
    
    
    def compute_reprojection_loss(self, pred, target):
        """
        Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    def forward_no_posenet(self, 
            current_image, lookup_frames, 
            proj_matrices, 
            gt_depths, gt_masks, 
            min_depth_bin, 
            max_depth_bin, 
            is_regress_loss
            ):
            
        #ref_frame_idx = 0
        #outputs = {}
        outputs = self.depth(current_image, lookup_frames, proj_matrices, min_depth_bin, max_depth_bin)
        
        if self.is_train:
            depth_est = outputs["depths"]
            confidences_est = outputs["confidences"]
            depth_upsampled_est = outputs["depths_upsampled"]
            pred_depth = outputs["depths_upsampled"][-1]
            pred_confidence = outputs["confidence_upsampled"][-1]
            
            # optimizer loss 
            optimizer_loss = full_loss(
                            depth_est, 
                            depth_upsampled_est, 
                            confidences_est, 
                            gt_depths, gt_masks, 
                            min_depth_bin, 
                            max_depth_bin, 
                            is_regress_loss
                        )
        else:
            pred_depth = outputs["depths_upsampled"]
            pred_confidence = outputs["confidence_upsampled"]
            optimizer_loss = -1.0
        return pred_depth, optimizer_loss, pred_confidence

    def rectify_poses_warp(self, inputs, depth_for_warp):
        """
        Predict residual poses between input frames, to rectify the "Ground Truth" pose,
        E.g., we think the GT pose on ScanNet is not accurate enough;
        """
        outputs = {}
        img_key = "color_aug" if self.is_train else "color"

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            ## try 1): I_ref and I_src, to predict the residual pose;
            #pose_feats = {
            #    f_i: inputs[(img_key, f_i, 0)] for f_i in self.opt.frame_ids
            #    }

            ## try 2): I_ref and warped(I_src), to predict the residual pose;
            pose_feats = {}
            scale_int = 0
            ref_idx = 0
            with torch.no_grad():
                for f_i in self.opt.frame_ids:
                    # feat: [r, g, b, cord_x, cord_y];
                    #pose_feats[f_i] = get_coord_feat(inputs[(img_key, f_i, scale_int)])
                    # feat: [r, g, b];
                    pose_feats[f_i] = inputs[(img_key, f_i, scale_int)]

                if ('inv_K', ref_idx, scale_int) in inputs:
                    invK_ref = inputs[('inv_K', ref_idx, scale_int)]
                else:
                    invK_ref = inputs[('inv_K', scale_int)]

                if depth_for_warp.shape[2] != self.opt.height:
                        depth_for_warp = F.interpolate(
                            depth_for_warp,
                            size = [self.opt.height, self.opt.width],
                            mode = "bilinear",
                            align_corners = True,
                            #mode="nearest"
                        )
                
                for f_i in self.opt.frame_ids[1:]:
                    # get relative pose from ref to src view
                    # T^src_ref = T^src_w * T^w_ref = T^src_w * inv(T^ref_w),
                    # i.e., = Extrinsic_src * inv(Extrinsic_ref);
                    if ('K', f_i, scale_int) in inputs:
                        K_src = inputs[('K', f_i, scale_int)]
                    else:
                        K_src = inputs[('K', scale_int)]

                    rel_pose = torch.matmul(inputs[("pose", f_i)], inputs[("pose_inv", 0)])
                    warped_src_fea, edge_mask = warp_frame_depth(
                        depth_ref = depth_for_warp,
                        src_fea = pose_feats[f_i],
                        relative_pose = rel_pose,
                        K_src = K_src,
                        invK_ref = invK_ref,
                        is_edge_mask = True)
                    pose_feats[f_i] = warped_src_fea
                    outputs[("reproj_mask", f_i, scale_int)] = edge_mask

            for f_i in self.opt.frame_ids[1:]:
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]
                # channel = 11, i.e., (r, b, g, x, y) of ref and src, and edge_mask;
                # channel = 6, i.e., (r, b, g) of ref and src;
                pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]

                axisangle, translation = self.pose(pose_inputs)
                #outputs[("axisangle", 0, f_i)] = axisangle
                #outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                resi_pose = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                outputs[("residual_cam_T_cam", 0, f_i)] = resi_pose.float()

        else:
            raise NotImplementedError

        return outputs


    def cal_poses(self, inputs, residual_poses = None):
        """
        calculate relative poses using absolute poses, for example, from KITTI Raw;
        """
        outputs = {}
        for f_i in self.opt.frame_ids[1:]:
            # get relative pose from ref to src view
            # T^src_ref = T^src_w * T^w_ref = T^src_w * inv(T^ref_w),
            # i.e., = Extrinsic_src * inv(Extrinsic_ref);
            rel_pose = torch.matmul(inputs[("pose", f_i)], inputs[("pose_inv", 0)])

            if residual_poses:
                resi_pose = residual_poses[("residual_cam_T_cam", 0, f_i)]
                # left matrix-multiplication residual pose matrix to adjust pose;
                rel_pose = torch.matmul(resi_pose, rel_pose)

                # ------------------------
                # -------- new ----------
                # update the proj_matric
                # ------------------------
                for s in self.opt.scales:
                    if ('K', f_i, s) in inputs:
                        K_src = inputs[('K', f_i, s)]
                    else:
                        K_src = inputs[('K', s)]
 
                    ##NOTE:
                    # residual pose is for src_new vs src
                    # relative posse is for src_new vs ref
                    # so src_new projection matrix will be:
                    # P^{src_new} = K^{src}*T^{src_new}_w 
                    # = K^{src}*T^{src_new}_{src}*T^{src}_{w} ... Eq (1)
                    # = K^{src}*resi_pose*T^{src}_{ref}*T^{ref}_w
                    # = K^{src}*resi_pose*rel_pose*T^{ref}_w
                    # = K^{src}*resi_pose*rel_pose*inv(K_ref)*K_ref*T^{ref}_w
                    # = K^{src}*resi_pose*rel_pose*inv(K_ref)*P^{ref} ... Eq (2)
                    #So you can use Eq (2) as:
                    # P^{src_new} = K^{src}*resi_pose*rel_pose*inv(K_ref)*P^{ref}
                    #new_proj = K_src@rel_pose @ invK_ref @ ref_proj

                    # Or you can use Eq (1) as
                    # P^{src_new} = K^{src}*T^{src_new}_{src}*T^{src}_{w}
                    new_proj = K_src@resi_pose @ inputs[("pose", f_i)]

                    # save new proj
                    inputs[("proj_mat", f_i, s)] = new_proj
                    

            ## NOTE: Syntax: ("cam_T_cam", reference frame, source frame):
            # backward-map the coordinates from target view to source view;
            # then the backward-warped coordinates (aka grid) is used in
            # `torch.NN.F.grid_sample()` func, to generate the synthesized view
            # of source frame (i.e., by warping it to target view);
            outputs[("cam_T_cam", 0, f_i)] = rel_pose

            # save the relative_pose to inputs dict;
            inputs[('relative_pose', f_i)] = rel_pose


        return outputs


    def generate_images_pred(self, inputs, outputs):
        """
        Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        #for scale in self.opt.scales:
        for scale in [0]:
            source_scale = 0
            depth = outputs[("depth", 0, scale)] # already in full-res;
            outputs[("disp", scale)] = 1.0/(depth + self.epsilon) # for disparity smoothness loss;

            ## since we use supervised loss,
            # just save those results for tensorboard summary;
            #with torch.no_grad():
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                if ('K', frame_id, source_scale) in inputs:
                    #K_src = inputs[('K', source_scale, frame_id)]
                    K_src = inputs[('K', frame_id, source_scale)]
                else:
                    K_src = inputs[('K', source_scale)]

                if ('inv_K', source_scale, 0) in inputs:
                    #invK_ref = inputs[('inv_K', source_scale, 0)]
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


""" Evaluation """
class baseline_itermvs_eval(baseline_itermvs):
    def __init__(self, options):
        ## updated for evaluation mode
        print ("[!!!] supervised_MVSModel_PSS_eval : Reset class attributes !!!")
        options.mode = 'test'# for test, model only return last step depth, as a tensor; 
        # make sure to set is_test = True, before initializing the depth model;
        super(baseline_itermvs_eval, self).__init__(options)

    
    
    # modified for eval/test mode
    def cal_poses(self, data, frames_to_load, residual_poses = None):
        """
        predict the poses and save them to `data` dictionary;
        """
        for f_i in frames_to_load[1:]:
            rel_pose = torch.matmul(data[("pose", f_i)], data[("pose_inv", 0)])
            if residual_poses:
                resi_pose = residual_poses[("residual_cam_T_cam", 0, f_i)]
                # left matrix-multiplication residual pose matrix to adjust pose;
                rel_pose = torch.matmul(resi_pose, rel_pose)
                # ------------------------
                # -------- new ----------
                # update the proj_matric
                # ------------------------
                for s in self.opt.scales:
                    if ('K', f_i, s) in data:
                        K_src = data[('K', f_i, s)]
                    else:
                        K_src = data[('K', s)]

                    
                    ##NOTE:
                    # residual pose is for src_new vs src
                    # relative posse is for src_new vs ref
                    # so src_new projection matrix will be:
                    # P^{src_new} = K^{src}*T^{src_new}_w 
                    # = K^{src}*T^{src_new}_{src}*T^{src}_{w} ... Eq (1)
                    # = K^{src}*resi_pose*T^{src}_{ref}*T^{ref}_w
                    # = K^{src}*resi_pose*rel_pose*T^{ref}_w
                    # = K^{src}*resi_pose*rel_pose*inv(K_ref)*K_ref*T^{ref}_w
                    # = K^{src}*resi_pose*rel_pose*inv(K_ref)*P^{ref} ... Eq (2)
                    #So you can use Eq (2) as:
                    # P^{src_new} = K^{src}*resi_pose*rel_pose*inv(K_ref)*P^{ref}
                    #new_proj = K_src@rel_pose @ invK_ref @ ref_proj

                    # Or you can use Eq (1) as
                    # P^{src_new} = K^{src}*T^{src_new}_{src}*T^{src}_{w}
                    new_proj = K_src@resi_pose @ data[("pose", f_i)]
                    # save new proj
                    data[('proj_mat', f_i, s)] = new_proj

            data[('relative_pose', f_i)] = rel_pose
    
    
    def forward(self, inputs, frames_to_load, is_verbose, is_residual_pose = True):
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
 

        #---------------------------------------
        # 1) inference depth without pose-net
        #---------------------------------------
        if self.pose and self.pose_encoder:
            self.cal_poses(inputs, frames_to_load, residual_poses=None)
            # grab poses + frames and stack for input to the multi frame network
            pred_depth_no_resi_pose, _, pred_confidence_no_resi_pose = self.forward_no_posenet(
                current_image, lookup_frames, 
                proj_matrices, 
                gt_depths,
                gt_masks,
                min_depth_bin, max_depth_bin,
                is_regress_loss = True)
        



        #-----------------------------------------------
        # 2) another round of inference depth with pose-net
        #-----------------------------------------------
        # rectify the pose if possible
        if self.pose and self.pose_encoder and is_residual_pose:
            residual_poses = self.rectify_poses_warp(inputs, pred_depth_no_resi_pose)

            # predict poses for all frames
            self.cal_poses(inputs, frames_to_load, residual_poses)
            
            # Updated proj_matrix;
            for s in self.opt.scales:
                # projection matrix
                projs_tmp = [ inputs[("proj_mat", idx, s)]  for idx in self.matching_ids]
                proj_matrices[f'level_{s}'] = torch.stack(projs_tmp, dim=1)  # batch x (1+V) x 3 x h x w
            
            outputs = self.depth(current_image, lookup_frames, proj_matrices, 
                                min_depth_bin, max_depth_bin)
            pred_depth = outputs["depths_upsampled"]
            pred_confidence = outputs["confidence_upsampled"]
        
        else:
            pred_depth = pred_depth_no_resi_pose
            pred_confidence = pred_confidence_no_resi_pose
            residual_poses = None
            
        
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
