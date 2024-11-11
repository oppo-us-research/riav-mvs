"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys


""" load modules from third_parties/IterMVS """
from third_parties.IterMVS.utils import AbsDepthError_metrics

""" load our own moduels """
from src.layers import (
    transformation_from_parameters,
    warp_frame_depth,
    SSIM
    )

from src import models
from .module import (
    mvsnet_loss, 
    mvsnet_loss_L1,
    compute_mvsnet_losses
)
from .mvsnet import baseline_mvsnet as my_baseline_mvsnet


#-------------------------------
#-- backbone mvsnet + pose -----
#-------------------------------
class baseline_mvsnet(my_baseline_mvsnet):
    def __init__(self, options):
        super(baseline_mvsnet, self).__init__(options)
    
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
                        #num_frames_to_predict_for=2
                        num_frames_to_predict_for=1 # updated by CCJ;
                        )
        if not self.opt.no_ssim:
            self.ssim = SSIM()
        
        our_pretrain_mvsnet_path = self.opt.our_pretrain_mvsnet_path
        if our_pretrain_mvsnet_path and self.is_train:
            self.load_our_pretrained_ckpt(our_pretrain_mvsnet_path)
        
        pretrain_pose_path = self.opt.pretrain_residual_pose_path
        if pretrain_pose_path and self.is_train:
            self.load_pretrained_pose_ckpt(pretrain_pose_path)
 
        
    
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
    

    def forward_no_posenet(self, ref_frame, ref_proj, lookup_frames, src_projs, depth_values, depth_gt, valid_mask): 
        depth, _, L1_cost_volume, _ = self.run_depth(ref_frame, ref_proj, lookup_frames, 
                            src_projs, depth_values, 
                            ref_frame_quarter=None, 
                            refine = False,
                            is_L1_volume=True
                            )
        with torch.no_grad():
            # should choose high value, not min()
            #print ("here here L1_cost_volume shape = ", L1_cost_volume.shape)
            _, argbest = torch.max(L1_cost_volume, dim=1)
            #print ("??? argbest = ", argbest)
            lowest_cost_disp = self.indices_to_disp(argbest, self.depth_bins)
        

        coeff = self.opt.m_2_mm_scale_loss # make loss larger;
        loss = coeff*mvsnet_loss(depth_est = depth,  depth_gt = depth_gt, mask = valid_mask)  
        
        # for summary, no gradient;
        losses = {}
        losses["loss/no_posenet/abs_depth_error"] = AbsDepthError_metrics(coeff*depth, coeff*depth_gt, valid_mask > 0.5)

        #return depth
        return depth, lowest_cost_disp, loss
    
    def forward(self, inputs, is_train, is_verbose, min_depth_bin=None, max_depth_bin=None, is_freeze_fnet=False):
        outputs = {}
        losses = {}
        losses['loss'] = .0
        
        ref_frame_idx = 0
        img_key = "color_aug" if self.is_train else "color"
        #The reference frame
        ref_frame = inputs[img_key, ref_frame_idx, 0]
        # The source frames
        lookup_frames = [inputs[(img_key, idx, 0)] for idx in self.matching_ids[1:]]
        
        bs, _, img_height, img_width = ref_frame.shape
        #if self.is_print_info_gpu0:
        #    print ("??? depth_values = \n", inputs['linear_depth_values'].shape, "\n=", inputs['linear_depth_values'])
        if 'linear_depth_values' in inputs:
            depth_values = inputs['linear_depth_values']
            #print ("??? read data depth_values = ", depth_values.shape)
        else:
            depth_values = self.depth_bins.to(ref_frame.device)
            if depth_values.shape[0] ==1:
                depth_values = depth_values.repeat(bs, 1)
        num_depth = depth_values.shape[1]
        assert num_depth == self.num_depth_bins, f"Wrong num_depth_bins={num_depth}"
        #num_views = len(self.matching_ids)
        
        ## For calculating loss
        s = self.depth_map_scale_int # in quarter-res;
        if (f"dep_gt_level_{s}", ref_frame_idx) in inputs:
            depth_gt = inputs[(f"dep_gt_level_{s}", ref_frame_idx)]# [N,1,H,W]
            mask = inputs[(f"dep_mask_level_{s}", ref_frame_idx)]# [N,1,H,W]
        elif ('depth_gt', 0) in inputs:
            depth_gt = F.interpolate(inputs[("depth_gt", 0)], 
                            #size = (depth.size(2), depth.size(3)), 
                            scale_factor = 1.0/(2**s),
                            mode="nearest" 
                            )
            mask = (depth_gt > .0).float()
            #print ("???", depth.shape, depth_gt.shape)
        else:
            raise NotImplementedError
        
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
        
        s = self.volume_scale # quarter scale, 1/2^2 = 1/4
        ref_proj = inputs[("proj_mat", 0, s)]
        src_projs = [inputs[("proj_mat", idx, s)] for idx in self.matching_ids[1:]]
        
        #---------------------------------------
        # 1) If pose-net exists, we do two-round forward;
        #   first: inference depth without pose-net
        #---------------------------------------
        do_warp_with_pred_depth = random.random() > 0.4
        
        pred_depth_no_resi_pose, lowest_disp_no_resi_pose, loss_no_resi_pose = \
            self.forward_no_posenet(ref_frame, ref_proj, lookup_frames, 
                                    src_projs, depth_values, 
                                    depth_gt, mask,
                                    ) 
        # save to output to follow our code API;
        outputs[('disp/no_resi_pose', self.depth_map_scale_int)] = 1.0 / (pred_depth_no_resi_pose + 1.0e-8)
        outputs['lowest_cost/no_resi_pose'] = lowest_disp_no_resi_pose
        
         
        losses['loss/no_resi_pose']  = loss_no_resi_pose
        # metric only
        coeff = self.opt.m_2_mm_scale_loss
        losses['loss/metric/L1_no_resi_pose']  = coeff*mvsnet_loss_L1(
            pred_depth_no_resi_pose, depth_gt, mask)

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
        s = self.volume_scale # quarter scale, 1/2^2 = 1/4
        #print ("??? Update proj_mat via residual pose")
        ref_proj = inputs[("proj_mat", 0, s)]
        src_projs = [inputs[("proj_mat", idx, s)] for idx in self.matching_ids[1:]]
        if self.refine:
            if (img_key, 0, 2) in inputs:
                ref_frame_quarter = inputs[(img_key, 0, 2)]
            else:
                ref_frame_quarter = F.interpolate(ref_frame, 
                                [img_height//4, img_width//4], 
                                mode="bilinear", 
                                align_corners=True)
        else:
            ref_frame_quarter = None
        
        depth, prob_volume, L1_cost_volume, refined_depth = self.run_depth(
                            ref_frame, 
                            ref_proj.detach(), 
                            #ref_proj, 
                            lookup_frames, 
                            #src_projs, 
                            [src_proj.detach() for src_proj in src_projs], 
                            depth_values,
                            ref_frame_quarter, 
                            refine = self.refine,
                            is_L1_volume=True
                            )
        with torch.no_grad():
            #print ("L1_cost_volume shape = ", L1_cost_volume.shape)
            _, argbest = torch.max(L1_cost_volume, dim=1)
            lowest_cost_disp = self.indices_to_disp(argbest, self.depth_bins)
            # save to output to follow our code API;
            outputs['lowest_cost'] = lowest_cost_disp
 
        # save to output to follow our code API;
        outputs['depth'] = depth
        outputs[("depth", 0, self.depth_map_scale_int)] = depth
        if self.refine:
            outputs[("depth_refine", 0, self.depth_map_scale_int)] = refined_depth
            outputs[('disp_refine', self.depth_map_scale_int)] = 1.0 / (refined_depth + 1.0e-8)
        
        # save depth as scale=0 for image reconstruction
        if self.depth_map_scale_int > 0:
            depth_full = F.interpolate(depth, 
                            [img_height, img_width], 
                            mode="bilinear", 
                            align_corners=True)
            outputs[("depth", 0, 0)] = depth_full
        
        # generate the predicted images before loss calculation;
        #source_scale = self.depth_map_scale_int # mvsnet only returns depth at 1/4 scale;
        source_scale = 0
        for i, frame_id in enumerate(self.opt.frame_ids):
            if ("color", frame_id, source_scale) not in inputs:
                img_0 = inputs[("color", i, 0)]
                inputs[("color", frame_id, source_scale)] = F.interpolate(
                    img_0, 
                    [img_0.shape[2]//(2**source_scale), img_0.shape[3]//(2**source_scale)], 
                    mode="bilinear", 
                    align_corners=True
                    )
        self.generate_images_pred(inputs, outputs,
                    is_depth_returned = self.is_mvsnet_depth_returned,
                    is_multi=True)
        
        #----------------------
        # Added for pose net;
        #----------------------
        optimizer_loss = .0
        scale = 0 # actually we only have 1 scale;
        #source_scale = self.volume_scale
        source_scale = 0
        photo_loss_weight = 10.0
        
        if self.pose and self.pose_encoder:
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
            optimizer_loss = optimizer_loss + photo_loss_weight*reprojection_loss.mean()
        

        #if is_verbose and min_depth_bin is not None:
        #    print ("  [**dep_min/dep_max]: tracker %f/%f" %(
        #        min_depth_bin.min().item(), max_depth_bin.max().item(),
        #    ))

        with torch.no_grad():
            # photometric confidence
            photometric_confidence = self.compute_photometric_conf(num_depth, prob_volume)
            outputs["confidence"] = photometric_confidence #[N,H,W]
 
        #print ("///???/// depth_gt max = ", depth_gt.max(), ", min = ", depth_gt.min())
        #print ("///???/// mask avg = ", mask.mean())
        
        loss_resi_pose = coeff*mvsnet_loss(depth_est=depth, depth_gt=depth_gt, mask=mask)
        # metric only
        losses['loss/metric/L1_resi_pose'] = coeff*mvsnet_loss_L1(depth, depth_gt, mask)
        losses['loss/resi_pose']  = loss_resi_pose
        optimizer_loss = optimizer_loss + loss_resi_pose
        
        if self.refine:
            new_loss = coeff*mvsnet_loss(depth_est = refined_depth,  depth_gt = depth_gt, mask = mask)
            optimizer_loss = optimizer_loss + new_loss
            
        l1_meter, l1_inv_meter, l1_rel_meter, _ = \
            compute_mvsnet_losses(is_train, depth, depth_gt, mask, self.loss_type) 
        
        if is_train: 
            # we will calculate the running loss in run_epoch() function;
            losses['loss/L1'] = losses['loss/L1'] + coeff*l1_meter.avg
            losses['loss/L1-inv'] = losses['loss/L1-inv'] + coeff*l1_inv_meter.avg
            # no need *coeff for relative depth loss;
            losses['loss/L1-rel'] = losses['loss/L1-rel'] + l1_rel_meter.avg
        else:
            # for validation, return the meter loss, to accumulate batch loss;
            losses['batch_l1_meter_sum'] += coeff*l1_meter.sum
            losses['batch_l1_inv_meter_sum'] += coeff*l1_inv_meter.sum
            losses['batch_l1_rel_meter_sum'] += l1_rel_meter.sum
            losses['batch_l1_meter_count'] += l1_meter.count
            #print ("l1_meter", l1_meter)
        

        losses['loss'] = losses['loss'] + optimizer_loss
        
        # for summary, no gradient;
        losses["loss/abs_depth_error"] = AbsDepthError_metrics(coeff*depth, coeff*depth_gt, mask > 0.5)
        
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
    
    def cal_poses(self, inputs, residual_poses = None):
        """
        calculate relative poses using absolute poses, for example, from KITTI Raw;
        """
        outputs = {}
        ref_idx = 0
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
                    # P^ref = K^ref * E^ref = K^ref * T^ref_w
                    #ref_proj = inputs[('proj_mat', ref_idx, s)]
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
                    inputs[('proj_mat', f_i, s)] = new_proj
                    

            ## Syntax: ("cam_T_cam", reference frame, source frame):
            # backward-map the coordinates from target view to source view;
            # then the backward-warped coordinates (aka grid) is used in
            # `torch.NN.F.grid_sample()` func, to generate the synthesized view
            # of source frame (i.e., by warping it to target view);
            outputs[('cam_T_cam', 0, f_i)] = rel_pose

            # save the relative_pose to inputs dict;
            inputs[('relative_pose', f_i)] = rel_pose


        return outputs
    
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




""" Evaluation """
class baseline_mvsnet_eval(baseline_mvsnet):
    def __init__(self, options):
        ## updated for evaluation mode
        print ("[!!!] baseline_pairnet_eval : Reset class attributes !!!")
        options.mode = 'test'

        super(baseline_mvsnet_eval, self).__init__(options)

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
            
            # save the relative_pose to inputs dict;
            data[('relative_pose', f_i)] = rel_pose


    def forward(self, data, frames_to_load, is_verbose, is_residual_pose = True):

        # for test, use 'color' not 'color_aug'
        ref_frame_idx = 0
        img_key = "color"
        my_res = {}
         
        #The reference frame
        ref_frame = data[img_key, ref_frame_idx, 0].cuda()
        lookup_frames = [data[('color', idx, 0)].cuda() for idx in frames_to_load[1:]]
        
        bs, _, img_height, img_width = ref_frame.shape
        #if self.is_print_info_gpu0:
        #    print ("??? depth_values = \n", inputs['linear_depth_values'].shape, "\n=", inputs['linear_depth_values'])
        if 'linear_depth_values' in data:
            depth_values = data['linear_depth_values']
        else:
            depth_values = self.depth_bins
            if depth_values.shape[0] ==1:
                depth_values = depth_values.repeat(bs, 1)
        num_depth = depth_values.shape[1]
        #num_views = len(self.matching_ids)
        
        
        s = self.volume_scale # quarter scale, 1/2^2 = 1/4
        ref_proj = data[("proj_mat", 0, s)].cuda()
        src_projs = [data[("proj_mat", idx, s)].cuda() for idx in self.matching_ids[1:]]
        
        #---------------------------------------
        # 1) inference depth without pose-net
        #---------------------------------------
        pred_depth_no_resi_pose, prob_volume, _ = self.run_depth(
                            ref_frame, ref_proj, lookup_frames, 
                            src_projs, depth_values, 
                            ref_frame_quarter=None, 
                            refine = False)
        #-----------------------------------------------
        # 2) another round of inference depth with pose-net
        #-----------------------------------------------
        # rectify the pose if possible
        #print ("??? cal residual pose eval")
        if self.pose and self.pose_encoder and is_residual_pose:
            residual_poses = self.rectify_poses_warp(
                                    data, 
                                    pred_depth_no_resi_pose)
            # save relative pose to inputs;
            self.cal_poses(data, frames_to_load,residual_poses)
        
            # Updated proj_matrix;
            #print ("??? Update proj_mat via residual pose")
            ref_proj = data[("proj_mat", 0, s)]
            src_projs = [data[("proj_mat", idx, s)] for idx in self.matching_ids[1:]]
        
            if self.refine:
                if (img_key, 0, 2) in data:
                    ref_frame_quarter = data[(img_key, 0, 2)]
                else:
                    ref_frame_quarter = F.interpolate(ref_frame, 
                                    [img_height//4, img_width//4], 
                                    mode="bilinear", 
                                    align_corners=True)
            else:
                ref_frame_quarter = None
            
            pred_depth, prob_volume, refined_depth = self.run_depth(
                                ref_frame, 
                                ref_proj, 
                                lookup_frames, 
                                src_projs, 
                                depth_values,
                                ref_frame_quarter, 
                                refine = self.refine)
        else:
            pred_depth = pred_depth_no_resi_pose
            refined_depth = None
         
        with torch.no_grad():
            # photometric confidence
            photometric_confidence = self.compute_photometric_conf(
                num_depth, prob_volume)
            my_res["confidence"]= photometric_confidence #[N,H,W]
        
        my_res['depth'] = pred_depth
        my_res['disp'] = 1.0/(pred_depth + 1e-8)
        #my_res[("disp", self.img_scales[0])] = pred_disp
        if refined_depth is not None:
            my_res['depth_refine'] = refined_depth
            my_res['disp_refine'] = 1.0/(refined_depth + 1e-8)
        
        ## for print
        if is_verbose:
            _min_depth = pred_depth.min(-1)[0].min(-1)[0]
            _max_depth = pred_depth.max(-1)[0].max(-1)[0]
            print ("  -- pred-depth: min_depth=%f, max_depth=%f" %(
                _min_depth.mean(), _max_depth.mean()))
        
        return my_res

