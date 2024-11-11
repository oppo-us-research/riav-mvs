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
from einops import rearrange, reduce, repeat

""" load modules from third_parties/IterMVS """
from third_parties.IterMVS.utils import AbsDepthError_metrics

""" load our own moduels """
from src.models.mvsdepth.att_gma import (
    Attention, FeatureAggregator
)
from src import models
from .mvsnet import baseline_mvsnet as my_baseline_mvsnet
from .module import (
    mvsnet_loss, 
    compute_mvsnet_losses, 
    homography_warp
)


#------------------------------------
#-- backbone mvsnet + attention -----
#------------------------------------
class baseline_mvsnet(my_baseline_mvsnet):
    def __init__(self, options):
        super(baseline_mvsnet, self).__init__(options)
        
        #---------------------------------
        # newly added methods and variables
        #--------------------------------- 
        self.atten_num_heads = 4
        self.fmap_dim = fmap_dim = 32 # feature_fusion output dim;

        # attention mechanism
        self.f1_att = Attention(
                        dim= fmap_dim, 
                        heads= self.atten_num_heads,
                        max_pos_size=160, 
                        dim_head= 128, 
                        position_type = 'position_and_content'
                        )
        
        self.f1_aggregator = FeatureAggregator(
                        input_dim = fmap_dim, 
                        head_dim = 128, 
                        num_heads = self.atten_num_heads
                        ) 
        
        # gma module
        pretrain_atten_path = self.opt.pretrain_atten_path
        if pretrain_atten_path and self.is_train:
            self.load_pretrained_atten_ckpt(pretrain_atten_path)
    
        
    def load_pretrained_atten_ckpt(self, pretrain_weights_path):
        old_ckpt = torch.load(pretrain_weights_path)
        old_dict = old_ckpt['state_dict'] if 'state_dict' in old_ckpt else old_ckpt['model']
        #for k, v in old_dict.items():
        #    print (f"{k}: {v.shape}")
        new_dict = {'f1_att': {}, 'f1_aggregator': {}}
        #sys.exit()
        for k, v in old_dict.items():
            if 'module.depth.f1_att.' in k:
                new_k = k[len('module.depth.f1_att.'):]
                new_dict['f1_att'][new_k] = v
            if 'module.depth.f1_aggregator.' in k:
                new_k = k[len('module.depth.f1_aggregator.'):]
                new_dict['f1_aggregator'][new_k] = v
            else:
                continue

            #if self.is_print_info_gpu0:
            #    print (f"{k} --> {new_k}")
        
        for k in ['f1_att', 'f1_aggregator']:
            try:
                getattr(self, k).load_state_dict(new_dict[k])
                if self.is_print_info_gpu0:
                    print(f"Loaded weights for model {k} from ckpt {pretrain_weights_path}")
            except Exception as e:
                if self.is_print_info_gpu0:
                    print(e)
                    print(f"[XXXXXXXXXXXXX] Skipping ... model {k}")
                #sys.exit()
    
    def run_depth(self, ref_frame, ref_proj, lookup_frames, src_projs, depth_values, ref_frame_quarter=None, refine=False): 
        #bs, _, img_height, img_width = ref_frame.shape
        num_depth = depth_values.shape[1]
        num_views = len(self.matching_ids) 
        
        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature = self.encoder(ref_frame)
        
        # attention mechanism to ref feature
        attention = self.f1_att(ref_feature)
        ref_feat_global = self.f1_aggregator(attention, ref_feature)
        #print ("??? Attention global feature ...")

        src_features = [self.encoder(img) for img in lookup_frames]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feat_global.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        # if torch.isnan(ref_proj).any():
        #     print ("Found nan,  ref_proj= ", ref_proj.shape)
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homography_warp(src_fea, src_proj, ref_proj, depth_values)
            
            #if torch.isnan(warped_volume).any():
            #    print ("Found nan,  warped_volume= ", warped_volume.shape)
            #    import pdb
            #    pdb.set_trace()
            
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        #NOTE: Var(X) = E[(X-E[X])^2) = E[X^2] - (E[X])^2
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization and depth regression;
        outputs = self.depth(volume_variance, depth_values, ref_frame_quarter, refine)
        depth = outputs['depth']
        prob_volume = outputs['prob_volume']
        if refine:
            refined_depth = outputs["refined_depth"]
        else:
            refined_depth = None
        return depth, prob_volume, refined_depth
    
    def forward(self, inputs, is_train, is_verbose, min_depth_bin=None, max_depth_bin=None, is_freeze_fnet=False):
        outputs = {}
        losses = {}
        
        
        img_key = "color_aug" if self.is_train else "color"
        ref_frame_idx = 0
        #The reference frame
        ref_frame = inputs[img_key, ref_frame_idx, 0]
        # The source frames
        lookup_frames = [inputs[(img_key, idx, 0)] for idx in self.matching_ids[1:]]
        
        bs, _, img_height, img_width = ref_frame.shape
        #if self.is_print_info_gpu0:
        #    print ("??? depth_values = \n", inputs['linear_depth_values'].shape, "\n=", inputs['linear_depth_values'])
        if 'linear_depth_values' in inputs:
            depth_values = inputs['linear_depth_values']
        else:
            depth_values = self.depth_bins
            depth_values = depth_values.repeat(bs// depth_values.size(0), 1).to(ref_frame.device)
            
        num_depth = depth_values.shape[1]
        assert num_depth == self.num_depth_bins, f"Wrong num_depth_bins={num_depth}"
        #num_views = len(self.matching_ids)
        
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
        
        s = self.volume_scale # quarter scale, 1/2^2 = 1/4
        ref_proj = inputs[("proj_mat", 0, s)]
        src_projs = [inputs[("proj_mat", idx, s)] for idx in self.matching_ids[1:]]
        
        if (img_key, 0, 2) in inputs:
            ref_frame_quarter = inputs[img_key, 0, 2]
        else:
            ref_frame_quarter = F.interpolate(ref_frame, 
                            [img_height//4, img_width//4], 
                            mode="bilinear", 
                            align_corners=True)

        depth, prob_volume, refined_depth = self.run_depth(ref_frame, ref_proj, lookup_frames, 
                            src_projs, 
                            depth_values,
                            ref_frame_quarter, 
                            refine = self.refine)
        
        outputs[("depth", 0, self.depth_map_scale_int)] = depth
        outputs[("disp", self.depth_map_scale_int)] = 1.0 / (depth + 1.0e-8)
        # save depth as scale=0 for image reconstruction
        if self.depth_map_scale_int > 0:
            depth_full = F.interpolate(depth, 
                            [img_height, img_width], 
                            mode="bilinear", 
                            align_corners=True)
            outputs[("depth", 0, 0)] = depth_full
            outputs[("disp", 0)] = 1.0 / (depth_full + 1.0e-8)

        
        self.generate_images_pred(inputs, outputs,
                    is_depth_returned = self.is_mvsnet_depth_returned,
                    is_multi=True)

        with torch.no_grad():
            # photometric confidence
            photometric_confidence = self.compute_photometric_conf(
                num_depth, prob_volume)
            outputs["confidence"]= photometric_confidence #[N,H,W]

        ## calculating loss
        s = self.depth_map_scale_int # in quarter-res;
        if (f"dep_gt_level_{s}", ref_frame_idx) in inputs:
            depth_gt = inputs[(f"dep_gt_level_{s}", ref_frame_idx)]# [N,1,H,W]
            mask = inputs[(f"dep_mask_level_{s}", ref_frame_idx)]# [N,1,H,W]
        elif ('depth_gt', 0) in inputs:
            depth_gt = F.interpolate(inputs[("depth_gt", 0)], 
                            scale_factor= 1.0/(2**s),
                            mode="nearest" 
                            )
            mask = (depth_gt >= self.opt.min_depth) & (depth_gt <= self.opt.max_depth)
            mask = mask.float()
        else:
            raise NotImplementedError
         
        
        coeff = self.opt.m_2_mm_scale_loss # make loss larger;
        optimizer_loss = coeff*mvsnet_loss(depth_est = depth,  depth_gt = depth_gt, mask = mask)
        
        if self.refine:
            outputs[("depth_refine", 0, self.depth_map_scale_int)] = refined_depth
            outputs[('disp_refine', 0)] = 1.0 / (refined_depth + 1.0e-8)
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
        

        losses['loss'] = optimizer_loss
        
        # for summary, no gradient;
        losses["loss/abs_depth_error"] = AbsDepthError_metrics(coeff*depth, coeff*depth_gt, mask > 0.5)
        
        return outputs, losses

    

""" Evaluation """
class baseline_mvsnet_eval(baseline_mvsnet):
    def __init__(self, options):
        ## updated for evaluation mode
        print ("[!!!] baseline_pairnet_eval : Reset class attributes !!!")
        options.mode = 'test'

        super(baseline_mvsnet_eval, self).__init__(options)

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
                depth_values = depth_values.repeat(bs//depth_values.size(0), 1).to(ref_frame.device)
        
        num_depth = depth_values.shape[1]
        num_views = len(self.matching_ids)
        
        ## mvs frames and poses
        # predict poses for all frames
        self.cal_poses(data, frames_to_load)
        
        s = self.volume_scale # quarter scale, 1/2^2 = 1/4
        ref_proj = data[("proj_mat", 0, s)]
        src_projs = [data[("proj_mat", idx, s)] for idx in self.matching_ids[1:]]
        if (img_key, 0, 2) in data:
            ref_frame_quarter = data[img_key, 0, 2]
        else:
            ref_frame_quarter = F.interpolate(ref_frame, 
                            [img_height//4, img_width//4], 
                            mode="bilinear", 
                            align_corners=True)
        
        #print ("callinr run_depth in eval")
        pred_depth, prob_volume, refined_depth = self.run_depth(ref_frame, ref_proj, lookup_frames, 
                            src_projs, 
                            depth_values,
                            ref_frame_quarter, 
                            refine = self.refine)
        

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
