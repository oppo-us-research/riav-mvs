"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

""" load our own moduels """
from .raft_mvs import RAFT_MVS as RAFT_MVS_Base
from .mvs_base import autocast
from src.models.model_utils import (
    updepth, upflow,
    get_n_downsample, bilinear_sampler
    )
from src.utils.utils import check_nan_inf
from src.utils.comm import print0

from src.models.mvsdepth.att_gma import (
    Attention, GMAUpdateBlock, 
    GMAMultiUpdateBlock
)

#_IS_DEBUG_= True
_IS_DEBUG_= False


"""
MVS + RAFT backbone + GMA(i.e., attention on context feature): 
we changed FlowHead in RAFT/RAFT-Stereo
to output delta 1D-flow (along frontal-parallel depth planes);
"""
class RAFT_MVS(RAFT_MVS_Base):
    def __init__(self, *args, **kwargs):
        # load parent model and initialization;
        super(RAFT_MVS, self).__init__(*args, **kwargs)
        self.model_card = "raft_mvs_gma"
        if self.is_verbose:
            print0 (f"[***] {self.model_card} initialization done")
        
        #---------------------------------
        # newly added methods and variables
        #--------------------------------- 
        self.atten_num_heads = kwargs.get('atten_num_heads', 1)
        encoder_output_dim = 128
        self.att = Attention(
                        dim=encoder_output_dim, 
                        heads=self.atten_num_heads,
                        max_pos_size=160, 
                        dim_head= self.hidden_dims[2] if self.is_multi_gru else self.hidden_dim, 
                        position_type = 'position_and_content'
                        )
        
        if self.is_multi_gru:
            self.update_block = GMAMultiUpdateBlock(
                self.corr_levels, self.corr_radius, 
                self.hidden_dims,
                head_type= 'depth',
                volume_scale = self.raft_volume_scale,
                n_gru_layers = self.n_gru_layers, # number of hidden GRU levels;
                num_heads = self.atten_num_heads
                )
            self.context_extraction = self.context_extraction_multi_gru
        else:
            self.update_block = GMAUpdateBlock(
                self.corr_levels, self.corr_radius, self.hidden_dim,
                head_type='depth',
                volume_scale = self.raft_volume_scale,
                num_heads = self.atten_num_heads
                )

            self.context_extraction = self.context_extraction_single_gru
        
        
        if self.is_verbose:
            print0 ("////// _IS_DEBUG_ =", _IS_DEBUG_)
        
        # gma
        gma_weights_path = kwargs.get("gma_weights_path", None)
        if gma_weights_path and self.is_training:
            self.load_pretrained_gma(gma_weights_path)
        
        #---------------------------------
        #----------- init done -----------
        #---------------------------------
        
        

    # run RAFT-backbone for MVS depth estimation;
    def run_raft_depth(self, current_image, lookup_images, 
                relative_poses, Ks_src, invK_ref,
                min_depth_bin=None, max_depth_bin=None,
                iters=12,
                save_corr_hidden = False, # save this tensor for student
                freeze_fnet_cnet = False, # for warmup training for several iterations
                ):

        outputs = {}
        if self.adaptive_bins:
            assert min_depth_bin is not None and max_depth_bin is not None, \
                "adaptive_bins=True, requires non-None inputs"
            self.depth_bins = self.compute_depth_bins(min_depth_bin, max_depth_bin, self.num_depth_bins)
            self.depth_bins_up = self.compute_depth_bins(
                min_depth_bin, max_depth_bin, self.num_depth_bins_up)

        # 1 or more source images
        batch_size, num_frames, chns_img, height_img, width_img = lookup_images.shape
        lookup_images = lookup_images.reshape(batch_size * num_frames, chns_img, height_img, width_img)
        
        # run the feature network on reference image and source images
        if freeze_fnet_cnet:
            with torch.no_grad():
                #print ("freezeing fnet and cnet ...")
                current_feat = self.feature_extraction(current_image)
                lookup_feat = self.feature_extraction(lookup_images)
                if self.is_multi_gru:
                    # hidden state, context feature, context feature raw;
                    net, inp, inp_raw = self.context_extraction(current_image, return_raw_context_inp=True)
                else:
                    # hidden state, context feature;
                    net, inp = self.context_extraction(current_image)
                
                #---- feature fusion --- 
                if self.feature_fusion is not None:
                    current_feat = self.feature_fusion(current_feat,target_scale=self.raft_volume_scale)
                    lookup_feat = self.feature_fusion(lookup_feat,target_scale=self.raft_volume_scale)
                else:
                    # without fusion: pairnet return a dict;
                    if isinstance (current_feat, dict):
                        current_feat = current_feat[self.raft_volume_scale]
                        lookup_feat = lookup_feat[self.raft_volume_scale]
        else:
            current_feat = self.feature_extraction(current_image)
            lookup_feat = self.feature_extraction(lookup_images)
            if self.is_multi_gru:
                # hidden state, context feature, context feature raw;
                net, inp, inp_raw = self.context_extraction(current_image, return_raw_context_inp=True)
            else:
                net, inp = self.context_extraction(current_image)# hidden state, context feature;
            
            #---- feature fusion --- 
            if self.feature_fusion is not None:
                current_feat = self.feature_fusion(current_feat,target_scale=self.raft_volume_scale)
                lookup_feat = self.feature_fusion(lookup_feat,target_scale=self.raft_volume_scale)
            else:
                # without fusion: pairnet return a dict;
                if isinstance (current_feat, dict):
                    current_feat = current_feat[self.raft_volume_scale]
                    lookup_feat = lookup_feat[self.raft_volume_scale]
         

        _, chns, feat_height, feat_width = lookup_feat.shape
        lookup_feat = lookup_feat.reshape(batch_size, num_frames, chns, feat_height, feat_width)
        
        ##in case in mixed_precision
        if self.is_mixed_precision:
            # retain the f32;
            current_feat = current_feat.float()
            lookup_feat = lookup_feat.float()
        
        depth_bins = self.depth_bins
        depth_bins = depth_bins.repeat(batch_size// depth_bins.size(0), 1).to(
            current_feat.device)
        
        cost_volume, missing_mask = self.my_match_func(
            depth_bins,
            current_feat,
            lookup_feats = lookup_feat,
            relative_poses = relative_poses,
            Ks_src = Ks_src,
            invK_ref = invK_ref,
            )
         
        # ------ cost aggregation & raft depth_init -----
        if self.cost_agg is not None:
            if self.raft_depth_init_type == 'soft-argmin-3dcnn':
                cost_volume = self.cost_agg(cost_volume.unsqueeze(1))
                cost_volume = cost_volume.squeeze(1)
            elif self.raft_depth_init_type == 'soft-argmin-2dcnn':
                cost_volume = self.cost_agg(cost_volume)
        
        cost_idx_init = self.get_raft_cost_idx_init(cost_volume, outputs, depth_bins)
        
        # make index D in the last axis;
        cost_volume = cost_volume.permute((0,2,3,1)).contiguous() #[N,H,W,D]
        
        if _IS_DEBUG_: #TODO???
            check_nan_inf(inp = cost_volume, name="vost_volume")
        
        
        # initial idx, [N,1,H,W]
        cost_idx_size = [batch_size, 1, feat_height,  feat_width]
        cost_idx1 = torch.zeros(cost_idx_size).to(current_feat.device).float()
        cost_idx0 = torch.zeros_like(cost_idx1)
        #print ("[???] cost_idx_init = ", cost_idx_init.shape)
        if cost_idx_init is not None:
            assert cost_idx_init.shape[1] == 1, "cost_idx_init in shape [N,1,H,W]"
            cost_idx_tmp = F.interpolate(
                    cost_idx_init.detach(), 
                    [feat_height, feat_width],
                    mode="bilinear",
                    align_corners=True
                    ) #[N,1,H,W]
            cost_idx1 = cost_idx1 + cost_idx_tmp
        
        # with depth_bins, not depth_bins_up;
        depth_init = self.indices_to_depth(cost_idx1, depth_bins)
        
        # key = ('depth_iters', scale )
        # detach the first iteration
        outputs[('depth_iters', 0)] = [updepth(depth_init, scale=self.down_scale_int).detach()]
        if self.is_gt_flow_1D:
            # key = ('flow1d_iters', frame_id )
            outputs[('flow1d_iters', 0)] = [ 
                upflow(cost_idx1, 
                    scale=self.down_scale_int, 
                    value_scale = 1.0*self.num_depth_bins_up / self.num_depth_bins).detach() 
                ] #changed to [N,1,H,W]

        
        #if _IS_DEBUG_:
        #    check_nan_inf(inp = relative_poses, name="relative_poses")
        
        if self.is_multi_gru:
            #[out_8, out_16, out_21]
            # to index 0: for out_8
            attention = self.att(inp_raw[0]) 
        else:
            attention = self.att(inp)
        
        #-----------------------
        # start RAFT iterations
        #-----------------------
        """ check initilization, get correlation volume at begining,
            to avoid repeated computation during RAFT iteration;
        """
        corr_fn = self.corr_block(cost_volume, num_levels=self.corr_levels, \
                                radius = self.corr_radius)
        for itr in range(0, iters):
            cost_idx1 = cost_idx1.detach()
            flow_1d = cost_idx1 - cost_idx0
            corr = corr_fn(cost_idx1) # index correlation volume
            
            with autocast(enabled=self.is_mixed_precision):
                if self.is_multi_gru:
                    ## Do no use slow_fast gru: removed the slow_fast_gru arg;
                    # to perform lookup from the corr and update flow
                    net, up_mask, delta_flow_1d = self.update_block(
                                net, 
                                inp, 
                                corr, flow_1d, attention,
                                iter32 = self.n_gru_layers==3, 
                                iter16 = self.n_gru_layers>=2
                                )
                else: # single GRU
                    net, up_mask, delta_flow_1d = self.update_block(
                                net, inp, corr, 
                                flow_1d, attention
                                )

            # F(t+1) = F(t) + \Delta(t)
            cost_idx1 = cost_idx1 + delta_flow_1d
            
            # We do not need to upsample or output intermediate results in test_mode
            if (not self.is_training) and itr < iters-1:
                continue

            if _IS_DEBUG_:
                check_nan_inf(inp = delta_flow_1d, name= "delta_flow_1d")
            
            """ upsample predictions """
            ##~~ use depth_bin_up
            flow1d_up = self.upsample_flow1d(cost_idx1-cost_idx0, up_mask)
            
            if self.is_gt_flow_1D:
                outputs[('flow1d_iters', 0)].append(flow1d_up)

            hidden = net[0] if self.is_multi_gru else net
            prob_mask = self.prob_net(hidden) #[N,H,W,D]
            ##~~ use depth_bin_up
            depth_bins_up = self.depth_bins_up
            depth_bins_up = depth_bins_up.repeat(batch_size// depth_bins_up.size(0), 1).to(
                current_feat.device)
            depth_up, prob_K = self.indices_to_depth_up_regression(
                flow1d_up, prob_mask, 
                depth_bins_up,
                is_biliear_index_prob_mask= False
                )
            outputs[('depth_iters', 0)].append(depth_up)
            # --- end of raft iteration
        
        if self.is_gt_flow_1D:
            # key = ('flow', frame_id )
            with torch.no_grad(): # just for visualization
                outputs[('flow1d', 0)] = outputs[('flow1d_iters', 0)][-1] # [N, 1, H, W]

        # save for refine if needed;
        if save_corr_hidden:
            outputs['corr'] = corr
            outputs['hidden'] = hidden
            outputs['context'] = inp[0] if self.is_multi_gru else inp
        
        # iter done
        outputs[("depth", 0, 0)] = depth_up #Nx1xHxW
        outputs["confidence"] = torch.sum(prob_K, dim=1, keepdim=True)
        return outputs

    
    def forward(self, current_image, lookup_images, 
                relative_poses, Ks_src, invK_ref,
                min_depth_bin,
                max_depth_bin,
                iters=12,
                save_corr_hidden = False,
                freeze_fnet_cnet = False # for warmup training for several iterations
                ):

        if self.freeze_raft_net:
            with torch.no_grad():
                outputs = self.run_raft_depth(
                    current_image,
                    lookup_images,
                    relative_poses,
                    Ks_src, invK_ref,
                    min_depth_bin,
                    max_depth_bin,
                    iters,
                    save_corr_hidden = save_corr_hidden,
                    freeze_fnet_cnet = freeze_fnet_cnet,
                    )
        else:
            outputs = self.run_raft_depth(
                current_image,
                lookup_images,
                relative_poses,
                Ks_src, invK_ref,
                min_depth_bin,
                max_depth_bin,
                iters,
                save_corr_hidden = save_corr_hidden,
                freeze_fnet_cnet = freeze_fnet_cnet,
                )
        
        return outputs
