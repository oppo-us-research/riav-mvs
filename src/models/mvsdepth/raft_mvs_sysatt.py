"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import os
import sys

""" load our own moduels """
from .mvs_base import autocast
from .raft_mvs import RAFT_MVS as RAFT_MVS_Base
from src.models.model_utils import (
    updepth, upflow, 
    get_n_downsample, bilinear_sampler
)
from src.utils.utils import check_nan_inf
from src.utils.comm import print0
from .basics import FeatureSPP, offset_layer

from src.models.mvsdepth.att_gma import (
    Attention, FeatureAggregator
)
from src.layers import match_features_fst

#_IS_DEBUG_= True
_IS_DEBUG_= False

"""
MVS + RAFT backbone 
+ apply symetric attention to the convolution features of 
  reference and source images, i.e., self-attention applied to 
  feature f1 and f2, respectively.
  This architecture is named as f1&2-attention for short;
  It is not used in our final design, listed here for ablation study only;
we changed FlowHead in RAFT/RAFT-Stereo
to output delta 1D-flow (along frontal-parallel depth planes);
"""
class RAFT_MVS(RAFT_MVS_Base):
    def __init__(self, *args, **kwargs):
        # load parent model and initialization;
        super(RAFT_MVS, self).__init__(*args, **kwargs)
        self.model_card = "raft_mvs_f1&2_atten"
        if self.is_verbose:
            print0 (f"[***] {self.model_card} initialization done")
        
        # reset
        self.cost_agg = None
        match_kwargs = {
                'is_training': self.is_training,
                'is_dot_product': self.is_dot_product, # dot product, not L1 distance;
                'set_missing_to_max' : False, # Do not set missing cost volume to its max_values
                'is_edge_mask': False, # do not consider pixels warped out of boundary;
                'is_max_corr_pixel_view': self.is_max_corr_pixel_view,
                # f1-attention;
                'scale_same_to_transformer': True, # normalization, used as in Transformer;
                }
        
        self.my_match_func = match_features_fst(**match_kwargs)
        if self.is_verbose:
            print0 ('[***] using layer match_features_fst()')
        
        #------ hyper-parameters ----
        #base_convx_upscale = 4
        # spf: Spatial Pyramid feature Fusion
        fpn_output_channels = 32
        self.fmap_dim = fmap_dim = 128 # feature_fusion output dim;
        self.spf = FeatureSPP(in_planes=4*fpn_output_channels, out_planes= fmap_dim)

        #---------------------------------
        # newly added methods and variables
        #--------------------------------- 
        self.atten_num_heads = kwargs.get('atten_num_heads', 4)

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
        
        # f1_att and f2_att (attention layer) are shared.
        self.f2_att = self.f1_att
        self.f2_aggregator = self.f1_aggregator
        
        
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
                min_depth_bin, max_depth_bin,
                iters=12
                ):

        outputs = {}
        if self.adaptive_bins:
            assert min_depth_bin is not None and max_depth_bin is not None, \
                "adaptive_bins=True, requires non-None inputs"
            #print0 (f"??? adaptive min_depth_bin={min_depth_bin}, max_depth_bin={max_depth_bin}") 
            self.depth_bins = self.compute_depth_bins(min_depth_bin, max_depth_bin, self.num_depth_bins)
            self.depth_bins_up = self.compute_depth_bins(
                min_depth_bin, max_depth_bin, self.num_depth_bins_up)

        # 1 or more source images
        batch_size, num_frames, chns_img, height_img, width_img = lookup_images.shape
        lookup_images = rearrange(lookup_images, 'b v c h w -> (b v) c h w')
        
        #---- feature extraction --- #
        # run the feature network on reference image and source images
        ref_feat_dict = self.feature_extraction(current_image)
        lookup_feat_dict = self.feature_extraction(lookup_images)

        # hidden state, context feature; 1/4 scale;
        net, inp = self.context_extraction(current_image)
        #---- feature fusion 1/4 scale --- #
        #spf_layer
        ref_feat = self.spf(ref_feat_dict, target_scale=self.raft_volume_scale)
        lookup_feat = self.spf(lookup_feat_dict, target_scale=self.raft_volume_scale)
        _, chns, feat_height, feat_width = lookup_feat.shape
        lookup_feat = rearrange(lookup_feat, '(b v) c h w -> b v c h w', v=num_frames)
        
        #in case in mixed_precision
        ref_feat = ref_feat.float()
        lookup_feat = lookup_feat.float()
         
        depth_bins = self.depth_bins
        depth_bins = depth_bins.repeat(batch_size// depth_bins.size(0), 1).to(
            ref_feat.device)
        
        # attention mechanism to ref feature
        attention = self.f1_att(ref_feat)
        ref_feat_global = self.f1_aggregator(attention, ref_feat)
        
        # attention mechanism to each source feature
        lookup_feat_global = [] 
        for v in range(num_frames):
            src_feat = lookup_feat[:, v].contiguous()
            attention = self.f2_att(src_feat)
            lookup_feat_global.append(
                self.f2_aggregator(attention, src_feat))
        lookup_feat_global = torch.stack(lookup_feat_global, dim=1)
        #print0 ("??? lookup_feat_global shape = ", lookup_feat_global.shape)
        
        cost_volume, missing_mask = self.my_match_func(
            depth_bins,
            ref_feat_global,
            lookup_feats = lookup_feat_global,
            relative_poses = relative_poses,
            Ks_src = Ks_src,
            invK_ref = invK_ref,
            )
         
        cost_idx_init, softargmin_depth = self.get_raft_cost_idx_init(
                                            cost_volume, depth_bins)
        outputs['softargmin_depth'] = softargmin_depth
        
        # make index D in the last axis;
        cost_volume = cost_volume.permute((0,2,3,1)).contiguous() #[N,H,W,D]
        
        if _IS_DEBUG_:
            check_nan_inf(inp = cost_volume, name="vost_volume")
        
        
        # initial idx, [N,1,H,W]
        cost_idx_size = [batch_size, 1, feat_height,  feat_width]
        cost_idx1 = torch.zeros(cost_idx_size).to(ref_feat.device).float()
        cost_idx0 = torch.zeros_like(cost_idx1)
        if cost_idx_init is not None:
            assert cost_idx_init.shape[1] == 1, "cost_idx_init in shape [N,1,H,W]"
            cost_idx1 = cost_idx1 + cost_idx_init
        
        # with depth_bins, not depth_bins_up;
        depth_init = self.indices_to_depth(cost_idx1, depth_bins)
        
        # key = ('depth_iters', scale )
        # detach the first iteration
        outputs[('depth_iters', 0)] = [updepth(depth_init, scale=self.down_scale_int).detach()]

        if _IS_DEBUG_:
            check_nan_inf(inp = relative_poses, name="relative_poses")
        
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
            #if _IS_DEBUG_:
            #    check_nan_inf(inp = corr, name="corr")
            
            with autocast(enabled=self.is_mixed_precision):
                if self.is_multi_gru:
                    ## Do no use slow_fast gru: removed the slow_fast_gru arg;

                    # to perform lookup from the corr and update flow
                    net, up_mask, delta_flow_1d = self.update_block(
                                net, 
                                inp, 
                                corr, flow_1d,
                                iter32 = self.n_gru_layers==3, 
                                iter16 = self.n_gru_layers>=2
                                )
                else: # single GRU
                    net, up_mask, delta_flow_1d = self.update_block(
                                net, inp, corr, 
                                flow_1d
                                )

            # F(t+1) = F(t) + \Delta(t)
            cost_idx1 = cost_idx1 + delta_flow_1d
            
            # We do not need to upsample or output intermediate results in test_mode
            if (not self.is_training) and itr < iters-1:
                continue

            if _IS_DEBUG_:
                check_nan_inf(inp = delta_flow_1d, name= "delta_flow_1d")
            
            """ upsample predictions """
            flow1d_up = self.upsample_flow1d(cost_idx1-cost_idx0, up_mask)
            
            if self.is_gt_flow_1D:
                outputs[('flow1d_iters', 0)].append(flow1d_up)

            hidden = net[0] if self.is_multi_gru else net
            prob_mask = self.prob_net(hidden) #[N,H,W,D]
            ##~~ use depth_bin_up
            depth_bins_up = self.depth_bins_up
            depth_bins_up = depth_bins_up.repeat(batch_size// depth_bins_up.size(0), 1).to(
                ref_feat.device)
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
        # iter done
        outputs[("depth", 0, 0)] = depth_up #Nx1xHxW
        
        # at last we save photometric confidence
        with torch.no_grad():
            # photometric confidence
            prob_volume = rearrange(prob_mask, 'b h w d -> b 1 d h w')
            prob_volume_sum4 = 4*F.avg_pool3d(
                F.pad(prob_volume, pad=(0, 0, 0, 0, 1, 2)),
                kernel_size=(4, 1, 1),
                stride=1, padding=0).squeeze(1)
            # index at last GRU iteration step;
            depth_index = flow1d_up.long() 
            depth_index = depth_index.clamp(min=0, max=self.num_depth_bins_up-1)
            photometric_confidence = torch.gather(
                prob_volume_sum4, dim=1, index=depth_index)
            outputs["confidence"] = photometric_confidence
        return outputs

    
    
    def forward(self, current_image, lookup_images, 
                relative_poses, Ks_src, invK_ref,
                min_depth_bin,
                max_depth_bin,
                iters=12,
                # dummy args
                save_corr_hidden = False, 
                freeze_fnet_cnet = False # for warmup training for several iterations
                ):

        outputs = self.run_raft_depth(
            current_image,
            lookup_images,
            relative_poses,
            Ks_src, invK_ref,
            min_depth_bin,
            max_depth_bin,
            iters
            )
        
        return outputs
