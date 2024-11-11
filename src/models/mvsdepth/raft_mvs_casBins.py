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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


""" load our own moduels """
from .mvs_base import autocast
from .basics import ProbMaskBlock
from .raft_mvs import RAFT_MVS as RAFT_MVS_Base
from .basics import FeatureSPP
from .update import BasicUpdateBlock
from src.models.model_utils import (
    updepth, 
    get_n_downsample, bilinear_sampler,
    coords_grid_normlized,
    
)
from src.utils.utils import check_nan_inf, is_empty
from src.utils.comm import print0
from src.loss_utils import dMap_to_indxMap


#_IS_DEBUG_= True
_IS_DEBUG_= False


"""
MVS + RAFT backbone 
+ cascaded depth bins, for multi-stage MVS plane-sweeping calculation;
we changed FlowHead in RAFT/RAFT-Stereo
to output delta 1D-flow (along frontal-parallel depth planes);
"""
class RAFT_MVS(RAFT_MVS_Base):
    def __init__(self, *args, **kwargs):
        # load parent model and initialization;
        super(RAFT_MVS, self).__init__(*args, **kwargs)
        self.model_card = "raft_mvs_casBins"
        if self.is_verbose:
            print0 (f"[***] {self.model_card} initialization done")

        assert isinstance (self.num_depth_bins, list), \
            f"Requires num_depth_bins as a list, e.g., [64,32], but got {self.num_depth_bins}"
        
        # --- reset ----
        self.adaptive_bins = True
        
        """features in which spatial scale are used to build cost volume at each stage;"""
        self.down_scale_int = None # disable it, use the feat_scales below;
        self.feat_scales = kwargs.get('feat_scales', [8,8]) # eighth scale
        
        # whether to share the modules among different stages
        self.share_module_stages = kwargs.get('share_module_stages', True)
        self.num_stage = kwargs.get('num_stage', 2)
        assert self.num_stage == 2, "Two stages!!!"
        
        #---------------------------------
        # newly added methods and variables
        #---------------------------------
        target_scales_dict = {
            16: 'sixteenth',
            8: 'eighth',
            4: 'quarter',
            2: 'half',
            1: 'full',
        }
        
        # added for cascade methods;
        self.depth_interval = 1.0*(self.opt_max_depth_bin - self.opt_min_depth_bin) / max(self.num_depth_bins, 192)
        self.depth_interals_ratio = kwargs.get('depth_interals_ratio', [3,2,1]) # eighth scale
        self.depth_sampling_func = {
            'inverse': self.get_depth_range_samples,
            'linear': self.get_inverse_depth_range_samples
            }[self.depth_binning]
        
        base_fmap_dim = 128 # feature_fusion output dim;
        hdim, cdim = 128, 128
        base_convx_upscale = 4

        base_corr_levels = self.corr_levels # 4
        base_corr_radius = self.corr_radius # 4

        """ all stages info, part or all of them will be used,
            depending on self.num_stage
        """
        self.stage_infos = {}
        for i in range(self.num_stage):
            cur_faet_scale = self.feat_scales[i]
            self.stage_infos[f'stage{i}'] = {
                "feat_scale": cur_faet_scale,
                'target_scale_str': target_scales_dict[cur_faet_scale], # use the same target scale;
                "ndepth": self.num_depth_bins[i],
                "fmap_dim": base_fmap_dim,
                'corr_levels': base_corr_levels, # 4
                'corr_radius': base_corr_radius, # 4
                'specify_convx_upscale': base_convx_upscale,
            }
        
        if self.is_verbose:
            print ("/////// self.stage_infos = ", self.stage_infos)
            print ("/////// self.target_scales_dict = ", self.target_scales_dict)

        fpn_output_channels = 32
        # spf: Spatial Pyramid feature Fusion
        self.spf = FeatureSPP(in_planes=4*fpn_output_channels, out_planes= base_fmap_dim)
        if self.share_module_stages:
            if self.is_verbose:
                print(f"Using shared stages, self.share_module_stages = {self.share_module_stages}")
            corr_levels = base_corr_levels
            corr_radius = base_corr_radius
            sepcify_convx_upscale = base_convx_upscale
            self.update_block = BasicUpdateBlock(
                    corr_levels, corr_radius, 
                    hidden_dim = hdim,
                    head_type ='depth',
                    volume_scale = None,
                    you_specify_upscale = sepcify_convx_upscale # if set, then disable "volume_scale"
                )
            self.prob_net = ProbMaskBlock(
                    down_scale_int= get_n_downsample(self.raft_volume_scale),
                    hidden_dim = hdim,
                    output_dim = self.num_depth_bins_up
                )
        else:
            for stage_idx in range(self.num_stage):
                corr_levels = self.stage_infos[f'stage{stage_idx}']['corr_levels']
                corr_radius = self.stage_infos[f'stage{stage_idx}']['corr_radius']
                target_scale_str = self.stage_infos[f'stage{stage_idx}']['target_scale_str']
                sepcify_convx_upscale = self.stage_infos[f'stage{stage_idx}']['specify_convx_upscale']
                setattr(self, 
                    f'stage{stage_idx}_update_block',
                    BasicUpdateBlock(
                        corr_levels, corr_radius, hidden_dim=hdim,
                        head_type='depth',
                        volume_scale = None,
                        you_sepcify_upscale = sepcify_convx_upscale # if set, then disable "volume_scale"
                    )
                )
                
                setattr(self, 
                    f'stage{stage_idx}_prob_net',
                    ProbMaskBlock(
                        down_scale_int= get_n_downsample(target_scale_str),
                        hidden_dim = hdim,
                        output_dim = self.num_depth_bins_up
                        )
                )
    
    
    # uniform sampling in the depth range
    def get_depth_range_samples(
                self, 
                cur_depth: torch.Tensor, 
                num_depth: int, 
                min_depth: torch.Tensor,
                max_depth: torch.Tensor,
                height: int,
                width: int,
                depth_interval_scale: float, 
                device: torch.device,  
                ) -> torch.Tensor:
        #cur_depth: (B, 1, H, W) or torch.empty
        #return depth_range_samples: (B,D) or (B, D, H, W)
        
        depth_interval_pixel = depth_interval_scale
        batch_size = min_depth.size()[0]
        min_depth = min_depth.view(batch_size, 1)
        max_depth = max_depth.view(batch_size, 1)
        
        if is_empty(cur_depth):
            new_interval = (max_depth - min_depth) / (num_depth - 1)  # (B, 1)

            tmp_bins = new_interval*torch.arange(0, num_depth, device=device,requires_grad=False).view(1, num_depth)
            depth_range_samples = min_depth + tmp_bins #(B, D)
            new_depth_min = repeat(min_depth, 'b 1 -> b 1 new_dim1 new_dim2', new_dim1 = height, new_dim2=width)
            new_depth_max = repeat(max_depth, 'b 1 -> b 1 new_dim1 new_dim2', new_dim1 = height, new_dim2=width)

        else:
            cur_depth_min = (cur_depth - num_depth / 2 * depth_interval_pixel)  # (B, 1, H, W)
            cur_depth_max = (cur_depth + num_depth / 2 * depth_interval_pixel)

            assert cur_depth.dim() == 4 and cur_depth.size(1)==1, "cur_depth: size = {}".format(cur_depth.shape)
            new_interval = (cur_depth_max - cur_depth_min)/(num_depth - 1)  # (B, 1, H, W)

            tmp_bins = rearrange(
                torch.arange(0, num_depth, device=cur_depth.device, dtype=cur_depth.dtype, requires_grad=False),
                'd -> 1 d 1 1')
            depth_range_samples = cur_depth_min + tmp_bins * new_interval #(B,D,H,W)
            new_depth_min = depth_range_samples[:,0:1,:,:] #[B,1,H,W]
            new_depth_max = depth_range_samples[:,-1:,:,:]

        return depth_range_samples, new_depth_min, new_depth_max
    
    # uniform sampling in the inverse depth range
    def get_inverse_depth_range_samples(
                self, 
                cur_depth: torch.Tensor, 
                num_depth: int, 
                min_depth: torch.Tensor,
                max_depth: torch.Tensor,
                height: int,
                width: int,
                depth_interval_scale: float, 
                device: torch.device, 
            ) -> torch.Tensor:
        
        #shape: (B, 1, H, W)
        #cur_depth: (B, 1, H, W) or (B, D)
        #min_depth: (B,1) 
        #max_depth: (B,1)
        #return depth_range_samples: (B,D) or (B, D, H, W)
        
        batch_size = min_depth.size()[0]
        inv_min_depth = (1.0 / min_depth).view(batch_size, 1)
        inv_max_depth = (1.0 / max_depth).view(batch_size, 1)
        
        if is_empty(cur_depth):
            # index: [0, D-1]
            # make 1/d in ascending order;
            index = torch.arange(num_depth-1, -1, step=-1, device=device)
            #index = torch.arange(0, num_depth, step=1, device=device)

            index = index.view(1, num_depth).float()
            normalized_index = index /(num_depth - 1)
            inverse_depth_sample = inv_max_depth + normalized_index*(inv_min_depth - inv_max_depth)
            depth_samples = 1.0 / inverse_depth_sample
            new_depth_min = repeat(min_depth, 
                                   'b 1 -> b 1 new_dim1 new_dim2', new_dim1 = height, new_dim2=width)
            new_depth_max = repeat(max_depth, 
                                   'b 1 -> b 1 new_dim1 new_dim2', new_dim1 = height, new_dim2=width)

 
        else:
            # uniform samples in an inversed depth range
            ## make 1/d in ascending order;
            index = torch.arange(num_depth//2-1, num_depth//2+1, step=-1, device=device)
            ## make 1/d in decending order;
            #index = torch.arange(-num_depth//2, num_depth//2, step=1, device=device)
            index = index.view(1, num_depth, 1, 1).float()
            
            inv_depth_interval = (inv_min_depth - inv_max_depth) * depth_interval_scale
            inv_depth_interval = inv_depth_interval.view(batch_size, 1, 1, 1)

            inv_depth_samples = 1.0/cur_depth.detach() + inv_depth_interval * index
            inv_depth_clamped = []
            for k in range(batch_size):
                inv_depth_clamped.append(
                    torch.clamp(inv_depth_samples[k], min=inv_max_depth[k], max=inv_min_depth[k]).unsqueeze(0)
                )
            depth_samples = 1.0 / torch.cat(inv_depth_clamped, dim=0)
            
            # have to make sure depth_samples are in ascending order;
            new_depth_min = depth_samples[:,0:1,:,:]
            new_depth_max = depth_samples[:,-1:,:,:]

        return depth_samples, new_depth_min, new_depth_max

    # run RAFT-backbone for MVS depth estimation;
    def run_raft_depth(self, ref_image, lookup_images,
                relative_poses,
                Ks_src_stages,
                invK_ref_stages,
                min_depth_bin,
                max_depth_bin,
                iters_list, # e.g., [12,12];
                is_verbose = False
                ):

        outputs = {}

        # 1 or more source images
        batch_size, num_frames, chns_img, height_img, width_img = lookup_images.shape
        lookup_images = rearrange(lookup_images, 'b v c h w -> (b v) c h w')

        #---- feature extraction --- #
        # run the feature network on reference image and source images
        ref_feat_dict = self.feature_extraction(ref_image)
        lookup_feat_dict = self.feature_extraction(lookup_images)
        
        # hidden state, context feature; 1/4 scale;
        net, inp = self.context_extraction(ref_image)

        spf_layer = getattr(self, 'spf')
        
        # adaptive bins low and upper bound
        #ada_bins_low, ada_bins_high = min_depth_bin, max_depth_bin
        # preprated for cascaded stages
        outputs[('depth_iters', 0)]  = []
        
        # depth from previous stage, initialized as None;
        depth_prev = None

        for stage_idx in range(self.num_stage):
            # raft iteration times 
            raft_iter = iters_list[stage_idx]
         
            cur_scale_str = self.stage_infos[f'stage{stage_idx}']['target_scale_str'] # e.g., 'quarter';
            cur_scale = int(self.stage_infos[f'stage{stage_idx}']['feat_scale']) # e.g., 4;
            ndepth = self.stage_infos[f'stage{stage_idx}']['ndepth']
            fmap_dim = self.stage_infos[f'stage{stage_idx}']['fmap_dim']
            corr_l = self.stage_infos[f'stage{stage_idx}']['corr_levels']
            corr_r = self.stage_infos[f'stage{stage_idx}']['corr_radius']
            cur_h, cur_w = height_img//cur_scale, width_img // cur_scale
            
            depth_interval = self.depth_interval*self.depth_interals_ratio[stage_idx]

            if 1 and is_verbose:
                print(f"@stage{stage_idx+1}/{self.num_stage}: raft_iter={raft_iter:>2}, " \
                      f"scale={cur_scale:>2}/{cur_scale_str:<10}, " \
                      f"dimen={cur_w:>3}x{cur_h:<3}, ndepth={ndepth:>2}, " \
                      f"depth_interval={depth_interval:.4f}, " \
                      f"feat_dim={fmap_dim:>3}, corr_l={corr_l}, corr_r={corr_r}"
                      )
            
            #---- feature fusion ---
            ref_feat = spf_layer(ref_feat_dict, target_scale= cur_scale_str)
            lookup_feat = spf_layer(lookup_feat_dict, target_scale= cur_scale_str)
            lookup_feat = rearrange(lookup_feat, '(b v) c h w -> b v c h w', v=num_frames)
            #in case in mixed_precision
            ref_feat = ref_feat.float()
            lookup_feat = lookup_feat.float()
            
            if depth_prev is None:
                #depth_bins = self.depth_bins
                #depth_cur = depth_bins.repeat(batch_size// depth_bins.size(0),1).to(
                #    ref_feat.device) #[N,D]
                # initialized as an empty Tensor;
                depth_cur = torch.empty(0, device= ref_image.device)  
            else:
                if depth_prev.shape[-2:] != (cur_w, cur_h):
                    depth_prev = F.interpolate(depth_prev, (cur_h, cur_w), mode='bilinear', align_corners=True)
                depth_cur = depth_prev

            # [B,D] or [B,D,H,W] 
            depth_bins, depth_min, depth_max = self.depth_sampling_func(
                            cur_depth = depth_cur,
                            num_depth = ndepth,
                            min_depth = min_depth_bin,
                            max_depth = max_depth_bin,
                            height = cur_h,
                            width = cur_w,
                            depth_interval_scale= depth_interval,
                            device = ref_feat.device,
                            )
            outputs[f'depth_min/stage{stage_idx}'] = depth_min
            outputs[f'depth_max/stage{stage_idx}'] = depth_max
             
            # plane-sweeping stereo
            # pose and K for this stage
            #if depth_bins.size(-1) == depth_bins.size(-2) == 1:
            #    # change dim for plane-sweeping;
            #    depth_bins = rearrange(depth_bins, "b d 1 1 -> b d")

            Ks_src = Ks_src_stages[cur_scale_str]
            invK_ref = invK_ref_stages[cur_scale_str]

            cost_volume, missing_mask = self.my_match_func(
                    depth_bins,
                    ref_feat,
                    lookup_feats = lookup_feat,
                    relative_poses = relative_poses,
                    Ks_src = Ks_src,
                    invK_ref = invK_ref,
                    offset_layer= None #no used;
                )
            
            cost_idx_init, softargmin_depth = self.get_raft_cost_idx_init(
                                                    cost_volume, depth_bins)
            outputs[f'softargmin_depth/stage{stage_idx}'] = softargmin_depth


            # resize the hidden and context feature to target_scale
            if net.shape[-2:] != (cur_h, cur_w):
                cur_net = F.interpolate(net, (cur_h, cur_w), mode='bilinear', align_corners=True)
            if inp.shape[-2:] != (cur_h, cur_w):
                cur_inp = F.interpolate(inp, (cur_h, cur_w), mode='bilinear', align_corners=True)
            
            # ---- main raft iteration ---#
            depth_up, prob_K = self.run_raft_depth_per_stage(
                    stage_idx,
                    cur_net, cur_inp,
                    # make index D in the last axis;
                    rearrange(cost_volume, 'b d h w -> b h w d').contiguous(),
                    cost_idx_init,
                    depth_bins,
                    raft_iter,
                    outputs
                    )
            # update for the next iteration
            depth_prev = depth_up 

        # done multi stages
        #check_dict_k_v(outputs, "outputs")
        #sys.exit()
        return outputs

    # run RAFT forward at one of the cascaded stages;
    def run_raft_depth_per_stage(self,
            stage_idx,
            net, # hidden
            inp, # context feature
            cost_volume, # plane sweeping cost volume, [N,H,W,D]
            cost_idx_init, # corresponding to depth bin index;
            depth_bins,
            iters,
            outputs
        ):
        ndepth = self.stage_infos[f'stage{stage_idx}']['ndepth']
        assert cost_volume.size(-1) == ndepth
        batch_size, feat_height, feat_width = cost_volume.size()[0:3]
        if _IS_DEBUG_: #TODO???
            check_nan_inf(inp = cost_volume, name="vost_volume")

        scale = int(self.stage_infos[f'stage{stage_idx}']['feat_scale'])
        base_convx_upscale = self.stage_infos[f'stage{stage_idx}']['specify_convx_upscale']
            

        corr_levels = self.stage_infos[f'stage{stage_idx}']['corr_levels']
        corr_radius = self.stage_infos[f'stage{stage_idx}']['corr_radius']

        # initial idx, [N,1,H,W]
        cost_idx_size = [batch_size, 1, feat_height,  feat_width]
        cost_idx1 = torch.zeros(cost_idx_size).to(cost_volume.device).float()
        #cost_idx0 = torch.zeros_like(cost_idx1)
        #print ("[???] cost_idx_init = ", cost_idx_init.shape)
        if cost_idx_init is not None:
            assert cost_idx_init.shape[1] == 1, "cost_idx_init in shape [N,1,H,W]"
            cost_idx1 = cost_idx1 + cost_idx_init.detach()

        # with depth_bins, not depth_bins_up;
        depth_init = self.indices_to_depth(cost_idx1, depth_bins)
        outputs[(f'depth/stage{stage_idx}/init', 0, 0)] = updepth(depth_init.detach(), scale=scale)
        #print ("???", depth_init)
        #sys.exit()

        # key = ('depth_iters', scale )
        # detach the first iteration
        if stage_idx == 0:
            outputs[('depth_iters', 0)].append(updepth(depth_init.detach(), scale=scale))

        #-----------------------
        # start RAFT iterations
        #-----------------------
        corr_fn = self.corr_block(cost_volume, num_levels=corr_levels, radius = corr_radius)

        if self.share_module_stages:
            cur_update_block = self.update_block
            cur_prob_net = self.prob_net
        else:
            #NOTE: Need conditional since TorchScript only allows "getattr" access 
            # with string literals
            if stage_idx == 0:
                cur_update_block = self.stage0_update_block
                cur_prob_net = self.stage0_prob_net
            elif stage_idx == 1:
                cur_update_block = self.stage1_update_block
                cur_prob_net = self.stage1_prob_net
            elif stage_idx == 2:
                cur_update_block = self.stage2_update_block
                cur_prob_net = self.stage2_prob_net
            else:
                raise NotImplementedError

        for itr in range(0, iters):
            cost_idx1 = cost_idx1.detach()
            corr = corr_fn(cost_idx1) # index correlation volume
            #if _IS_DEBUG_:
            #    check_nan_inf(inp = corr, name="corr")

            with autocast(enabled=self.is_mixed_precision):
                if self.is_multi_gru:
                    # to perform lookup from the corr and update flow
                    net, up_mask, delta_cost_idx = cur_update_block(
                                net, 
                                inp, 
                                corr, cost_idx1,
                                iter32 = self.n_gru_layers==3, 
                                iter16 = self.n_gru_layers>=2
                                )
                else: # single GRU
                    net, up_mask, delta_cost_idx = cur_update_block(
                                net, inp, corr, 
                                cost_idx1
                                )

            # F(t+1) = F(t) + \Delta(t)
            cost_idx1 = cost_idx1 + delta_cost_idx

            # We do not need to upsample or output intermediate results in test_mode
            if (not self.is_training) and itr < iters-1:
                continue

            if _IS_DEBUG_:
                check_nan_inf(inp = delta_cost_idx, name= "delta_cost_idx")


            """ upsample predictions """
            ##~~ use depth_bin_up
            flow1d_up = self.upsample_flow1d(cost_idx1, up_mask)
            # get new disparity
            hidden = net[0] if self.is_multi_gru else net
            prob_mask = cur_prob_net(hidden) #[N,H,W,D] 
            ##~~ use depth_bin_up
            depth_bins_up = self.depth_bins_up
            depth_bins_up = depth_bins_up.repeat(batch_size// depth_bins_up.size(0), 1).to(
                cost_volume.device)
            depth_up, prob_K = self.indices_to_depth_up_regression(
                flow1d_up, prob_mask, 
                depth_bins_up,
                is_biliear_index_prob_mask= False
                )

            outputs[('depth_iters', 0)].append(depth_up)
            # --- end of raft iteration

        # iter done
        outputs[(f"depth/stage{stage_idx}/raft", 0, 0)] = depth_up #Nx1xHxW
        if stage_idx == self.num_stage - 1:
            outputs[(f"depth", 0, 0)] = depth_up #Nx1xHxW
            
        # last stage we save photometric confidence
        if stage_idx == self.num_stage - 1:
            with torch.no_grad():
                # photometric confidence
                prob_volume_sum4 = 4*F.avg_pool3d(
                    F.pad(
                        rearrange(prob_mask, 'b d h w -> b 1 d h w'),
                        pad=(0, 0, 0, 0, 1, 2)),
                    kernel_size=(4, 1, 1),
                    stride=1, padding=0).squeeze(1)
                # soft-agrmin index;
                # index at last GRU iteration step;
                depth_index = flow1d_up.long()
                depth_index = depth_index.clamp(min=0, max=self.num_depth_bins_up-1)
                photometric_confidence = torch.gather(
                    prob_volume_sum4, dim=1, index=depth_index)
                outputs["confidence"] = photometric_confidence

        return depth_up, prob_K

    def get_raft_cost_idx_init(self, cost_volume, depth_bins):
        """ 
        Args: 
            cost_volume: [N,D,H,W]
            depth_bins: [N,D]
        """
        num_depth_bins = depth_bins.size(1)
        if depth_bins.dim() == 2:
            depth_bins = rearrange(depth_bins, 'b d -> b d 1 1')
        prob = F.softmax(cost_volume, dim=1)
        #print ("??? prob depth_bins: ", prob.shape, depth_bins.shape)
        softargmin_depth = reduce(prob*depth_bins, 'b d h w -> b 1 h w', 'sum')
        #print ("??? saving lowest_cost, softargmin_depth")
        if self.raft_depth_init_type == 'none':
            cost_idx_init = None
         
        elif self.raft_depth_init_type == 'soft-argmin':
            ## discritized index
            #cost_idx_init = dMap_to_indxMap(softargmin_depth, depth_bins) # long type
            ## soft-agrmin index;
            idx_values = torch.arange(num_depth_bins,
                                device=cost_volume.device,
                                dtype=torch.float).view(1,num_depth_bins,1,1)
            depth_idx = reduce(prob*idx_values, 'b d h w -> b 1 h w', 'sum')
            depth_idx = depth_idx.clamp(min=0, max=num_depth_bins-1)
            cost_idx_init = depth_idx
        else:
            raise NotImplementedError
        
        return cost_idx_init, softargmin_depth
    
    def indices_to_depth_regression(self,
                indices, #[N,1,H,W]
                prob_mask, #[N,D,H,W]
                depth_bins, #[N,D,H,W]
                is_biliear_index_prob_mask,
                # due to cuDNN error: a non-contiguous input in torch.grid_sampler()
                #divid_small_chunk_grid=True
                ):
        """
        Convert cost volume indices to depth by grid_sample
        Args:
           prob_mask: [N,D,H,W], here D=num_depth_bins_up;
        """
        indices = indices.permute(0, 2, 3, 1) #to [N,H,W,1]
        batch, h1, w1, _ = indices.shape

        # due to F.grid_sample only supports spatial (4-D) and volumetric (5-D) input;
        assert depth_bins.dim() == 4 and depth_bins.size(-1) == w1 and depth_bins.size(-2) == h1
        num_depth_bins = depth_bins.size(1)
        assert prob_mask.size(1) == num_depth_bins
        
        """ avoid min max, directly indexing the first or the last slice; """
        #depth_min = reduce(depth_bins, 'b d h w -> b 1 h w',reduction='min')
        #depth_max = reduce(depth_bins, 'b d h w -> b 1 h w',reduction='max')
        """ have to make sure depth_samples are in ascending order """
        depth_min = depth_bins[:,0:1,:,:] # [B,1,H,W]
        depth_max = depth_bins[:,-1:,:,:]

        # we consider a 1 by K neigibor,
        # with K=2*r+1, r is the radius;
        r = self.prob_radius
        K = 2*r + 1
        dx = torch.linspace(-r, r, 2*r+1).to(indices.device)
        # rearrange: use 1 to create a new axis of length 1;
        dx = rearrange(dx, 'k -> k 1 1 1 1')
        x0 = dx + rearrange(indices, 'b h w c -> 1 b h w c')
        x0 = dx + indices.reshape(1, batch, h1, w1, 1) # [K, N, H, W, 1]
        y0 = torch.zeros_like(x0)
        grids = torch.cat([x0,y0], dim=-1)# [K, N, H, W, 2]


        xy_norm = coords_grid_normlized(batch, h1, w1, indices.device) #[N,H,W,2]
        xy_K_norm = repeat(xy_norm, 'b h w c -> b new_axis h w c', new_axis=K)#[N,K,H,W,2]
        dz = torch.linspace(-r, r, 2*r+1).to(indices.device)
        dz = rearrange(dz, 'k -> 1 k 1 1 1')
        z_K = dz + indices.unsqueeze(1) # [N, K, H, W, 1]
        z_K_norm = 2*z_K / (num_depth_bins - 1) - 1.0 # normalized to range [-1, 1];

        grid_3d_norm = torch.cat([xy_K_norm, z_K_norm], dim=-1) #[N, K, H, W, 3]
        depth_K = F.grid_sample(
            rearrange(depth_bins, 'b d h w -> b () d h w'),
            grid_3d_norm,
            mode= 'bilinear',
            align_corners= True
            )
        depth_K = rearrange(depth_K, "b 1 k h w -> b k h w")
        #print ("???? depth_K = ", depth_K.shape)

        if is_biliear_index_prob_mask:
            #grid_sample: output will have shape [N, C=1, D_out=K, H_out=H, W_out=W]
            prob_K = F.grid_sample(
                rearrange(prob_mask, 'b d h w -> b () d h w'),
                grid_3d_norm,
                mode= 'bilinear',
                align_corners= True
                )
            prob_K = rearrange(prob_K, "b 1 k h w -> b k h w")

        else:
            # here we would like to "gather" the weights with integer "indices"
            # no grad needed to get the integer "indices",
            # since the "differentiable" thing has been gauranteed by the above
            # depth_K sampling;
            with torch.no_grad():
                index = torch.floor(indices).type(torch.float) #[N,H,W,1]
                index = rearrange(index, 'b h w c -> b c h w') #[N,1,H,W]
                dx = torch.linspace(-r, r, 2*r+1).view(1,K,1,1).to(indices.device) # [1, K, 1, 1]
                index = index + dx # [N, K, H, W]
                index = torch.clamp(index, min=0, max=num_depth_bins-1)
                index = index.type(torch.long)

            prob_K_list = []
            for k_idx in range(0, K):
                # torch.gather: input and index must have the same number of dimensions.
                prob_k = torch.gather(prob_mask, #[N,D,H,W]
                                    dim=1,
                                    index=index[:,k_idx:k_idx+1] #[N,1,H,W]
                                    )
                prob_K_list.append(prob_k)
            prob_K = torch.cat(prob_K_list, dim=1) #[N,K,H,W]

        # weighted summation for final depth;
        depth = torch.sum(depth_K*prob_K, dim=1, keepdim=True) / (1e-6+torch.sum(prob_K, dim=1, keepdim=True))
        depth = torch.clip(depth, min=depth_min, max=depth_max)
        #print ("???? prob_K = ", prob_K.shape, " depth = ", depth.shape)
        return depth, prob_K.detach()



    def indices_to_depth(self, indices, depth_bins, #[N,D] or [N,D,H,W], 2D/4D tensor
        ):
        """ Convert cost volume indices to depth by grid_sample"""
        indices = indices.permute(0, 2, 3, 1).contiguous() #to [N,H,W,1]
        num_depth_bins = depth_bins.size(1)
        batch, h1, w1, _ = indices.shape
        #print ("[???] indices shape = ", indices.shape)
        if depth_bins.dim() == 2:
            assert depth_bins.size(0) == batch
            """ avoid min max, directly indexing the first or the last slice; """
            #depth_min = reduce(depth_bins, 'b d -> b 1',reduction='min')
            #depth_min = repeat(depth_min, 'b 1 -> b 1 new_dim1 new_dim2', new_dim1 = h1, new_dim2=w1)
            #depth_max = reduce(depth_bins, 'b d -> b 1',reduction='max')
            #depth_max = repeat(depth_max, 'b 1 -> b 1 new_dim1 new_dim2', new_dim1 = h1, new_dim2=w1)

            """ have to make sure depth_samples are in ascending order """
            depth_min = depth_bins[:,0:1,] # [B,1,H,W]
            depth_min = repeat(depth_min, 'b 1 -> b 1 new_dim1 new_dim2', new_dim1 = h1, new_dim2=w1)
            depth_max = depth_bins[:,-1:]
            depth_max = repeat(depth_max, 'b 1 -> b 1 new_dim1 new_dim2', new_dim1 = h1, new_dim2=w1)
            
            # due to F.grid_sample only supports spatial (4-D) and volumetric (5-D) input;
            depth_bins = rearrange(depth_bins, 'b d -> b 1 1 d')

            # grid: [N, H_out, W_out, 2]
            # here H_out=1, W_out=1;
            x0 = indices # [batch, h1, w1, 1]
            y0 = torch.zeros_like(x0)
            grid = torch.cat([x0,y0], dim=-1)# [N, H, W, 2]

            #grid_sample: output will have shape [N, C, H_out, W_out]
            depth = bilinear_sampler(depth_bins, grid) #[N, C=1, h1, w1]

        elif depth_bins.dim() == 4:
            assert depth_bins.size(-1) == w1 and depth_bins.size(-2) == h1
            # 3D grid_sampling volumetric (5-D) input;
            xy_norm = coords_grid_normlized(batch, h1, w1, indices.device) #[N,H,W,2]
            z_norm = 2*indices/(num_depth_bins - 1) - 1.0 # normalized to range [-1, 1];
            # grid: [N, H_out, W_out, 2]
            # here H_out=1, W_out=1;
            grid_3d_norm = torch.cat([xy_norm, z_norm], dim=-1) #[N,H,W,3]
            grid_3d_norm = grid_3d_norm.unsqueeze(dim=1) #[N,1,H,W,3]

            #grid_sample: output will have shape [N, C, Z_out, H_out, W_out]
            depth = F.grid_sample(
                rearrange(depth_bins, 'b d h w -> b 1 d h w'),
                grid_3d_norm,
                mode= 'bilinear',
                align_corners= True
                ) #[N,C=1,Z_out=1,H,W]=[N,1,1,H,W]
            depth = depth.squeeze(1) #[N,1,H,W]
            """ avoid min max, directly indexing the first or the last slice; """
            #depth_min = reduce(depth_bins, 'b d h w -> b 1 h w',reduction='min')
            #depth_max = reduce(depth_bins, 'b d h w -> b 1 h w',reduction='max')
            """ have to make sure depth_samples are in ascending order """
            depth_min = depth_bins[:,0:1,:,:] # [B,1,H,W]
            depth_max = depth_bins[:,-1:,:,:]

        else:
            raise NotImplementedError

        # assuming max-depth < 100 meters, o.w. change the max=1000 to other values;
        #depth = torch.clip(depth, min=0.001, max=1000) # in meters

        # dynamic bins, so we cannot use constant min/max values;
        depth = torch.clip(depth, min=depth_min, max=depth_max)

        return depth



    def upsample_flow1d(self, down_scale_int, flow1d, mask,
                        ndepth_scale=1.0, # scale to reflect the num_bins 'magnitude'
                                          # change before and after the upsample,
                                          # num_bins_old = 64, num_bins_new = 128;
                                          # then ndepth_scale=128/64=2;
                        channel_last=False):

        if channel_last:
            flow1d = rearrange(flow1d, 'b h w 1 -> b 1 h w').contiguous()#[N,1,H,W]

        """ Upsample depth [H/8, W/8, 1] -> [H, W, 1] using convex combination """
        #print ("[???] calling upsample_depth()")
        N, _, H, W = flow1d.shape # [N, 1, H, W]
        mask = mask.view(N, 9, down_scale_int, down_scale_int, H, W)
        mask = torch.softmax(mask, dim=1)
        up_flow1d = F.unfold(ndepth_scale*flow1d, [3,3], padding=1)
        up_flow1d = up_flow1d.view(N, 9, 1, 1, H, W)

        up_flow1d = torch.sum(mask*up_flow1d, dim=1) # [N, 9, 8, 8, H, W] -> [N, 8, 8, H, W]
        #up_flow1d = up_flow1d.permute(0, 3, 1, 4, 2) #[N, 8, 8, H, W] -> [N, H, 8, W, 8]
        #up_flow1d = up_flow1d.reshape(N, 1, down_scale_int*H, down_scale_int*W)
        ## use rearrange for clean code:
        up_flow1d = rearrange(up_flow1d, 'b k1 k2 h w -> b 1 (h k1) (w k2)')
        return up_flow1d


    def forward(self, current_image, lookup_images,
                relative_poses,
                Ks_src_stages,
                invK_ref_stages,
                min_depth_bin,
                max_depth_bin,
                iters_stages = [12,12],
                is_verbose = False
                ):

        outputs = self.run_raft_depth(
                current_image,
                lookup_images,
                relative_poses,
                Ks_src_stages, invK_ref_stages,
                min_depth_bin,
                max_depth_bin,
                iters_stages,
                is_verbose
                )

        return outputs
