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
from .mvs_base import MVSDepth, autocast
from .basics import ProbMaskBlock
from src.models.model_utils import (
    updepth, upflow, 
    get_n_downsample, bilinear_sampler,
    coords_grid_normlized
)
from src.utils.utils import check_nan_inf
from src.utils.comm import print0

_IS_DEBUG_= False


"""
MVS + RAFT backbone: we changed FlowHead in RAFT/RAFT-Stereo
to output delta 1D-flow (along frontal-parallel depth planes);
"""
class RAFT_MVS(MVSDepth):
    def __init__(self, *args, **kwargs):
        # load parent model and initialization;
        super(RAFT_MVS, self).__init__(*args, **kwargs)
        self.model_card = "raft_mvs"
        if self.is_verbose:
            print0 (f"[***] {self.model_card} initialization done")
        
        #---------------------------------
        # newly added methods and variables
        #---------------------------------
        self.refine_net_type = kwargs.get('refine_net_type', 'none')
        if self.freeze_raft_net:
            assert self.refine_net_type != 'none', "Need refinenet, cannot be None!"
        if self.refine_net_type == 'none' or self.refine_net_type is None:
            if self.is_verbose:
                print0 ("No refine net")
            self.rfnet = None
        else:
            raise NotImplementedError

        # for different feature nets
        if self.fnet_name == 'raft_fnet':          
            self.feature_extraction = self.feature_extraction_raft_fnet
            # e.g., == 'pretrained_model_KITTI2015.tar'
            raft_weights_path = kwargs.get("raft_weights_path", None) 
            
            if raft_weights_path != '' and raft_weights_path is not None:
                assert self.is_training, "Only for training, we load pretrained raft encoder."\
                    "\nOtherwise, load weights from your own checkpoint instead!!!"
                self.load_pretrained_raft_encoder(raft_weights_path)
        
        elif self.fnet_name == 'pairnet_fnet':
            pretrain_dvmvs_pairnet_dir = kwargs.get('pretrain_dvmvs_pairnet_dir', None)
            self.feature_extraction = self.feature_extraction_pairnet
            
            if pretrain_dvmvs_pairnet_dir != '' and pretrain_dvmvs_pairnet_dir is not None:
                assert self.is_training, "Only for training, we load pretrained pairnet encoder."\
                    "\nOtherwise, load weights from your own checkpoint instead!!!"
                self.load_pretrained_pairnet_ckpt(pretrain_dvmvs_pairnet_dir)
        
        else:
            raise NotImplementedError
        
        if self.is_multi_gru:
            self.context_extraction = self.context_extraction_multi_gru
        else:
            self.context_extraction = self.context_extraction_single_gru
        
        # now we just want to disable it;
        assert self.slow_fast_gru == False, "Now we require self.slow_fast_gru == False"
        
        if self.is_verbose:
            print0 (f"////// _IS_DEBUG_ = {_IS_DEBUG_}")
        
        #------------------------
        # probability module
        #------------------------
        hidden_dim = self.hidden_dims[2] if self.is_multi_gru else self.hidden_dim
        self.prob_radius = kwargs.get('prob_radius', 4)

        self.prob_net = ProbMaskBlock(
                        down_scale_int= get_n_downsample(self.raft_volume_scale),
                        hidden_dim = hidden_dim,
                        output_dim = self.num_depth_bins_up
                        )
        
    # run RAFT-backbone for MVS depth estimation;
    def run_raft_depth(self, current_image, lookup_images, 
                relative_poses, Ks_src, invK_ref,
                min_depth_bin, max_depth_bin,
                iters=12,
                save_corr_hidden = False, # save this tensor for student
                freeze_fnet_cnet = False, # for several iterations of warmup training
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
                current_feat = self.feature_extraction(current_image)
                lookup_feat = self.feature_extraction(lookup_images)
                net, inp = self.context_extraction(current_image)# hidden state, context feature;
                
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
            net, inp = self.context_extraction(current_image)# hidden state, context feature;
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
        #print ("??? ", batch_size// depth_bins.size(0), depth_bins.shape)
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

        cost_idx_init = self.get_raft_cost_idx_init_hard(cost_volume, outputs, depth_bins) 

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
         
        #-----------------------
        # start RAFT iterations
        #-----------------------
        """ check initilization, get correlation volume at begining,
            to avoid repeated computation during RAFT iteration;
        """
        corr_fn = self.corr_block(cost_volume, 
                                  num_levels=self.corr_levels, 
                                  radius = self.corr_radius)
        for itr in range(0, iters):
            cost_idx1 = cost_idx1.detach()
            if _IS_DEBUG_:
                check_nan_inf(inp = cost_idx1, name="cost_idx1")
            
            flow_1d = cost_idx1 - cost_idx0
            corr = corr_fn(cost_idx1) # index correlation volume
            #if _IS_DEBUG_:
            #    check_nan_inf(inp = corr, name="corr")
            
            with autocast(enabled=self.is_mixed_precision):
                if self.is_multi_gru:
                    # slow-fast GRU: update the 1/16 and 1/32 resolution hidden
                    # states several times for every single update to the 1/8 
                    # resolution hidden state
                    if self.n_gru_layers == 3 and self.slow_fast_gru: 
                        # Update low-res GRU
                        net = self.update_block(net, inp, iter32=True, iter16=False, 
                                            iter08=False, update=False)
                    if self.n_gru_layers >= 2 and self.slow_fast_gru: 
                        # Update low-res GRU and mid-res GRU
                        net = self.update_block(net, inp, iter32=self.n_gru_layers==3, 
                                            iter16=True, iter08=False, update=False)
                    
                    # to perform lookup from the corr and update flow
                    net, up_mask, delta_flow_1d = self.update_block(
                                net, 
                                inp, 
                                corr, flow_1d, 
                                iter32 = self.n_gru_layers==3, 
                                iter16 = self.n_gru_layers>=2
                                )
                else: # single GRU
                    net, up_mask, delta_flow_1d = self.update_block(net, inp, corr, flow_1d)

            # F(t+1) = F(t) + \Delta(t)
            cost_idx1 = cost_idx1 + delta_flow_1d
            
            # We do not need to upsample or output intermediate results in test_mode
            if (not self.is_training) and itr < iters-1:
                continue

            if _IS_DEBUG_:
                check_nan_inf(inp = delta_flow_1d, name="delta_flow_1d")
            
            """ upsample predictions """
            ##~~ use depth_bin_up
            flow1d_up = self.upsample_flow1d(cost_idx1-cost_idx0, up_mask)
            
            if self.is_gt_flow_1D:
                outputs[('flow1d_iters', 0)].append(flow1d_up)

            # get new disparityindex
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
            with torch.no_grad(): # just for tensorboard visualization
                outputs[('flow1d', 0)] = outputs[('flow1d_iters', 0)][-1] # [N, 1, H, W]

        # save for refine if needed;
        if save_corr_hidden:
            outputs['corr'] = corr
            outputs['hidden'] = net
            outputs['context'] = inp
        
        # iter done
        outputs[("depth", 0, 0)] = depth_up #Nx1xHxW
        return outputs
    
    def indices_to_depth_up_regression(self,
                indices, #[N,1,H,W]
                prob_mask, #[N,H,W,D]
                depth_bins_up,
                is_biliear_index_prob_mask,
                ):
        """
        Convert cost volume indices to depth by grid_sample
        Args:
           prob_mask: [N,H,W,D], here D=num_depth_bins_up;
        """
        indices = indices.permute(0, 2, 3, 1) #to [N,H,W,1]
        batch, h1, w1, _ = indices.shape

        # due to F.grid_sample only supports 
        # spatial (4-D) and volumetric (5-D) input;
        depth_bins = depth_bins_up
        assert depth_bins.dim() == 2
        num_depth_bins = depth_bins.size(1)

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
        
        depth_K_list = []
        for k_idx in range(0, K):
            grid = grids[k_idx]
            #grid_sample: output will have shape [N, C=1, H_out=H, W_out=W]
            depth_k = bilinear_sampler(
                depth_bins[:,None,None,:], #[N,C_in,H_in,W_in]=[N,1,1,D]
                grid #[N,H,W,2]
                )
            depth_K_list.append(depth_k)
        depth_K = torch.cat(depth_K_list, dim=1) #[batch, K, h1, w1]
            
        if is_biliear_index_prob_mask:
            xy_norm = coords_grid_normlized(batch, h1, w1, indices.device) #[N,H,W,2] 
            xy_K_norm = repeat(xy_norm, 'b h w c -> b new_axis h w c', new_axis=K)#[N,K,H,W,2]
            dz = torch.linspace(-r, r, 2*r+1).to(indices.device)
            dz = rearrange(dz, 'k -> 1 k 1 1 1')
            z_K = dz + indices.unsqueeze(1) # [N, K, H, W, 1]
            z_K_norm = 2*z_K / (num_depth_bins - 1) - 1.0 # normalized to range [-1, 1];
            
            grid_3d_norm = torch.cat([xy_K_norm, z_K_norm], dim=-1) #[N, K, H, W, 3] 
            #grid_sample: output will have shape [N, C=1, D_out=K, H_out=H, W_out=W]
            prob_K = F.grid_sample(
                rearrange(prob_mask, 'b h w d -> b () d h w'),
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
                dx = torch.linspace(-r, r, 2*r+1).view(1,1,1,K).to(indices.device) # [1, 1, 1, K]
                index = index + dx # [N, H, W, K]
                index = torch.clamp(index, min=0, max=num_depth_bins-1)
                index = index.type(torch.long)

            prob_K_list = []
            for k_idx in range(0, K):
                # torch.gather: input and index must have the same number of dimensions. 
                prob_k = torch.gather(prob_mask, #[N,H,W,D]
                                    dim=-1, 
                                    index=index[..., k_idx:k_idx+1] #[N,H,W,1]
                                    )
                prob_K_list.append(prob_k.squeeze(dim=-1))
            prob_K = torch.stack(prob_K_list, dim=1) #[N,K,H,W]
        
        #print ("???? prob_K = ", prob_K.shape)    
        # weighted summation for final depth;
        depth = torch.sum(depth_K*prob_K, dim=1, keepdim=True) / (1e-6+torch.sum(prob_K, dim=1, keepdim=True))
         
        depth = torch.clip(depth, min=0.001, max=1000) # in meters
        return depth, prob_K.detach()

    def forward(self, current_image, lookup_images, 
                relative_poses, Ks_src, invK_ref,
                min_depth_bin,
                max_depth_bin,
                iters=12,
                save_corr_hidden = False,
                freeze_fnet_cnet = False # warmup training for several iterations
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
            #print0 ("[////] not freeze_raft_net")
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
    
    
    
    # load pretrained checkpoints from GMA repo;
    def load_pretrained_gma(self, gma_weights_path):
        if self.is_verbose:
            print0 (f"[***] {self.model_card} Try to load pretrained ckpt from {gma_weights_path}")
        if os.path.isfile(gma_weights_path):
            my_pretrained_dict = torch.load(gma_weights_path)
            if 'state_dict' in my_pretrained_dict:
                my_pretrained_dict = my_pretrained_dict['state_dict'] # e.g., 'module.feature_extraction.firstconv.0.0.weight'

            #print ("pretrained gma")
            #for k, v in my_pretrained_dict.items():
            #    print ("{}: {}".format(k, v.shape))
            
            print (f"our model {self.model_card}")
            name_list = ['att', 'update_block']
            # e.g.,: module.update_block.gru.convq2.bias: torch.Size([128])
            #        module.att.to_qk.weight: torch.Size([256, 128, 1, 1])
            wanted_str = {
                'att': 'att.', 
                'update_block': 'update_block.aggregator.'
                }
            for name in name_list:
                if self.is_verbose:
                    print0 ("\t Trying {} ...".format(name))
                
                #for k,v in getattr(self, name).state_dict().items():
                #    print ("{}: {}".format(k, v.shape))
                
                wanted_dict = {k[len(f'module.{name}.'):]: v for k,v in my_pretrained_dict.items() \
                                            if f'module.{wanted_str[name]}' in k }
                
                for k, v in wanted_dict.items():
                    print0 ("{}: {}".format(k, v.shape))
                getattr(self, name).load_state_dict(wanted_dict, strict=False)
            if self.is_verbose:
                print0 ("\t GMA ckpt loading done!")
        else:
            if self.is_verbose:
                print0 ("[!!] Not a valid .pth, skip the ckpt loading ..." )
        
        #sys.exit()
        #import pdb
        #pdb.set_trace()