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
from .raft_mvs import RAFT_MVS as RAFT_MVS_Base
from .mvs_base import autocast
from .basics import FeatureSPP, offset_layer
from .ada_bins import mViT
from src.models.model_utils import (
    updepth, bilinear_sampler, 
    coords_grid_normlized
)
from src.utils.utils import check_nan_inf
from src.utils.comm import print0
from src.layers import (
    match_features_fst,
    warp_frame_depth,
    transformation_from_parameters
    )
from src.models.mvsdepth.att_gma import (
    Attention, FeatureAggregator
    
)

#_IS_DEBUG_= True
_IS_DEBUG_= False

"""
MVS + RAFT backbone  
+ Adaptive Depth Bins (see this paper: https://arxiv.org/pdf/2011.14141);
  AdaBins: Depth Estimation using Adaptive Bins, CVPR'21;
we changed FlowHead in RAFT/RAFT-Stereo
to output delta 1D-flow (along frontal-parallel depth planes);
"""
class RAFT_MVS(RAFT_MVS_Base):
    def __init__(self, *args, **kwargs):
        # load parent model and initialization;
        super(RAFT_MVS, self).__init__(*args, **kwargs)
        self.model_card = "raft_mvs_adaBins"
        if self.is_verbose:
            print0 (f"[***] {self.model_card} initialization done")

        #---------------------------------
        # newly added methods and variables
        #---------------------------------
        self.adaptive_bins = True
        self.nll_loss = kwargs.get('nll_loss', True)
        self.num_stage = kwargs.get('num_stage', 2)
        self.pose_iters = kwargs.get('pose_iters', 3)
        self.frame_ids = kwargs.get('frame_ids', [0,1,2])

        print0 ("[***] residual pose_iters # = ", self.pose_iters)
        self.is_offset_at_warping = False
        assert self.fnet_name == 'pairnet_fnet', "Requires pairnet_fnet as feature net"
        assert self.is_fusion_feats == True, \
                    "AdaBins: Use SPF (Spatial Pyramid Feature)"
        assert self.num_depth_bins == 64, \
                    f"Requires num_depth_bins=64, but got {self.num_depth_bins}"
        
        self.is_f1gma = kwargs.get('is_f1gma', False)
        
        # Map model to be loaded to specified single gpu.
        #self.gpu_rank = kwargs.get('gpu_rank', None)
        #self.map_location =  "cuda:{}".format(self.gpu_rank) if self.gpu_rank is not None else "cuda"



        # --- reset ----
        pretrain_dvmvs_pairnet_dir = kwargs.get('pretrain_dvmvs_pairnet_dir', None)
        self.feature_extraction = self.feature_extraction_pairnet
        if pretrain_dvmvs_pairnet_dir != '' and pretrain_dvmvs_pairnet_dir is not None:
            assert self.is_training, "Only for training, we load pretrained pairnet encoder."\
                "\nOtherwise, load weights from your checkpoint instead!!!"
            self.load_pretrained_pairnet_ckpt(pretrain_dvmvs_pairnet_dir)



        self.is_dot_product = True
        match_kwargs = {
                'is_training': self.is_training,
                'is_dot_product': self.is_dot_product, # dot product, not L1 distance;
                'set_missing_to_max' : False, # Do not set missing cost volume to its max_values
                'is_edge_mask': False, # do not consider pixels warped out of boundary;
                'is_max_corr_pixel_view': self.is_max_corr_pixel_view,
                }
        self.my_match_func = match_features_fst(**match_kwargs)
        if self.is_verbose:
            print ('[***] using layer match_features_fst()')
        self.is_gt_flow_1D = False
        # no cost_agg;
        assert self.raft_depth_init_type in ['soft-argmin', 'none']

        #------ hyper-parameters ----
        #base_convx_upscale = 4
        # spf: Spatial Pyramid feature Fusion
        fpn_output_channels = 32
        self.fmap_dim = fmap_dim = 128 # feature_fusion output dim;
        self.spf = FeatureSPP(in_planes=4*fpn_output_channels, out_planes= fmap_dim)

        offset_range = 1
        if self.is_offset_at_warping:
            self.offset = offset_layer(
                2*fmap_dim, # ref_feat and warped(src_feat)
                offset_range
                )

        # for mViT, we use quarter or half scale;
        self.mViT_scale = kwargs.get('mViT_scale', 'half')
        self.mVit_fmap_dim = fmap_dim
        assert self.mViT_scale in ['half', 'full', 'none']
        ndepth = self.num_depth_bins

        if self.mViT_scale == 'none':
            self.adaptive_bins_layer = None
        else:
            self.adaptive_bins_layer = mViT(
                    in_channels = self.mVit_fmap_dim,
                    patch_size=16,
                    dim_out= ndepth,
                    group_num = 4,
                    embedding_dim=128,
                    num_heads = 4,
                    norm= 'linear',
                )
        
        if self.is_f1gma:
            self.atten_num_heads = kwargs.get('atten_num_heads', 4)
            # attention mechanism
            self.f1_att = Attention(
                            dim= fmap_dim, 
                            heads= self.atten_num_heads,
                            max_pos_size=160, 
                            dim_head= 128, 
                            #position_type = 'content_only'
                            position_type = 'position_and_content'
                            )
            self.f1_aggregator = FeatureAggregator(
                            input_dim = fmap_dim, 
                            head_dim = 128, 
                            num_heads = self.atten_num_heads
                            )
        else:
            self.f1_att = None
            self.f1_aggregator = None



        pretrain_adabin_ckpt = kwargs.get('pretrain_adabin_ckpt', None)
        if self.adaptive_bins_layer is not None:
            self.load_pretrained_adabin_ckpt(pretrain_adabin_ckpt)

        pretrain_cnet_ckpt = kwargs.get('pretrain_cnet_ckpt', None)
        pretrain_gru_ckpt = kwargs.get('pretrain_gru_ckpt', None)
        if self.cnet is not None:
            self.load_pretrained_cnet_ckpt(pretrain_cnet_ckpt)
        if self.update_block is not None:
            self.load_pretrained_gru_ckpt(pretrain_gru_ckpt)
        
        if self.is_f1gma:
            pretrain_gma_ckpt = kwargs.get('pretrain_gma_ckpt', None)
            print0 (f"???? pretrain_gma_ckpt = {pretrain_gma_ckpt}")
            self.load_pretrained_gma_ckpt(pretrain_gma_ckpt)


    def load_pretrained_adabin_ckpt(self, pretrain_adabin_ckpt):
        if pretrain_adabin_ckpt != '' and pretrain_adabin_ckpt is not None:
            # load pretrained checkpoint
            if self.is_verbose:
                print0 ("loading ada_bin ckpt from ", pretrain_adabin_ckpt)
            tmp_ckpt = pretrain_adabin_ckpt
            assert os.path.exists(tmp_ckpt), f"Not exists {tmp_ckpt}"
            # Map model to be loaded to specified single gpu.
            pretrain_weights = torch.load(tmp_ckpt, 
                                          #map_location=self.map_location
                                          )['state_dict']

            #tmp_idx = 0
            #for k, v in pretrain_weights.items():
            #    if self.is_verbose:
            #        print ("{:03d} {}: {} {}".format(tmp_idx, k, v.shape, v.device))
            #    tmp_idx += 1

            len1 = len('module.depth.adaptive_bins_layer.')
            to_load_ada_bin = {
                k[len1:]: v for k, v in pretrain_weights.items() if ".adaptive_bins_layer." in k
                }

            # adaptive bins
            self.adaptive_bins_layer.load_state_dict(to_load_ada_bin, strict=True)
            if self.is_verbose:
                print0 ("\tSuccessfully loaded!!!")

    def load_pretrained_cnet_ckpt(self, pretrain_cnet_ckpt):
        if pretrain_cnet_ckpt != '' and pretrain_cnet_ckpt is not None:
            # load pretrained checkpoint
            if self.is_verbose:
                print0 ("loading cnet ckpt from ", pretrain_cnet_ckpt)
            tmp_ckpt = pretrain_cnet_ckpt
            assert os.path.exists(tmp_ckpt), f"Not exists {tmp_ckpt}"
            pretrain_weights = torch.load(tmp_ckpt, map_location=self.map_location)['state_dict']
            len2 = len('module.depth.cnet.')
            to_load = {
                k[len2:]: v for k, v in pretrain_weights.items() if ".cnet." in k
                }
            self.cnet.load_state_dict(to_load, strict=True)
            if self.is_verbose:
                print0 ("\tSuccessfully loaded!!!")

    def load_pretrained_gru_ckpt(self, pretrain_gru_ckpt):
        if pretrain_gru_ckpt != '' and pretrain_gru_ckpt is not None:
            # load pretrained checkpoint
            if self.is_verbose:
                print0 ("loading gru ckpt from ", pretrain_gru_ckpt)
            tmp_ckpt = pretrain_gru_ckpt
            assert os.path.exists(tmp_ckpt), f"Not exists {tmp_ckpt}"
            # Map model to be loaded to specified single gpu.
            pretrain_weights = torch.load(tmp_ckpt, map_location=self.map_location)['state_dict']
            len1 = len('module.depth.update_block.')
            to_load = {
                k[len1:]: v for k, v in pretrain_weights.items() if ".update_block." in k
                }
            # adaptive bins
            self.update_block.load_state_dict(to_load, strict=True)
            if self.is_verbose:
                print0 ("\tSuccessfully loaded!!!")
            #sys.exit()

    def load_pretrained_gma_ckpt(self, pretrain_gma_ckpt):
        if pretrain_gma_ckpt != '' and pretrain_gma_ckpt is not None:
            # load pretrained checkpoint
            if self.is_verbose:
                print ("loading gma ckpt from ", pretrain_gma_ckpt)
            tmp_ckpt = pretrain_gma_ckpt
            assert os.path.exists(tmp_ckpt), f"Not exists {tmp_ckpt}"
            # Map model to be loaded to specified single gpu.
            pretrain_weights = torch.load(tmp_ckpt, map_location=self.map_location)['state_dict']
            len1 = len('module.depth.f1_att.')
            to_load = {
                k[len1:]: v for k, v in pretrain_weights.items() if ".f1_att." in k
                }
            self.f1_att.load_state_dict(to_load, strict=True)
            
            len2 = len('module.depth.f1_aggregator.')
            to_load2 = {
                k[len2:]: v for k, v in pretrain_weights.items() if ".f1_aggregator." in k
                }
            # adaptive bins
            self.f1_aggregator.load_state_dict(to_load2, strict=True)
            if self.is_verbose:
                print ("\tSuccessfully loaded!!!")
            #sys.exit()

    def rectify_poses_warp_iter(self,
            ref_image, lookup_images,
            relative_poses,
            Ks_src_full, # full resolution as image;
            invK_ref_full,
            pred_depth,
            pose_iters = 1,
            pose_encoder = None,
            pose_decoder = None
            ):
        """
        Predict residual poses between input frames,
        to rectify the "Ground Truth" pose.
        E.g., we think the GT pose on ScanNet is not accurate enough;
        """
        assert lookup_images.dim() == 5
        batch_size, num_frames, chns_img, height_img, width_img = lookup_images.shape
        assert num_frames == relative_poses.size(1) == Ks_src_full.size(1)
        scale_int = 0
        outputs = {}

        relative_poses_new = []

        for idx, f_i in enumerate(self.frame_ids[1:]):
            K_src = Ks_src_full[:, idx]
            # initialization relative pose;
            rel_pose = relative_poses[:,idx]
            for iter in range(pose_iters):
                # predict poses for reprojection loss
                # select what features the pose network takes as input
                ## I_ref and warped(I_src), to predict the residual pose;

                # get relative pose from ref to src view
                # T^src_ref = T^src_w * T^w_ref = T^src_w * inv(T^ref_w),
                # i.e., = Extrinsic_src * inv(Extrinsic_ref);

                warped_src_fea, edge_mask = warp_frame_depth(
                        depth_ref = pred_depth,
                        src_fea = lookup_images[:,idx],
                        relative_pose = rel_pose,
                        K_src = K_src,
                        invK_ref = invK_ref_full,
                        is_edge_mask = True)

                # assuming frames in temporal order
                pose_inputs = [ref_image, warped_src_fea]
                pose_inputs = [pose_encoder(torch.cat(pose_inputs, 1))]

                axisangle, translation = pose_decoder(pose_inputs)

                # Invert the matrix if the frame id is negative
                resi_pose = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0)).float()

                # left matrix-multiplication residual pose matrix to adjust pose;
                rel_pose = torch.matmul(resi_pose, rel_pose)


            # end of iteration, save results
            relative_poses_new.append(rel_pose)
            outputs[("cam_T_cam", 0, f_i)] = rel_pose # used for image reconstruction loss;
            outputs[("reproj_mask", f_i, scale_int)] = edge_mask
            #outputs[('relative_pose', f_i)] = rel_pose
            #outputs[("residual_cam_T_cam", 0, f_i)] = resi_pose
            #outputs[("axisangle", 0, f_i)] = axisangle
            #outputs[("translation", 0, f_i)] = translation

        #return outputs
        relative_poses_new = torch.stack(relative_poses_new, dim=1) #[N,V,4,4]
        return relative_poses_new, outputs

    # run RAFT-backbone for MVS depth estimation;
    def run_raft_depth(self, ref_image, lookup_images,
                relative_poses,
                Ks_src_stages,
                invK_ref_stages,
                min_depth_bin,
                max_depth_bin,
                iters_stages = [6,6,6],
                pose_encoder = None,
                pose_decoder = None,
                is_verbose = False,
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

        """ Ada_Bin layer: 1/2 scale"""
        # bin_widths_normed: [N, D]
        if self.adaptive_bins_layer is not None:
            # spf_layer
            feat_2_mViT = self.spf(ref_feat_dict, target_scale= self.mViT_scale)
            bin_widths_normed = self.adaptive_bins_layer(feat_2_mViT, is_split=False)

        #---- feature fusion 1/4 scale --- #
        ref_feat = self.spf(ref_feat_dict, target_scale=self.raft_volume_scale)
        lookup_feat = self.spf(lookup_feat_dict, target_scale=self.raft_volume_scale)
        lookup_feat = rearrange(lookup_feat, '(b v) c h w -> b v c h w', v=num_frames)
        #in case in mixed_precision
        ref_feat = ref_feat.float()
        lookup_feat = lookup_feat.float()


        # adaptive bins low and upper bound
        ada_bins_low, ada_bins_high = min_depth_bin, max_depth_bin
        # preprated for cascaded stages
        outputs[('depth_iters', 0)]  = []
        outputs[('prob_iters', 0)] = []
        depth_prev = None # depth from previous stage
        cost_idx_init = None

        # offset for warping
        if self.is_offset_at_warping:
            offset_layer = getattr(self, 'offset')
        else:
            offset_layer = None

        # ----------------------------
        # ---- adaptive depth bins ---
        # ----------------------------
        if self.adaptive_bins_layer is not None:
            depth_bins = self.compute_adaptive_depth_bins(
                bin_widths_normed, #[N,D]
                depth_binning = 'linear',
                cur_depth = None,
                min_depth_bin = ada_bins_low,
                max_depth_bin = ada_bins_high
                ) #[N,D,H,W] or [N,D,1,1]

        else:
            depth_bins = self.compute_depth_bins(min_depth_bin, max_depth_bin,
                                                self.num_depth_bins)
        if depth_bins.dim() == 4:
            assert depth_bins.size(-1) == depth_bins.size(-2) == 1, \
                    "depth bins in size [N,D,1,1]"
            #outputs[f'ada_bins/stage{stage_idx}'] = rearrange(
            #    depth_bins, 'b d 1 1 -> b d')
            depth_bins = rearrange( depth_bins, 'b d 1 1 -> b d')

        outputs['depth_bins'] = depth_bins

        for stage_idx in range(self.num_stage):
            raft_iter = iters_stages[stage_idx]
            cur_scale_str = self.raft_volume_scale
            cur_scale = self.down_scale_int
            num_depth_bins = self.num_depth_bins
            fmap_dim = self.fmap_dim
            corr_l = self.corr_levels
            corr_r = self.corr_radius
            cur_h, cur_w = height_img//cur_scale, width_img // cur_scale

            if is_verbose:
                print("@stage{}/{}: raft_iter={:>2}, scale={:>2}/{:<10}, {:>3}x{:<3}, ndepth={:>2}, feat_dim={:>3}, corr_l={}, corr_r={}".format(
                    stage_idx+1, self.num_stage, raft_iter,
                    cur_scale,
                    cur_scale_str,
                    cur_w, cur_h,
                    num_depth_bins, fmap_dim,
                    corr_l, corr_r
                    ))
            # plane-sweeping stereo
            # pose and K for this stage
            Ks_src = Ks_src_stages[cur_scale_str]
            invK_ref = invK_ref_stages[cur_scale_str]
            # plane-sweeping stereo
            # pose and K for this stage
            if depth_bins.dim() == 4 and depth_bins.size(-1) == depth_bins.size(-2) == 1:
                # change dim for plane-sweeping;
                depth_bins = rearrange(depth_bins, "b d 1 1 -> b d")

            # residuaol pose to rectify "GT" pose which is not accurate enough;
            # skip first stage, due to depth_prev is None at stage 0;
            if (pose_encoder is not None) and (pose_decoder is not None) \
                and (depth_prev is not None):
                relative_poses_new, pose_outputs = self.rectify_poses_warp_iter(
                    ref_image,
                    rearrange(
                        lookup_images, '(b v) c h w -> b v c h w', v=num_frames),
                    relative_poses,
                    Ks_src_full = Ks_src_stages['full'], # full resolution as image;
                    invK_ref_full = invK_ref_stages['full'],
                    pred_depth = depth_prev.detach(),
                    pose_iters = self.pose_iters,
                    pose_encoder = pose_encoder,
                    pose_decoder = pose_decoder
                    )
                #print0("[???] relative_poses_old = \n", relative_poses[0], "\n new=\n", relative_poses_new[0])
                # detach(): not pass pose net to cost volume,
                # instead, the pose net will pass to image reconstruction loss;
                #relative_poses = relative_poses_new.detach()
                relative_poses = relative_poses_new
                #NOTE: update ("cam_T_cam", 0, frame_id) for image reconstruction loss;
                outputs.update(pose_outputs)
            
            if self.is_f1gma:
                # attention mechanism to ref feature
                attention = self.f1_att(ref_feat)
                ref_feat_global = self.f1_aggregator(attention, ref_feat)
                ref_feat = ref_feat_global
                if is_verbose:
                    print("[***] ada_depths_bins with f1gma to ref frame")
            
            # cost volume via plane sweeping stereo
            cost_volume, missing_mask = self.my_match_func(
                    depth_bins,
                    ref_feat,
                    lookup_feats = lookup_feat,
                    relative_poses = relative_poses,
                    Ks_src = Ks_src,
                    invK_ref = invK_ref,
                    offset_layer= offset_layer
                )

            soft_cost_idx, softargmin_depth = self.get_raft_cost_idx_init(
                                                cost_volume, depth_bins)
            outputs[f'softargmin_depth/stage{stage_idx}'] = softargmin_depth
            if cost_idx_init is None:
                cost_idx_init = soft_cost_idx

            # ---- main raft iteration ---#
            depth_up, cost_idx1, prob_volume = self.run_raft_depth_per_stage(
                    stage_idx,
                    net, inp,
                    num_depth_bins,
                    # make index D in the last axis;
                    rearrange(cost_volume, 'b d h w -> b h w d').contiguous(),
                    cost_idx_init,
                    depth_bins,
                    raft_iter,
                    outputs
                    )
            # update for next stage;
            cost_idx_init = cost_idx1
           
            # update low and up bound for depth bins;
            #ada_bins_low = outputs[f'depth_min/stage{stage_idx}']
            #ada_bins_high = outputs[f'depth_max/stage{stage_idx}']
            #if stage_idx == self.num_stage - 1:
            #    outputs[(f"depth", 0, 0)] = depth_up #Nx1xHxW

            depth_prev = depth_up


            # last stage we save photometric confidence
            if stage_idx == self.num_stage - 1:
                with torch.no_grad():
                    # photometric confidence
                    prob_volume_sum4 = 4*F.avg_pool3d(
                        F.pad(
                            rearrange(prob_volume, 'b d h w -> b 1 d h w'),
                            pad=(0, 0, 0, 0, 1, 2)),
                        kernel_size=(4, 1, 1),
                        stride=1, padding=0).squeeze(1)
                    # index at last GRU iteration step;
                    depth_index = cost_idx1.long()
                    # soft-agrmin index;
                    #idx_values = torch.arange(num_depth_bins,
                    #                    device=prob_volume.device,
                    #                    dtype=torch.float).view(1,num_depth_bins,1,1)
                    #depth_index = torch.sum(
                    #    prob_volume*idx_values, dim=1, keepdim=True
                    #    ).long() #[N,1,H,W]
                    
                    depth_index = depth_index.clamp(min=0, max=num_depth_bins-1)
                    photometric_confidence = torch.gather(
                        prob_volume_sum4, dim=1, index=depth_index)
                    outputs["confidence"] = photometric_confidence


        # done multi stages
        #check_dict_k_v(outputs, "outputs")
        #sys.exit()
        return outputs


    # run RAFT forward at one of the cascaded stages;
    def run_raft_depth_per_stage(self,
            stage_idx,
            net, # hidden
            inp, # context feature
            num_depth_bins,
            cost_volume, # plane sweeping cost volume, [N,H,W,D]
            cost_idx_init, # corresponding to depth bin index;
            depth_bins,
            iters,
            outputs
            ):
        assert cost_volume.size(-1) == num_depth_bins
        batch_size, feat_height, feat_width = cost_volume.size()[0:3]
        if _IS_DEBUG_: #TODO???
            check_nan_inf(inp = cost_volume, name="vost_volume")

        scale = self.down_scale_int
        corr_levels = self.corr_levels
        corr_radius = self.corr_radius

        # initial idx, [N,1,H,W]
        cost_idx_size = [batch_size, 1, feat_height,  feat_width]
        cost_idx1 = torch.zeros(cost_idx_size).to(cost_volume.device).float()
        cost_idx0 = torch.zeros_like(cost_idx1)
        #print ("[???] cost_idx_init = ", cost_idx_init.shape)
        if cost_idx_init is not None:
            assert cost_idx_init.shape[1] == 1, "cost_idx_init in shape [N,1,H,W]"
            cost_idx1 = cost_idx1 + cost_idx_init.detach()

        # with depth_bins, not depth_bins_up;
        depth_init = self.indices_to_depth(cost_idx1, depth_bins)
        outputs[(f'depth/stage{stage_idx}/init', 0, 0)] = updepth(depth_init, scale=scale).detach()

        # key = ('depth_iters', scale )
        # detach the first iteration
        if stage_idx == 0:
            outputs[('depth_iters', 0)].append(updepth(depth_init, scale=scale).detach())

        #-----------------------
        # start RAFT iterations
        #-----------------------
        corr_fn = self.corr_block(cost_volume, num_levels=corr_levels,
                                radius = corr_radius)

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

            #if _IS_DEBUG_:
            #    check_nan_inf(inp = delta_cost_idx, name= "delta_cost_idx")


            """ upsample predictions """
            hidden = net[0] if self.is_multi_gru else net
            prob_mask = self.prob_net(hidden) #[N,H,W,D]
            if 0:
                # indices 1/4 scale --> 1/1 scale --> sampling depth 1/1 scale;
                flow1d_up = self.upsample_flow1d(
                        down_scale_int = scale,
                        flow1d = cost_idx1-cost_idx0,
                        mask = up_mask,
                        ndepth_scale = 1.0 # 1.0 since here we do not use depth_bins_up;
                    )
                depth_up, prob_K = self.indices_to_depth_regression_2DBins(
                    indices = flow1d_up, #[N,1,H,W]
                    prob_mask = prob_mask, #[N,H,W,D]
                    depth_bins = depth_bins, #[N,D]
                    #is_biliear_index_prob_mask = False,
                    #is_biliear_index_prob_mask = True,
                    is_biliear_index_prob_mask = (not self.nll_loss),
                    )
            if 1:
                # indices 1/4 scale --> sampling depth 1/4 scale 
                # --> upsample depth to 1/1 scale;
                prob_mask_down = F.avg_pool2d(
                    rearrange(prob_mask, 'b h w d -> b d h w'), 
                    kernel_size=4, stride=4
                    )
                prob_mask_down = rearrange(prob_mask_down, 'b d h w -> b h w d')

                depth, prob_K = self.indices_to_depth_regression_2DBins(
                    indices = cost_idx1-cost_idx0, #[N,1,H,W]
                    prob_mask = prob_mask_down, #[N,H,W,D]
                    depth_bins = depth_bins, #[N,D]
                    #is_biliear_index_prob_mask = False,
                    #is_biliear_index_prob_mask = True,
                    is_biliear_index_prob_mask = (not self.nll_loss),
                    )
                depth_up = self.upsample_flow1d(
                        down_scale_int = scale,
                        flow1d = depth,
                        mask = up_mask,
                        ndepth_scale = 1.0 # 1.0 since here we do not use depth_bins_up;
                    )

            outputs[('depth_iters', 0)].append(depth_up)
            prob_mask = rearrange(prob_mask, 'n h w d -> n d h w')
            outputs[('prob_iters', 0)].append(prob_mask)
            # --- end of raft iteration

        # raft iter done
        outputs[(f"depth/stage{stage_idx}/raft", 0, 0)] = depth_up #Nx1xHxW
        outputs[(f"depth", 0, 0)] = depth_up #Nx1xHxW

        # return the results at last iteration
        return depth_up, cost_idx1.detach(), prob_mask.detach()



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


    def indices_to_depth_regression_2DBins(self,
                indices, #[N,1,H,W]
                prob_mask, #[N,H,W,D]
                depth_bins, #[N,D]
                is_biliear_index_prob_mask,
                ):
        """
        Convert cost volume indices to depth by grid_sample
        Args:
           prob_mask: [N,H,W,D], here D=num_depth_bins_up;
        """
        indices = indices.permute(0, 2, 3, 1).contiguous() #to [N,H,W,1]
        batch, h1, w1, _ = indices.shape

        assert depth_bins.dim() == 2 and depth_bins.size(0) == batch
        num_depth_bins = depth_bins.size(1)
        assert prob_mask.size(-1) == num_depth_bins


        depth_min = reduce(depth_bins, 'b d -> b 1',reduction='min')
        depth_min = repeat(depth_min, 'b 1 -> b 1 new_dim1 new_dim2',
                            new_dim1 = h1, new_dim2=w1)

        depth_max = reduce(depth_bins, 'b d -> b 1',reduction='max')
        depth_max = repeat(depth_max, 'b 1 -> b 1 new_dim1 new_dim2',
                            new_dim1 = h1, new_dim2=w1)

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

        #if divid_small_chunk_grid:
        depth_K_list = []
        for k_idx in range(0, K):
            grid = grids[k_idx]
            #grid_sample: output will have shape [N, C=1, H_out=H, W_out=W]
            depth_k = bilinear_sampler(
                rearrange(depth_bins, 'b d -> b 1 1 d'),
                grid #[N,H,W,2]
                )
            depth_K_list.append(depth_k)
        depth_K = torch.cat(depth_K_list, dim=1) #[batch, K, h1, w1]
        #print ("???? depth_K = ", depth_K.shape)

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
            assert self.nll_loss == True, "Prob_mask needs to be learned via NLL loss"
            # here we would like to "gather" the weights with integer "indices"
            # no grad needed to get the integer "indices",
            # The "differentiable" thing of prob_mask has to be gauranteed by
            # the other loss (e.g., NLL loss)
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

        # weighted summation for final depth;
        depth = torch.sum(depth_K*prob_K, dim=1, keepdim=True) / (1e-6+torch.sum(prob_K, dim=1, keepdim=True))
        # assuming max-depth < 100 meters, o.w. change the max=1000 to other values;
        depth = torch.clip(depth, min=0.0001, max=2000) # in meters

        # dynamic bins, so we cannot use constant min/max values;
        #depth = torch.clip(depth, min=depth_min, max=depth_max)
        return depth, prob_K.detach()



    # depth bins is 4D tensor [N,D,H,W]
    # so we have to apply 3d-grid sampling;
    def indices_to_depth_regression_4DBins(self,
                indices, #[N,1,H,W]
                prob_mask, #[N,D,H,W]
                depth_bins, #[N,D,H,W]
                is_biliear_index_prob_mask,
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
        depth_min = reduce(depth_bins, 'b d h w -> b 1 h w',reduction='min')
        depth_max = reduce(depth_bins, 'b d h w -> b 1 h w',reduction='max')

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


    def sample_depth_bounds_for_next_stage(self, indices,
            depth_bins, #[N,D] or [N,D,H,W], 2D/4D tensor
            shift_radius = 2, # shift +2 and -2 bins to find new min/max;
        ):

        batch, _, h1, w1 = indices.shape
        # current depth_bins
        ndepth_cur = depth_bins.size(1)
        assert depth_bins.size(0) == batch
        if depth_bins.dim() == 2:
            dep_cur_idx = rearrange(indices, 'b 1 h w -> b (h w)')

        elif depth_bins.dim() == 4:
            assert depth_bins.size(-1) == w1 and depth_bins.size(-2) == h1
            dep_cur_idx = indices # [N,H*W] or [N,1,H,W]

        with torch.no_grad():
            new_bounds = []
            for shift in [-shift_radius, shift_radius]: # just two values;
                index = dep_cur_idx + shift
                index = torch.floor(index).type(torch.float)
                #index = torch.clip(index, min=0, max=ndepth_next-1) # this is BUG!!
                index = torch.clip(index, min=0, max=ndepth_cur-1)
                index = index.type(torch.long)
                # torch.gather: input and index must have the same number of dimensions.
                #print ("??? depth_bins = {}, index={}".format(depth_bins.shape, index.shape))
                dep_bound = torch.gather(depth_bins, #[N,D] or [N,D,H,W]
                                    dim=1,
                                    index=index # [N,H*W] or [N,1,H,W]
                                    )
                new_bounds.append(dep_bound.view(batch, 1, h1, w1))
        return new_bounds



    # sample one point (vs k-neighbor for regression)
    def indices_to_depth(self, indices, depth_bins, #[N,D] or [N,D,H,W], 2D/4D tensor
        ):
        """ Convert cost volume indices to depth by grid_sample"""
        indices = indices.permute(0, 2, 3, 1).contiguous() #to [N,H,W,1]
        num_depth_bins = depth_bins.size(1)
        batch, h1, w1, _ = indices.shape
        #print ("[???] indices shape = ", indices.shape)
        if depth_bins.dim() == 2:
            assert depth_bins.size(0) == batch
            depth_min = reduce(depth_bins, 'b d -> b 1',reduction='min')
            depth_min = repeat(depth_min, 'b 1 -> b 1 new_dim1 new_dim2', new_dim1 = h1, new_dim2=w1)

            depth_max = reduce(depth_bins, 'b d -> b 1',reduction='max')
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
            depth_min = reduce(depth_bins, 'b d h w -> b 1 h w',reduction='min')
            depth_max = reduce(depth_bins, 'b d h w -> b 1 h w',reduction='max')

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

    def compute_depth_bins_anchor(self,
            cur_depth, # as center bin or anchor bin;
            min_depth_bin,
            max_depth_bin,
            num_depth_bins,
            ):

        assert min_depth_bin.dim() in [2, 4], "should be a 2D or 4D tensor"
        batch = min_depth_bin.size()[0]
        if min_depth_bin.dim() == 2:
            min_depth_bin = min_depth_bin.view(batch,1, 1, 1)
            max_depth_bin = max_depth_bin.view(batch,1, 1, 1)

        def get_lin_bins(max_depth_bin, min_depth_bin, num_bins):
            #[0, num_bins)
            index = torch.arange(0, num_bins, step=1,
                device=max_depth_bin.device).view(1, num_bins, 1, 1).float()
            normalized_sample = index / (num_bins-1)
            depth_bins = min_depth_bin +  normalized_sample* (max_depth_bin-min_depth_bin)
            return depth_bins

        # for left lower bins: [d_min, ..., d]
        left_nums = num_depth_bins // 2
        right_nums = num_depth_bins - num_depth_bins// 2 + 1
        depth_bins_l = get_lin_bins(
            max_depth_bin = cur_depth,
            min_depth_bin = min_depth_bin,
            num_bins = left_nums)
        # for right higher bins: [d, ..., d_max]
        depth_bins_r = get_lin_bins(
            max_depth_bin = max_depth_bin,
            min_depth_bin = cur_depth,
            num_bins = right_nums)
        depth_bins = torch.cat(
            [depth_bins_l,
             depth_bins_r[:,1:,:,:] # skip first element, i.e., cur_depth;
            ], dim=1
            ).contiguous() #[N,D,1,1] or [N,D,H,W]
        assert depth_bins.size(1) == num_depth_bins, "Wrong num_depth_bins!!!"
        return depth_bins

    # NOTE:
    # assuming min_depth_bin and max_depth_bin is the same for all the pixels
    # in the current image; then we can smaple the bins by 1-D grid along
    # the frontal-parallel depth planes direction;
    # otherwise, we have to sample the bins by 3-D grid, with (x,y,z) for
    # (width, height, depth), i.e., pixel (x, y) has depth bin index z;
    def compute_adaptive_depth_bins(self,
            bin_widths_normed, #[N,D]
            depth_binning,
            cur_depth, # as center bin;
            min_depth_bin,
            max_depth_bin,
            ):
        """
        Compute the depths bins used to build the cost volume.
        Bins will depend upon depth_binning, to either
        be linear in depth (linear) or linear in inverse depth
        (inverse)
        """
        num_bins = bin_widths_normed.size(1)
        assert min_depth_bin.dim() in [2, 4], "should be a 2D/4D tensor"
        assert bin_widths_normed.dim() == 2, "shoule be a 2D tensor"
        bin_widths_normed = rearrange(bin_widths_normed, 'b d -> b d 1 1')
        batch = min_depth_bin.size()[0]
        if min_depth_bin.dim() == 2:
            min_depth_bin = min_depth_bin.view(batch,1, 1, 1)
            max_depth_bin = max_depth_bin.view(batch,1, 1, 1)


        #if depth_binning == 'inverse':
        #    disp_max = (1.0/min_depth_bin) #[N,1,H,W] or [N,1,1,1]
        #    disp_min = (1.0/max_depth_bin)

        #    bin_widths = (disp_max - disp_min) * bin_widths_normed #[N,D,*]
        #    bin_widths_leftpad = torch.cat([disp_min, bin_widths], dim=1) #[N,D+1,*]

        #    # cumulative sum
        #    bin_edges = torch.cumsum(bin_widths_leftpad, dim=1)
        #    centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:]) #[N,D,*]
        #    centers = centers.view(batch, num_bins, 1, 1)
        #    if maintain_depth_order:
        #        # reverse order: [D-1, D-2, ..., 0]
        #        flip_D_idx = list(range(num_bins-1, -1, -1))
        #        centers = centers[:, flip_D_idx]
        #    depth_bins = 1.0 / centers

        assert depth_binning == 'linear', "adaptive bins requires linear bins"

        def get_bins(max_depth_bin, min_depth_bin, bin_widths_normed):
            #print ("??? 33: ", max_depth_bin.shape, min_depth_bin.shape, bin_widths_normed.shape)
            bin_widths = (max_depth_bin - min_depth_bin) * bin_widths_normed #[N,D,*]
            bin_widths_leftpad = torch.cat( [min_depth_bin, bin_widths], dim=1) #[N,D+1,*]

            # cumulative sum
            bin_edges = torch.cumsum(bin_widths_leftpad, dim=1)
            centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:]) #[N,D,*]
            depth_bins = centers
            return depth_bins

        if cur_depth is not None:
            # left half and right half
            bin_normed_l, bin_normed_r = torch.split(bin_widths_normed, num_bins//2, dim=1)
            #print ("??? bin_normed_l, bin_normed_r = ", bin_normed_l.shape, bin_normed_r.shape)

            ## left half bins: [depth_min, ..., cur_depth]
            bins_l = get_bins(
                max_depth_bin = cur_depth,
                min_depth_bin = min_depth_bin,
                bin_widths_normed = bin_normed_l
                )

            ## right half bins: [cur_depth, ..., depth_max]
            bins_r = get_bins(
                max_depth_bin = max_depth_bin,
                min_depth_bin = cur_depth,
                bin_widths_normed = bin_normed_r
                )

            depth_bins = torch.cat([bins_l, bins_r], dim=1)

        else:
            depth_bins = get_bins(
                max_depth_bin = max_depth_bin,
                min_depth_bin = min_depth_bin,
                bin_widths_normed = bin_widths_normed
                )

        depth_bins = depth_bins.contiguous() #[N,D,H,W] or [N,D,1,1]
        return depth_bins


    def forward(self, current_image, lookup_images,
                relative_poses,
                Ks_src_stages,
                invK_ref_stages,
                min_depth_bin,
                max_depth_bin,
                iters_stages = [6,6],
                pose_encoder = None,
                pose_decoder = None,
                is_verbose = False
                ):


        outputs = self.run_raft_depth(
                current_image,
                lookup_images,
                relative_poses,
                Ks_src_stages,
                invK_ref_stages,
                min_depth_bin,
                max_depth_bin,
                iters_stages,
                pose_encoder,
                pose_decoder,
                is_verbose
                )

        return outputs

