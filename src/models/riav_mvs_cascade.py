"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

""" project related libs """
from src import models
from src.models.mvsdepth import __mvs_depth_models__
from src import models
from src.layers import (BackprojectDepth, Project3D,
        disp_to_depth, transformation_from_parameters,
        warp_frame_depth,
        get_coord_feat,
        SSIM
        )
from src.loss_utils import (calculate_loss_with_mask, calculate_loss, 
    calculate_loss_keepSum, update_losses, 
    update_raft_losses, LossMeter, BinsChamferLoss, 
    nll_losses
)
from src.utils.comm import (is_main_process, print0)


# exclude extremly large displacements
MAX_FLOW = 700
# save raft iter result
_ITR_STEP_FOR_SAVE_TB = 2

# cascaded mvs
class RIAV_MVS_CAS(nn.Module):
    def __init__(self, options):
        super(RIAV_MVS_CAS, self).__init__()
        self.opt = options
        self.mode = options.mode # 'train', 'resume', 'val' or 'test'
        self.is_train = str(options.mode).lower() in ['train', 'resume']
        self.raft_mvs = str(options.raft_mvs_type).lower()
        assert self.raft_mvs in [
                                "raft_mvs_casbins",
                                "raft_mvs_adabins",
                                ], \
            "Wrong raft_mvs type: {}".format(self.raft_mvs)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, f"height={self.opt.height} must be a multiple of 32"
        assert self.opt.width % 32 == 0, f"width={self.opt.width} must be a multiple of 32"

        self.num_scales = len(self.opt.scales)
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"


        # used to identify principal GPU to print some info;
        self.is_print_info_gpu0 = is_main_process()

        # here we mean dynamic depth bins for the cascade cost volume;
        self.is_adaptive_bins = self.opt.adaptive_bins
        assert self.is_adaptive_bins == True

        if self.is_print_info_gpu0:
            print ("[***] sequence loss gamma = %f" % self.opt.raft_loss_gamma)
            print ("[***] raft iters = %s" %self.opt.raft_iters )
            print ("[***] is_adaptive_bins = ", self.is_adaptive_bins)

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = frames_to_load


        if self.is_print_info_gpu0:
            print('Matching_ids : {}'.format(self.matching_ids))
            print('frames_to_load : {}'.format(frames_to_load))

        # small num
        self.epsilon = 1.0e-8
        # MODEL SETUP
        self.encoder = None

        self.loss_type = self.opt.loss_type
        assert self.loss_type in ['L1-inv', 'L1'], "Wrong loss type"
        if self.loss_type == 'L1-inv':
            assert abs(self.opt.m_2_mm_scale_loss - 1.0) < 1.0e-4, \
                f"Require self.opt.m_2_mm_scale_loss == 1.0, but get {self.opt.m_2_mm_scale_loss}"

        self.is_raft_alternate_corr = self.opt.is_raft_alternate_corr

        my_kargs = {
            'corr_implementation': 'reg',
            'is_max_corr_pixel_view': not self.opt.is_avg_corr_raft,
            'residual_pose_net': self.opt.pose_net_type,
            'num_stage': self.opt.num_stage,
            'mViT_scale': self.opt.mViT_scale,
            'pretrain_adabin_ckpt': self.opt.pretrain_adabin_ckpt,
            'nll_loss': self.opt.w_nll_loss > 0,
            'pose_iters': 3,
            'pretrain_pose_ckpt': self.opt.pretrain_pose_ckpt,
            'pretrain_cnet_ckpt': self.opt.pretrain_cnet_ckpt,
            'pretrain_gru_ckpt': self.opt.pretrain_gru_ckpt,
            'frame_ids': self.opt.frame_ids,
            'is_f1gma': self.opt.is_f1gma,
            'pretrain_gma_ckpt': self.opt.gma_pretrained_path,
            'is_verbose' : self.is_print_info_gpu0,
            }

        my_kargs['is_mixed_precision'] = self.opt.is_raft_mixed_precision
        my_kargs['fnet_norm_fn'] = self.opt.raft_fnet_norm_fn
        my_kargs['cnet_norm_fn'] = self.opt.raft_cnet_norm_fn
        my_kargs['fnet_name'] = self.opt.fnet_name
        my_kargs['raft_volume_scale'] = self.opt.raft_volume_scale

        my_kargs['raft_weights_path'] = self.opt.raft_pretrained_path
        my_kargs['pretrain_dvmvs_pairnet_dir'] = self.opt.pretrain_dvmvs_pairnet_dir
        my_kargs['n_gru_layers'] = self.opt.n_gru_layers
        print0 ("[???] self.opt.fusion_pairnet_feats = ", self.opt.fusion_pairnet_feats)
        
        my_kargs['fusion_pairnet_feats'] = self.opt.fusion_pairnet_feats
        my_kargs['raft_depth_init_type'] = self.opt.raft_depth_init_type
        my_kargs['prob_radius'] = self.opt.raft_mvs_pss_prob_radius
        self.is_depth_returned = True

        if self.is_print_info_gpu0:
            print ("[***] raft_mvs = {}".format(self.raft_mvs))

        my_kargs['atten_num_heads'] = self.opt.gma_atten_num_heads

        if not self.opt.no_ssim:
            self.ssim = SSIM()

        self.residual_pose_net = my_kargs.get('residual_pose_net', 'none')
        if self.residual_pose_net == 'resnet_pose':
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
                            num_frames_to_predict_for=1
                            )

            pretrain_pose_ckpt = my_kargs.get('pretrain_pose_ckpt', None)
            if pretrain_pose_ckpt != '' and pretrain_pose_ckpt is not None:
                self.load_pretrained_posenet_ckpt(pretrain_pose_ckpt)
        else:
            self.pose_encoder = self.pose = None

        self.depth = __mvs_depth_models__[self.raft_mvs](
            input_height=self.opt.height,
            input_width=self.opt.width,
            mode=self.mode,
            min_depth_bin=self.opt.min_depth,
            max_depth_bin=self.opt.max_depth,
            num_depth_bins=self.opt.num_depth_bins,
            adaptive_bins= self.is_adaptive_bins,
            depth_binning=self.opt.depth_binning,
            ** my_kargs
            )


        # TODO: now just try a single scale
        assert len(self.opt.scales) == 1, "supervisedmvs+raft-pss: now just try single scale!!!"
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            setattr(self, 'backproject_depth_{}'.format(scale),
                    BackprojectDepth(self.opt.batch_size, h, w, device=self.device)
                    )

            setattr(self, 'project_3d_{}'.format(scale),
                    Project3D(self.opt.batch_size, h, w)
                    )

    def load_pretrained_posenet_ckpt(self, pretrain_pose_ckpt):
        # load pretrained checkpoint
        if self.is_print_info_gpu0:
            print ("loading pose ckpt from ", pretrain_pose_ckpt)
        # Map model to be loaded to specified single gpu.
        pretrain_weights = torch.load(pretrain_pose_ckpt, map_location=self.device)['state_dict']
        #tmp_idx = 0
        #for k, v in pretrain_weights.items():
        #    print ("{:03d} {}: {} {}".format(tmp_idx, k, v.shape, v.device))
        #    tmp_idx += 1
        len1 = len('module.pose_encoder.')
        len2 = len('module.pose.')
        to_load_pose_encoder = {
            k[len1:]: v for k, v in pretrain_weights.items() if ".pose_encoder." in k
            }
        to_load_pose_decoder = {
            k[len2:]: v for k, v in pretrain_weights.items() if ".pose." in k
            }

        # encoder
        self.pose_encoder.load_state_dict(to_load_pose_encoder, strict=True)
        # decoder
        self.pose.load_state_dict(to_load_pose_decoder, strict=True)
        #sys.exit()
        if self.is_print_info_gpu0:
            print ("\tSuccessfully loaded!!!")



    def forward(self, inputs, is_train, is_verbose,
                freeze_raft_fnet_cnet=False):
        """
        ## for multi-gpu parallel efficiency,
        ## here we put warping, loss etc time consuming functions
        ## inside the forward()

        Pass a minibatch through the network and generate images and losses
        """

        outputs = {}
        # prepare data
        img_key = "color_aug" if self.is_train else "color"
        #The reference frame
        ref_frame = inputs[img_key, 0, 0]
        #The source frames
        lookup_frames = [inputs[(img_key, idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w


        # if dynamic depth bin max/min provided by the dataloader;
        if self.is_adaptive_bins:
            assert ("min_depth_tracker" in inputs) and ("max_depth_tracker" in inputs)
            min_depth_bin = inputs["min_depth_tracker"] #[N, 1]
            max_depth_bin = inputs["max_depth_tracker"] #[N, 1]

        else:
            min_depth_bin = inputs["depth_min"] #[N, 1]
            max_depth_bin = inputs["depth_max"] # [N, 1]

        if is_verbose:
            print ("[????] min_depth_bin/max_depth_bin = ",
                        min_depth_bin.min().item(),
                        max_depth_bin.max().item()
                        )


        outputs['min_depth_bin'] = min_depth_bin
        outputs['max_depth_bin'] = max_depth_bin

        scales_dict = {
            4: "sixteenth", # 2^4
            3: "eighth",
            2: 'quarter',
            1: 'half',
            0: 'full',
            }

        ref_idx = 0
        invK_ref_stages = {}
        Ks_src_stages = {}

        #for scale_int in [0,1,2,3,4]:
        for scale_int in [0,1,2,3]:
            if ('inv_K', ref_idx, scale_int) in inputs:
                invK_ref = inputs[('inv_K', ref_idx, scale_int)]
            else:
                invK_ref = inputs[('inv_K', scale_int)]
            invK_ref_stages[scales_dict[scale_int]] = invK_ref

            # Updated: assuming distinctive K per frame;
            Ks_src = []
            for f_i in self.matching_ids[1:]:
                if ('K', f_i, scale_int) in inputs:
                    K_src = inputs[('K', f_i, scale_int)]
                else:
                    K_src = inputs[('K', scale_int)]
                Ks_src.append(K_src)
            Ks_src = torch.stack(Ks_src, 1)  # batch x frames x 4 x 4
            Ks_src_stages[scales_dict[scale_int]] = Ks_src

        # load GT pose from dataset, save relative pose to inputs;
        pose_pred = self.cal_poses(inputs)
        # Also save relative pose to outputs;
        outputs.update(pose_pred)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for
                                                idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        # two stages: [4,8]
        iters_stages = self.opt.raft_iters

        outputs.update(
            self.depth(
                    ref_frame,
                    lookup_frames,
                    relative_poses,
                    Ks_src_stages, invK_ref_stages,
                    min_depth_bin, max_depth_bin,
                    iters_stages = iters_stages,
                    pose_encoder = self.pose_encoder,
                    pose_decoder = self.pose,
                    is_verbose = is_verbose
                ))

        #if is_verbose:
        #    print ("  [**dep_min/dep_max]: tracker %f/%f" %(
        #        min_depth_bin.min().item(), max_depth_bin.max().item(),
        #    ))

        for stage_idx in range(4):
            tmp_key = f'softargmin_depth/stage{stage_idx}'
            if tmp_key in outputs:
                soft_depth = outputs[tmp_key]
                outputs[tmp_key] = F.interpolate(
                    soft_depth, [self.opt.height, self.opt.width], mode="nearest")


        #---------------------------------------
        # already update ("cam_T_cam", 0, f_i) to output;
        self.generate_images_pred(inputs, outputs, \
                is_depth_returned=self.is_depth_returned, is_multi=True)


        # before get loss, do dMap_to_indxMap first;
        #self.get_flow_1d_idxDMap(inputs, outputs)

        losses = self.compute_mvs_and_sequence_losses(
            is_train = is_train,
            inputs = inputs,
            outputs = outputs,
            is_depth_returned = self.is_depth_returned,
            gamma = self.opt.raft_loss_gamma,
            min_depth = min_depth_bin.view(-1,1,1,1), #[N,1,1,1]
            max_depth = max_depth_bin.view(-1,1,1,1),
            is_verbose = is_verbose,
            loss_type = self.loss_type
            )

        # debuging
        #check_dict_k_v(outputs, 'outputs')
        #check_dict_k_v(losses, 'losses')

        return outputs, losses

    def cal_poses(self, inputs,
            #residual_poses = None
        ):
        """
        calculate relative poses using absolute poses, for example, from KITTI Raw;
        """
        outputs = {}
        for f_i in self.opt.frame_ids[1:]:
            # get relative pose from ref to src view
            # T^src_ref = T^src_w * T^w_ref = T^src_w * inv(T^ref_w),
            # i.e., = Extrinsic_src * inv(Extrinsic_ref);
            rel_pose = torch.matmul(inputs[("pose", f_i)], inputs[("pose_inv", 0)])

            #if residual_poses:
            #    resi_pose = residual_poses[("residual_cam_T_cam", 0, f_i)]
            #    # left matrix-multiplication residual pose matrix to adjust pose;
            #    rel_pose = torch.matmul(resi_pose, rel_pose)

            ## Syntax: ("cam_T_cam", reference frame, source frame):
            # backward-map the coordinates from target view to source view;
            # then the backward-warped coordinates (aka grid) is used in
            # `torch.NN.F.grid_sample()` func, to generate the synthesized view
            # of source frame (i.e., by warping it to target view);
            outputs[("cam_T_cam", 0, f_i)] = rel_pose

            # save the relative_pose to inputs dict;
            inputs[('relative_pose', f_i)] = rel_pose

        return outputs


    def generate_images_pred(self, inputs, outputs, is_depth_returned = False, is_multi=False):
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
                outputs[("disp", scale)] = 1.0/(depth + self.epsilon) # for disparity smoothness loss;

            ## since we use supervised loss,
            # just save those results for tensorboard summary;
            #with torch.no_grad():
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                if ('K', frame_id, source_scale) in inputs:
                    K_src = inputs[('K', frame_id, source_scale)]
                else:
                    K_src = inputs[('K', source_scale)]

                if ('inv_K', source_scale, 0) in inputs:
                    invK_ref = inputs[('inv_K', source_scale, 0)]
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

            if 0:
                with torch.no_grad():
                    for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                        if ('depth_gt', 0) in inputs:
                            depth_gt = inputs[("depth_gt", 0)] # [N,1,H,W]
                            cam_points = getattr(self, 'backproject_depth_{}'.format(source_scale))(
                                depth_gt, invK_ref)
                            pix_coords = getattr(self, 'project_3d_{}'.format(source_scale))(
                                cam_points, K_src, T)

                            outputs[("color_gtdepth", frame_id, scale)] = F.grid_sample(
                                inputs[("color", frame_id, source_scale)],
                                pix_coords,
                                padding_mode="border", align_corners=True)

                        # for each iteration in RAFT
                        if is_multi:
                            step = _ITR_STEP_FOR_SAVE_TB
                            if is_depth_returned:
                                num_iters = len(outputs[('depth_iters', scale)])
                                itr_list = range(1, num_iters, step)
                            elif ('disp_iters', scale) in outputs:
                                num_iters = len(outputs[('disp_iters', scale)])
                                itr_list = range(1, num_iters, step) # skip the disp_init
                            else: # just use loss on the flow at last iteration step;
                                itr_list = [-1]

                            for itr in itr_list:
                                if ('disp_iters', scale) in outputs:
                                    # change disparity to depth
                                    disp_cur = outputs[('disp_iters', scale)][itr] #Nx1xHxW
                                    _, depth_cur = disp_to_depth(disp_cur, self.opt.min_depth, self.opt.max_depth)
                                elif ('depth_iters', scale) in outputs:
                                    depth_cur = outputs[('depth_iters', scale)][itr] #Nx1xHxW

                                else:
                                    depth_cur = outputs[('depth', 0, 0)]

                                cam_points = getattr(self, 'backproject_depth_{}'.format(source_scale))(
                                    depth_cur, invK_ref)
                                pix_coords = getattr(self, 'project_3d_{}'.format(source_scale))(
                                    cam_points, K_src, T)

                                outputs[("color_iters", frame_id, scale, itr)] = F.grid_sample(
                                    inputs[("color", frame_id, source_scale)],
                                    pix_coords,
                                    padding_mode="border",
                                    #padding_mode="zeros",
                                    align_corners=True)

    # adapted from RAFT paper;
    def compute_flow_sequence_loss(self, flow_preds, flow_gt, gamma=0.9, max_flow=MAX_FLOW, valid_mask = None):

        """ Loss function defined over sequence of flow predictions """
        n_predictions = len(flow_preds)
        flow_loss = 0.0
        flow_unweighted_loss_list = []

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        if valid_mask is not None:
            valid = (valid_mask >= 0.5) & (mag < max_flow)
        else:
            # 0 for invalid value in GT flow;
            valid = (mag > 0) & (mag < max_flow)
        #if 1:
        #    tmp_N, tmp_H, tmp_W = valid.shape
        #    tmp_num = tmp_N * tmp_H * tmp_W
        #    print ("flow valid mask norm {}%={}/{}".format(valid.float().sum()/tmp_num*100, valid.float().sum(), tmp_num))
        #    ## > e.g.: flow valid mask norm 67.71621704101562%=355028.0/524288;

        for i in range(n_predictions):
            assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any(), \
                "i={},pred_flow1d shape = {}, val = {}".format(i, flow_preds[i].shape, flow_preds[i])
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            #loss = (valid[:, None] * i_loss).mean()
            loss = (valid[:, None] * i_loss).sum()/(valid.float().sum() + self.epsilon)

            flow_unweighted_loss_list.append(loss)
            flow_loss += i_weight * loss

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean(),
            '1px': (epe < 1).float().mean(),
            '3px': (epe < 3).float().mean(),
            '5px': (epe < 5).float().mean(),
        }
        return flow_unweighted_loss_list, flow_loss, metrics


    def get_flow1D_iters_loss_and_metrics(self, inputs, outputs, gamma, flow_loss_weight, is_verbose=False):
        # set up key-values
        losses = {}
        losses['flow_1d_epe'] = .0
        losses['flow_1d_1px'] = .0
        losses['flow_1d_3px'] = .0
        losses['flow_1d_5px'] = .0

        # a little naming abuse;
        # here flow_1d means the depth index map via torch.bucketize();
        flow_gt = inputs[('flow_1d', 0)][:,[0]] #[N, 1, H, W]
        flow_mask = inputs[('flow_1d', 0)][:,1] #[N,H,W]
        flow_preds = outputs[('flow1d_iters', 0)][1:]# skip the flow_init

        flow_unweighted_loss_list, flow_loss, flow_metrics = \
            self.compute_flow_sequence_loss(flow_preds, flow_gt, gamma, valid_mask=flow_mask)


        for key, val in flow_metrics.items():
            losses['flow_1d_' + key] += val

        if flow_loss_weight > 0:
            tmp_loss = flow_loss_weight* flow_loss
            losses['loss/flow_1d'] = tmp_loss
        if is_verbose:
            n_iters = len(flow_unweighted_loss_list)
            if flow_loss_weight > 0:
                messg = "{} Iter (un-weighted) FLOW-1D losses = ".format(n_iters) + ("{:.4f}, "*(n_iters)).format(
                    *flow_unweighted_loss_list)
            else:
                messg = "{} (NoCounted) Iter (un-weighted) FLOW-1D losses = ".format(n_iters) + ("{:.4f}, "*(n_iters)).format(
                    *flow_unweighted_loss_list)
            print (messg)
        return losses

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

    # compute sequence loss
    # sequence means a sequence of depth by each iteration in RAFT;
    def compute_mvs_and_sequence_losses(self,
        is_train,
        inputs, outputs, \
        is_depth_returned,
        gamma, #e.g., =0.9,
        min_depth, #e.g, =0.1,
        max_depth, #e.g, =80.0,
        is_verbose =False,
        loss_type = "L1-inv", # inverse depth
        ):
        """
        Compute the reprojection, smoothness and
        proxy supervised losses for a minibatch
        """
        losses = {}
        if is_train:
            # batch loss
            # those losses are use for tensorboard, not to Optimizer for back-propagation;
            losses['loss/last_itr/L1'] = .0
            losses['loss/last_itr/L1-inv'] = .0
            losses['loss/last_itr/L1-rel'] = .0
        else:
            # keep the sum and count for accumulation;
            losses['batch_l1_meter_sum'] = .0
            losses['batch_l1_inv_meter_sum'] = .0
            losses['batch_l1_rel_meter_sum'] = .0
            losses['batch_l1_meter_count'] = .0

        # loss accumulator
        l1_meter = LossMeter()
        l1_inv_meter = LossMeter()
        l1_rel_meter = LossMeter()

        depth_gt = inputs[("depth_gt", 0)]
        #print ("[???] depth_gt = ", depth_gt)
        if "depth_mask" in inputs:
            mask = inputs["depth_mask"] > 0.5
        else:
            mask = (depth_gt >= min_depth) & (depth_gt <= max_depth)

        scale = 0 # actually we only have 1 scale;

        num_iters = len(outputs[('depth_iters', scale)])
        itr_start = 1 # skip the init

        #print ("[???] num_iters = ", num_iters)
        #print ("[??] self.opt.freeze_raft_net: ", self.opt.freeze_raft_net)

        """ for losses """
        weights = []
        predictions = []
        for itr in range(itr_start, num_iters):
            multi_depth = outputs[('depth_iters', scale)][itr] #Nx1xHxW
            loss_weight_itr = gamma**(num_iters - itr - 1 - itr_start)
            weights.append(loss_weight_itr)
            predictions.append(multi_depth)

        # loss in raft iteraion
        optimizer_loss, unweighted_loss_iters = update_raft_losses(
                            predictions=predictions,
                            weights=weights,
                            groundtruth= depth_gt,
                            valid_mask = mask,
                            is_training= is_train,
                            l1_meter = l1_meter,
                            l1_inv_meter = l1_inv_meter,
                            l1_rel_meter = l1_rel_meter,
                            loss_type = loss_type)
        if is_verbose:
            print (f"{num_iters-1} (un-weighted) losses:")
            messg = "  l1_rel: " + ("{:.4f}, "*(num_iters-1)).format(*unweighted_loss_iters['l1_rel'])
            print (messg)
            messg = "  l1_abs: " + ("{:.4f}, "*(num_iters-1)).format(*unweighted_loss_iters['l1_abs'])
            print (messg)

        if self.pose and self.pose_encoder:
            source_scale = 0
            reprojection_losses = []
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                edge_mask = outputs[("reproj_mask", frame_id, scale)]
                reprojection_losses.append(
                    self.compute_reprojection_loss(pred*edge_mask, target*edge_mask)
                    )
            reprojection_losses = torch.cat(reprojection_losses, 1)
            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
            losses['loss/reproj_loss/{}'.format(scale)] = reprojection_loss
            #print ("[???] reproj_loss = ", reprojection_loss)
            optimizer_loss = optimizer_loss + reprojection_loss.mean()

        if self.opt.softargmin_loss:
            for stage_idx in range(self.opt.num_stage):
                tmp_key = f'softargmin_depth/stage{stage_idx}'
                if tmp_key in outputs:
                    #print ("soft depth loss")
                    l1_loss, l1_inv_loss, l1_rel_loss = calculate_loss(
                                groundtruth = depth_gt, #[N,1,H,W]
                                prediction = outputs[tmp_key], #[N,1,H,W]
                            )
                    if self.loss_type == "L1":
                        tmp_loss = l1_loss
                    else:
                        tmp_loss = l1_inv_loss
                    losses[f'loss/softargmin_depth/stage{stage_idx}'] = tmp_loss
                    optimizer_loss = optimizer_loss + weights[0]*tmp_loss

        if self.opt.w_chamfer > 0:
            adabins_loss_fn = BinsChamferLoss()
            #stage_idx = 0
            #tmp_key = f'ada_bins/stage{stage_idx}'
            tmp_key = 'depth_bins'
            if tmp_key in outputs:
                ada_depth_bins = outputs[tmp_key]
                assert ada_depth_bins.dim() == 2, "requries 2D tensor"
                chamfer_loss = adabins_loss_fn(
                    ada_depth_bins, # [N,D]
                    depth_gt, #[N,1,H,W]
                    mask, #[N,1,H,W]
                    )
                losses['loss/adabins_chamfer'] = self.opt.w_chamfer*chamfer_loss
                optimizer_loss = optimizer_loss + self.opt.w_chamfer*chamfer_loss

        if self.opt.w_nll_loss > 0:
            # classification loss
            nll_loss = nll_losses(
                probabilities = outputs[('prob_iters', scale)],
                weights = weights, # weight list
                depth_bins = outputs['depth_bins'], #[N,D]
                groundtruth = depth_gt, #[N,1,H,W]
                valid_mask = mask, #[N,1,H,W]
                )
            #print ("[???] nll_loss = ", nll_loss)
            losses['loss/nll_class'] = self.opt.w_nll_loss*nll_loss
            optimizer_loss = optimizer_loss + self.opt.w_nll_loss*nll_loss

        # to Optimizer for back-propagation
        losses['loss'] = optimizer_loss


        """ save for logging to tensorboard """
        # those losses are use for tensorboard,
        # not to Optimizer for back-propagation;
        with torch.no_grad():
            if is_train:
                # we will calculate the running loss in run_epoch() function;
                losses['loss/last_itr/L1'] += l1_meter.avg
                losses['loss/last_itr/L1-inv'] += l1_inv_meter.avg
                losses['loss/last_itr/L1-rel'] += l1_rel_meter.avg

                for i, itr in enumerate(range(itr_start, num_iters)):
                    if itr % _ITR_STEP_FOR_SAVE_TB == 0:
                        losses["loss/l1_rel/raftItr-{}".format(itr)] = unweighted_loss_iters['l1_rel'][i]* weights[i]
                        losses["loss/l1_abs/raftItr-{}".format(itr)] = unweighted_loss_iters['l1_abs'][i]* weights[i]
            else:
                # for validation, return the meter loss, to accumulate batch loss;
                losses['batch_l1_meter_sum'] += l1_meter.sum
                losses['batch_l1_inv_meter_sum'] += l1_inv_meter.sum
                losses['batch_l1_rel_meter_sum'] += l1_rel_meter.sum
                losses['batch_l1_meter_count'] += l1_meter.count
                #print ("l1_meter", l1_meter)

        return losses


""" Evaluation """
class RIAV_MVS_CAS_Eval(RIAV_MVS_CAS):
    def __init__(self, options):
        ## updated for evaluation mode
        print ("[!!!] RIAV_MVS_eval : Reset class attributes !!!")
        options.mode = 'test'
        options.raft_pretrained_path = None # do not use pretrained RAFT, since we will use our own checkpoint;
        options.pretrain_dvmvs_pairnet_dir = None

        # If track_running_stats is set to False,
        # this layer then does not keep running estimates,
        # and batch statistics are instead used during evaluation time.
        #print ("[???] ToDo: currently just set options.track_norm_running_stats = False, "
        #       "might be changed later???")
        #options.track_norm_running_stats = False # do not track
        super(RIAV_MVS_CAS_Eval, self).__init__(options, None)

    # modified for eval/test mode
    def cal_poses(self, data, frames_to_load,
                #residual_poses = None
            ):
        """
        predict the poses and save them to `data` dictionary;
        """
        for fi in frames_to_load[1:]:
            rel_pose = torch.matmul(data[("pose", fi)], data[("pose_inv", 0)])
            #if residual_poses:
            #    resi_pose = residual_poses[("residual_cam_T_cam", 0, fi)]
            #    # left matrix-multiplication residual pose matrix to adjust pose;
            #    rel_pose = torch.matmul(resi_pose, rel_pose)

            data[('relative_pose', fi)] = rel_pose



    def forward(self, data, frames_to_load, is_verbose):
        # prepare data
        # for test, use 'color' not 'color_aug'
        input_color = data[('color', 0, 0)]
        my_res = {}

        if torch.cuda.is_available():
            input_color = input_color.cuda()
            #print ("[???] input_color shape ", input_color.shape)

        ## use 'color' not 'color_aug'
        lookup_frames = [data[('color', idx, 0)] for idx in frames_to_load[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w
        #lookup_frame_ids = frames_to_load[1:]
        #batch_size, num_frames, _, height, width = lookup_frames.shape

        scale_int = {"sixteenth": 4, "eighth": 3,
                    'quarter': 2,
                    'half': 1,
                    'full': 0,
                    }[self.opt.raft_volume_scale] # 2^i, here i=1,2,3;


        ref_idx = 0
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


        if torch.cuda.is_available():
            lookup_frames = lookup_frames.cuda()
            Ks_src = Ks_src.cuda()
            invK_ref = invK_ref.cuda()

        if self.is_adaptive_bins:
            assert ("min_depth_tracker" in data) and ("max_depth_tracker" in data)
            min_depth_bin = data["min_depth_tracker"] #[N, 1]
            max_depth_bin = data["max_depth_tracker"] #[N, 1]

        else:
            min_depth_bin = data["depth_min"] #[N, 1]
            max_depth_bin = data["depth_max"] # [N, 1]

        if torch.cuda.is_available():
            min_depth_bin = min_depth_bin.cuda()
            max_depth_bin = max_depth_bin.cuda()

        self.min_depth_tracker = min_depth_bin.mean().item()
        self.max_depth_tracker = max_depth_bin.mean().item()


        # load GT pose from dataset, save relative pose to data;
        self.cal_poses(data, frames_to_load)
        # relative pose;
        relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
        relative_poses = torch.stack(relative_poses, 1)
        if torch.cuda.is_available():
            relative_poses = relative_poses.cuda()

        invK_ref_stages = {}
        Ks_src_stages = {}

        scales_dict = {
            4: "sixteenth", # 2^4
            3: "eighth",
            2: 'quarter',
            1: 'half',
            0: 'full',
            }
        for scale_int in [0,1,2,3]:
            if ('inv_K', ref_idx, scale_int) in data:
                invK_ref = data[('inv_K', ref_idx, scale_int)]
            else:
                invK_ref = data[('inv_K', scale_int)]
            invK_ref_stages[scales_dict[scale_int]] = invK_ref

            # Updated: assuming distinctive K per frame;
            Ks_src = []
            for f_i in self.matching_ids[1:]:
                if ('K', f_i, scale_int) in data:
                    K_src = data[('K', f_i, scale_int)]
                else:
                    K_src = data[('K', scale_int)]
                Ks_src.append(K_src)
            Ks_src = torch.stack(Ks_src, 1)  # batch x frames x 4 x 4
            Ks_src_stages[scales_dict[scale_int]] = Ks_src

        iters_stages = self.opt.raft_iters
        output = self.depth(
                input_color,
                lookup_frames,
                relative_poses,
                Ks_src_stages,
                invK_ref_stages,
                min_depth_bin,
                max_depth_bin,
                iters_stages = iters_stages,
                pose_encoder = self.pose_encoder,
                pose_decoder = self.pose,
                is_verbose = is_verbose
            )

        pred_depth = output[("depth", 0, 0)]
        #stage_idx = 0
        #pred_depth = output[(f"depth/stage{stage_idx}/raft", 0, 0)]
        pred_disp = 1.0/(pred_depth)
        output[("disp", 0)] = pred_disp

        my_res['disp'] = pred_disp
        my_res['depth'] = pred_depth
        if "confidence" in output:
            my_res["confidence"] =  output["confidence"]

        ## for print
        if is_verbose:
            _min_depth = pred_depth.min(-1)[0].min(-1)[0]
            _max_depth = pred_depth.max(-1)[0].max(-1)[0]
            min_depth_bin = self.min_depth_tracker
            max_depth_bin = self.max_depth_tracker
            print ("  -- pred-depth: min_depth=%f, max_depth=%f;" %(
                _min_depth.mean(), _max_depth.mean()
                ) + " ++ tracker: min_depth_bin = %f, max_depth_bin = %f" %(
                min_depth_bin, max_depth_bin
                ))
            batch_idx = 0
            print ("  -- depth bins = {}".format(output['depth_bins'][batch_idx]))

        return my_res

