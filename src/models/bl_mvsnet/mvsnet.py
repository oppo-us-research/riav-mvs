"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


""" load modules from third_parties/IterMVS """
from third_parties.IterMVS.utils import AbsDepthError_metrics 

""" load our own moduels """
from .module import (
    homography_warp, 
    depth_regression
)
from src.layers import (
    BackprojectDepth, 
    Project3D, 
    disp_to_depth
)
from .module import (
    FeatureNet, 
    Depth_Decoder, 
    mvsnet_loss, 
    compute_mvsnet_losses
    )

from src.utils.comm import (is_main_process, print0)

#-----------------------
#-- baseline mvsnet ----
#-----------------------
class baseline_mvsnet(nn.Module):
    def __init__(self, options):
        super(baseline_mvsnet, self).__init__()
        self.opt = options
        
        self.mode = options.mode # 'train', 'resume', 'val' or 'test'
        self.is_train = str(options.mode).lower() in ['train', 'resume']
        self.volume_scale = 2 # quarter scale, 1/2^2 = 1/4
        
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        
        #self.img_scales = [2] # only return depth in quater scale;
        self.img_scales = [0] # only do image reconstruction at s=0 (full scale);
        self.depth_map_scale_int = self.opt.depth_map_scale_int

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        # used to identify principal GPU to print some info;
        if is_main_process():
            self.is_print_info_gpu0 = True
        else:
            self.is_print_info_gpu0 = False
        
        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = frames_to_load
        
        self.is_mvsnet_depth_returned = True

         
        if self.is_print_info_gpu0:
            print('Matching_ids : {}'.format(self.matching_ids))
            print('frames_to_load : {}'.format(frames_to_load))
            print ("[***] is_adaptive_bins = ", self.opt.adaptive_bins)

        # MODEL SETUP
        self.depth_binning = self.opt.depth_binning
        self.num_depth_bins = self.opt.num_depth_bins
        assert self.depth_binning == 'linear', "Baseline MVSNet requires linear depth binning"
        assert self.num_depth_bins in [64, 192], "Baseline MVSNet uses M=64 or 192 plane hypotheses"
        
        self.loss_type = self.opt.loss_type
        if self.is_train:
            assert self.loss_type == 'L1', "MVSNet uses L1 loss type"

        # Dummy modules to align with our code pipeline !!!"
        self.mono_encoder = None
        self.mono_depth = None
        self.pose_encoder = None
        self.pose = None

        self.refine = False

        self.encoder = FeatureNet()
        self.depth = Depth_Decoder(refine=self.refine)
        
        self.min_depth_bin = self.opt.min_depth
        self.max_depth_bin = self.opt.max_depth
        self.depth_bins = self.compute_depth_bins(
            self.min_depth_bin, 
            self.max_depth_bin, 
            self.num_depth_bins)
        
        # TODO: now just try single scale
        for scale in self.img_scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            setattr(self, 'backproject_depth_{}'.format(scale),
                    BackprojectDepth(h, w))

            setattr(self, 'project_3d_{}'.format(scale), Project3D(h, w))
        
        pretrain_weights_path = self.opt.pretrain_mvsnet_path
        if pretrain_weights_path != '' and pretrain_weights_path is not None:
            self.load_pretrained_ckpt(pretrain_weights_path)
    
    
    
    def load_pretrained_ckpt(self, pretrain_weights_path):
        # load pretrained checkpoint
        old_ckpt = torch.load(pretrain_weights_path)
        old_dict = old_ckpt['state_dict'] if 'state_dict' in old_ckpt else old_ckpt['model']
        new_dict = {'encoder': {}, 'depth': {}}
        for k, v in old_dict.items():
            if 'module.feature.' in k:
                new_k = k[len('module.feature.'):]
                new_dict['encoder'][new_k] = v
            elif 'module.cost_regularization.' in k:
                new_k = k[len('module.'):]
                new_dict['depth'][new_k] = v
            elif 'module.refine_network.' in k:
                new_k = k[len('module.'):]
                new_dict['depth'][new_k] = v
            else:
                raise NotImplementedError
            
            #if self.is_print_info_gpu0:
            #    print (f"{k} --> {new_k}")
        
        for k in ['encoder', 'depth']:
            try:
                getattr(self, k).load_state_dict(new_dict[k])
                if self.is_print_info_gpu0:
                    print(f"Loaded weights for model {k} from ckpt {pretrain_weights_path}")
            except Exception as e:
                if self.is_print_info_gpu0:
                    print(e)
                    print(f"[XXXXXXXXXXXXX] Skipping ... model {k}")
                #sys.exit()
    
    def load_our_pretrained_ckpt(self, pretrain_weights_path):
        # load pretrained checkpoint
        old_ckpt = torch.load(pretrain_weights_path)
        old_dict = old_ckpt['state_dict'] if 'state_dict' in old_ckpt else old_ckpt['model']
        new_dict = {
            'encoder': {}, 
            'depth': {}, 
            'pose_encoder': {}, 
            'pose': {}
            }
        
        #for k, v in old_dict.items():
        #    if self.is_print_info_gpu0:
        #        print (f"{k}: {v.shape}")
        
        for k, v in old_dict.items():
            if 'module.encoder.' in k:
                new_k = k[len('module.encoder.'):]
                new_dict['encoder'][new_k] = v
            elif 'module.depth.' in k:
                new_k = k[len('module.depth.'):]
                new_dict['depth'][new_k] = v
            elif 'module.pose_encoder.' in k:
                new_k = k[len('module.pose_encoder.'):]
                new_dict['pose_encoder'][new_k] = v
            elif 'module.pose.' in k:
                new_k = k[len('module.pose.'):]
                new_dict['pose'][new_k] = v
            else: 
                continue
            
            #if self.is_print_info_gpu0:
            #    print (f"{k} --> {new_k}")
        
        for k in ['encoder', 'depth', 'pose_encoder', 'pose']:
            try:
                getattr(self, k).load_state_dict(new_dict[k])
                if self.is_print_info_gpu0:
                    print(f"Loaded weights for model {k} from ckpt {pretrain_weights_path}")
            except Exception as e:
                if self.is_print_info_gpu0:
                    print(e)
                    print(f"[XXXXXXXXXXXXX] Skipping ... model {k}")
                #sys.exit()
        
    
    # get photometric confidence
    def compute_photometric_conf(self,
        num_depth,
        prob_volume #[N,D,H,W]
        ):
        # photometric confidence
        prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
        #print ("prob_volume_sum4 = ", prob_volume_sum4.shape, prob_volume_sum4)
        depth_index = depth_regression(
            prob_volume, 
            depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float).view(1,num_depth)
            )#[N,D,H,W]
        # torch.clamp(): Clamps all elements in input into the range [ min, max ]. 
        depth_index = torch.clamp(depth_index.long(), min=0, max= num_depth-1)
        #print ("??? depth_index = ", depth_index.shape, depth_index.min(), depth_index.max())
        photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index) #[N,1,H,W]
        
        return photometric_confidence
    
    def indices_to_disp(self, indices, depth_bins #[N,D], 2-D tensor
        ):
        """Convert cost volume indices to 1/depth for visualization"""
        batch, height, width = indices.shape
        if depth_bins.shape[0] != batch:
            depth_bins = depth_bins.repeat(batch// depth_bins.size(0), 1).to(
                indices.device)
        #print ("[???] depth_bins = ", depth_bins.shape)
        depth = torch.gather(depth_bins, dim=1, index=indices.view(batch, -1))
        depth = depth.view(batch, 1, height, width)
        #print ("[???] indices = ", indices.shape, " depth = ", depth.shape)
        disp = 1.0 / (depth + 1.0e-8) # to avoid divided by 0;
        return disp

    def compute_depth_bins(self, 
            min_depth_bin, 
            max_depth_bin, 
            num_depth_bins,
            maintain_depth_order=True
            ):
        """
        Compute the depths bins used to build the cost volume. 
        Bins will depend upon self.depth_binning, to either 
        be linear in depth (linear) or linear in inverse depth
        (inverse)
        """
        if isinstance(min_depth_bin, float):
            min_depth_bin = torch.Tensor([min_depth_bin]).float()
            max_depth_bin = torch.Tensor([max_depth_bin]).float()
            min_depth_bin = min_depth_bin.view(1, 1)
            max_depth_bin = max_depth_bin.view(1, 1)
            
        elif isinstance(min_depth_bin, torch.Tensor):
            #assert min_depth_bin.dim() in [2, 4], "should be a 2D or 4D tensor"
            assert min_depth_bin.dim() == 2, "should be a 2D tensor"
            batch = min_depth_bin.size()[0]
            min_depth_bin = min_depth_bin.view(batch,1)
            max_depth_bin = max_depth_bin.view(batch,1)
         
        batch = min_depth_bin.size()[0]
        
        # index: [0, D-1]
        index = torch.arange(0, num_depth_bins, step=1).view(
            1, num_depth_bins).float()
        # in size [1, D], normalized depth, values in [0, 1]
        normalized_sample = index / (num_depth_bins-1)
        if self.depth_binning == 'inverse':
            inverse_depth_min = (1.0/min_depth_bin)
            inverse_depth_max = (1.0/max_depth_bin)
            if maintain_depth_order:
                flip_D_idx = list(range(num_depth_bins-1, -1, -1))
                normalized_sample = normalized_sample[:,flip_D_idx] # maintain depth order
            depth_bins = inverse_depth_max + normalized_sample * (inverse_depth_min - inverse_depth_max)
            depth_bins = 1.0 / depth_bins
        
        elif self.depth_binning == 'linear':
            depth_bins = min_depth_bin + normalized_sample * (max_depth_bin - min_depth_bin) 
        
        else:
            raise NotImplementedError
        
        depth_bins = depth_bins.contiguous() #[N, D]
        return depth_bins 
    
    def run_depth(self, ref_frame, ref_proj, lookup_frames, src_projs, depth_values, ref_frame_quarter=None, refine=False, is_L1_volume=False): 
        #bs, _, img_height, img_width = ref_frame.shape
        num_depth = depth_values.shape[1]
        num_views = len(self.matching_ids) 
        
        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature = self.encoder(ref_frame)
        src_features = [self.encoder(img) for img in lookup_frames]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        #del ref_volume
        # if torch.isnan(ref_proj).any():
        #     print ("Found nan,  ref_proj= ", ref_proj.shape)
        if is_L1_volume:
            #with torch.no_grad():
            L1_volume_shape = (ref_feature.size(0), num_depth, ref_feature.size(2), ref_feature.size(3))
            L1_cost_volume = torch.zeros(L1_volume_shape, dtype=torch.float, device=ref_feature.device)
        
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homography_warp(src_fea, src_proj, ref_proj, depth_values)
            
            if is_L1_volume:
                #with torch.no_grad():
                diffs = reduce(warped_volume*ref_volume,'b c d h w -> b d h w', 'sum')
                L1_cost_volume += diffs
                del diffs
            
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
        if is_L1_volume:
            #with torch.no_grad():
            L1_cost_volume /= len(src_features)
        if refine:
            refined_depth = outputs["refined_depth"]
        else:
            refined_depth = None
        
        if is_L1_volume:
            return depth, prob_volume, L1_cost_volume, refined_depth
        else:
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
        for scale in self.img_scales:
            #source_scale = self.depth_map_scale_int # mvsnet only returns depth at 1/4 scale;
            source_scale = 0 # we have to unsample the depth (at 1/4 scale) by mvsnet;
            
            if not is_depth_returned: # if depth decoder network returns disparity;
                disp = outputs[("disp", scale)]
                # change disparity to depth
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", 0, scale)] = depth # in full-res;
            
            else: # load depth
                depth = outputs[("depth", 0, scale)] # already in full-res;
                outputs[("disp", scale)] = 1.0/(depth + 1e-8) # for disparity smoothness loss;
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                if ('K', frame_id, source_scale) in inputs:
                    K_src = inputs[('K', frame_id, source_scale)]
                else:
                    K_src = inputs[('K', source_scale)]

                if ('inv_K', 0, source_scale) in inputs:
                    invK_ref = inputs[('inv_K', 0, source_scale)]
                else:
                    invK_ref = inputs[('inv_K', source_scale)]

                cam_points = getattr(self, 'backproject_depth_{}'.format(source_scale))(
                    depth, invK_ref)
                pix_coords = getattr(self, 'project_3d_{}'.format(source_scale))(
                    cam_points, K_src, T)

                #outputs[("sample", frame_id, scale)] = pix_coords
                if ("color", frame_id, source_scale) not in inputs:
                    img_0 = inputs[("color", i, 0)]
                    inputs[("color", frame_id, source_scale)] = F.interpolate(
                        img_0, 
                        [img_0.shape[2]//(2**source_scale), img_0.shape[3]//(2**source_scale)], 
                        mode="bilinear", 
                        align_corners=True
                        )
                
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border", align_corners=True)

                     
                # Just save those results for tensorboard summary;
                with torch.no_grad():
                    if ('depth_gt') in inputs:
                        depth_gt = inputs["depth_gt"] # [N,1,H,W]
                    elif ('depth_gt', 0) in inputs:
                        depth_gt = inputs[("depth_gt", 0)] # [N,1,H,W]
                    else:
                        depth_gt = None
                    
                    if depth_gt is not None:
                        depth_gt = F.interpolate(depth_gt,
                                            size=(depth.shape[2], depth.shape[3]),
                                            mode='nearest')
                        cam_points = getattr(self, 'backproject_depth_{}'.format(source_scale))(
                            depth_gt, invK_ref)
                        pix_coords = getattr(self, 'project_3d_{}'.format(source_scale))(
                            cam_points, K_src, T)

                        outputs[("color_gtdepth", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords,
                            padding_mode="border", align_corners=True)
     

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
