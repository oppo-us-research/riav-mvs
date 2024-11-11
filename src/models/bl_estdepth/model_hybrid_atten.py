"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

""" load modules from third_parties/ESTDepth """
from third_parties.ESTDepth.networks.layers_op import (
    convbn_3d, convbnrelu_3d
)
from third_parties.ESTDepth.networks.psm_submodule import psm_feature_extraction
from third_parties.ESTDepth.hybrid_models.resnet_encoder import ResnetEncoder
from third_parties.ESTDepth.hybrid_models.hybrid_depth_decoder import DepthHybridDecoder
from third_parties.ESTDepth.utils.homo_utils import homo_warping

""" load our own moduels """
from src.models.mvsdepth.att_gma import (
    Attention, 
    FeatureAggregator
)

Align_Corners_Range = False


class DepthNetHybrid_atten(nn.Module):
    def __init__(self, ndepths=64, depth_min=0.01, depth_max=10.0, resnet=50,
                 IF_EST_transformer=True,
                 **kwargs
                 ):
        """

        :param ndepths:
        :param depth_min:
        :param depth_max:
        :param featureNet: "psm" or "senet"
        """
        super(DepthNetHybrid_atten, self).__init__()

        self.ndepths = ndepths
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_interval = (depth_max - depth_min) / (ndepths - 1)

        # the to.(torch.float32) is required, if not will be all zeros
        self.depth_cands = torch.arange(0, ndepths, requires_grad=False).reshape(1, -1).to(
            torch.float32) * self.depth_interval + self.depth_min

        self.IF_EST_transformer = IF_EST_transformer

        self.matchingFeature = psm_feature_extraction()  # [1/4 scale], the features are not after bn and relu

        #self.semanticFeature = ResnetEncoder(resnet, "pretrained")  # the features after bn and relu
        # updated for UserWarning: 'pretrained' is deprecated `
        self.semanticFeature = ResnetEncoder(resnet, "IMAGENET1K_V2")  # the features after bn and relu


        self.stage_infos = {
            "stage1": {
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        self.CostRegNet = DepthHybridDecoder(self.semanticFeature.num_ch_enc,
                                             num_output_channels=1, use_skips=True,
                                             ndepths=self.ndepths, depth_max=self.depth_max,
                                             IF_EST_transformer=self.IF_EST_transformer)

        self.pre0 = convbn_3d(64, 32, 1, 1, 0)
        self.pre1 = convbnrelu_3d(32, 32, 3, 1, 1)
        self.pre2 = convbn_3d(32, 32, 3, 1, 1)
        
        #---------------------------------
        # newly added methods and variables
        #--------------------------------- 
        self.atten_num_heads = kwargs.get('atten_num_heads', 4)
        self.fmap_dim = fmap_dim = 32 # PSMNet feature;

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
                        num_heads = self.atten_num_heads,
                        head_dim = 128)
        # gma
        gma_weights_path = kwargs.get("gma_weights_path", None)
        if gma_weights_path and self.is_training:
            self.load_pretrained_gma(gma_weights_path)
        
    
    def load_pretrained_gma(self, gma_weights_path):
        if self.is_verbose:
            print ("[???] raft_mvs.RAFT Try to load pretrained ckpt from {}".format(gma_weights_path))
        if os.path.isfile(gma_weights_path):
            my_pretrained_dict = torch.load(gma_weights_path, map_location=self.device)
            if 'state_dict' in my_pretrained_dict:
                my_pretrained_dict = my_pretrained_dict['state_dict'] # e.g., 'module.feature_extraction.firstconv.0.0.weight'

            #print ("pretrained gma")
            #for k, v in my_pretrained_dict.items():
            #    print ("{}: {}".format(k, v.shape))
            
            print ("our model")
            #name_list = ['att', 'update_block', 'cnet']
            name_list = ['att', 'update_block']
            # e.g.,: module.update_block.gru.convq2.bias: torch.Size([128])
            #        module.att.to_qk.weight: torch.Size([256, 128, 1, 1])
            wanted_str = {
                'att': 'att.', 
                'update_block': 'update_block.aggregator.'
                }
            for name in name_list:
                if self.is_verbose:
                    print ("\t Trying {} ...".format(name))
                
                #for k,v in getattr(self, name).state_dict().items():
                #    print ("{}: {}".format(k, v.shape))
                
                wanted_dict = {k[len(f'module.{name}.'):]: v for k,v in my_pretrained_dict.items() \
                                            if f'module.{wanted_str[name]}' in k }
                
                for k, v in wanted_dict.items():
                    print ("{}: {}".format(k, v.shape))
                getattr(self, name).load_state_dict(wanted_dict, strict=False)
            if self.is_verbose:
                print ("\t GMA ckpt loading done!")
        else:
            if self.is_verbose:
                print ("[!!] Not a valid .pth, skip the ckpt loading ..." )
        
        #sys.exit()
        #import pdb
        #pdb.set_trace()




    def get_costvolume(self, features, cam_poses, cam_intr, depth_values):
        """
        return cost volume, [ref_feature, warped_feature] concat
        :param features: middle one is ref feature, others are source features
        :param cam_poses:
        :param cam_intr:
        :param depth_values:
        :return:
        """
        num_views = len(features)
        ref_feature = features[num_views // 2]
        ref_cam_pose = cam_poses[:, num_views // 2, :, :]
        ref_extrinsic = torch.inverse(ref_cam_pose)
        #-----------
        # attention mechanism to ref feature
        attention = self.f1_att(ref_feature)
        ref_feat_global = self.f1_aggregator(attention, ref_feature)
        #print ("[???] ref_feat_global =", ref_feat_global.shape)
        
        #print ("[???] cam_intr K = \n", cam_intr)
        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feat_global.unsqueeze(2).repeat(1, 1, self.ndepths, 1, 1)
        costvolume = torch.zeros_like(ref_volume).to(ref_volume.dtype).to(ref_volume.device)
        for view_i in range(num_views):
            if view_i == (num_views // 2):
                continue
            src_fea = features[view_i]
            src_cam_pose = cam_poses[:, view_i, :, :]
            src_extrinsic = torch.inverse(src_cam_pose)
            # warpped features
            src_proj_new = src_extrinsic.detach().clone()
            ref_proj_new = ref_extrinsic.detach().clone()
            src_proj_new[:, :3, :4] = torch.matmul(cam_intr, src_extrinsic[:, :3, :4])
            ref_proj_new[:, :3, :4] = torch.matmul(cam_intr, ref_extrinsic[:, :3, :4])
            #print ("after proj: src_proj_new = ", src_proj_new)
            #print ("after proj: ref_proj_new = ", ref_proj_new)
            #print ("processing view_i=", view_i)
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            #ipdb.set_trace()

            # it seems that ref_volume - warped_volume not good
            x = torch.cat([ref_volume, warped_volume], dim=1)
            x = self.pre0(x)
            x = x + self.pre2(self.pre1(x))

            costvolume = costvolume + x
        # aggregate multiple feature volumes by variance
        costvolume = costvolume / (num_views - 1)
        del warped_volume
        del x
        return costvolume

    def scale_cam_intr(self, cam_intr, scale):
        cam_intr_new = cam_intr.clone()
        cam_intr_new[:, :2, :] *= scale

        return cam_intr_new

    def forward(self, imgs, cam_poses, cam_intr, sample, pre_costs=None, pre_cam_poses=None, mode='train'):
        """
        input seqs (0,1,2,3,4) target view will be (1,2,3) or input three views
        :param imgs:
        :param cam_poses:
        :param cam_intr:
        :param sample:
        :return:
        """
        #imgs = 2 * (imgs / 255.) - 1.
        imgs = 2 *imgs - 1.
        batch_size, views_num, _, height_img, width_img = imgs.shape
        if mode == 'val':
            assert batch_size == 1, "sequential order requries bs=1: t-2,t-1,t,t+1,t+2"
        #print ("imgs shape = ", imgs.shape)
        height = height_img // 4
        width = width_img // 4
        assert views_num > 2  # the views_num should be larger than 2

        target_num = views_num - 2
        # step1, matching feature extraction

        matching_features = self.matchingFeature(imgs.view(batch_size * views_num, 3, height_img, width_img))
        matching_features = matching_features.view(batch_size, views_num, -1, height, width)
        matching_features = matching_features.permute(1, 0, 2, 3, 4).contiguous()

        # semantic_features = []
        # for target_i in range(target_num):
        #     sematic_feature = self.semanticFeature(imgs[:, target_i + 1])
        #     semantic_features.append(sematic_feature)

        # learn context
        semantic_features = self.semanticFeature(
            #imgs[:, 1:1 + target_num].view(batch_size * target_num, -1, height_img, width_img)
            imgs[:, 1:1 + target_num].contiguous().view(batch_size * target_num, -1, height_img, width_img)
            )
        # list of [feature0, feature1, ...]

        cam_intr_stage1 = self.scale_cam_intr(cam_intr, scale=1. / self.stage_infos["stage1"]["scale"])

        depth_values = self.depth_cands.view(1, self.ndepths, 1, 1
                                             ).repeat(batch_size, 1, 1, 1).to(imgs.dtype).to(imgs.device)

        cost_volumes = []
        target_cam_poses = []
        target_depths = []
        target_masks = []
        target_imgs = []
        for target_i in range(target_num):
            cost_volume = self.get_costvolume(matching_features[(target_i + 1) - 1: (target_i + 1) + 2],
                                              cam_poses[:, (target_i + 1) - 1: (target_i + 1) + 2, :, :],
                                              cam_intr_stage1,
                                              depth_values)

            # assert torch.all(torch.isfinite(cost_volume)), "Nan in costvolume"

            cost_volumes.append(cost_volume)
            target_cam_poses.append(cam_poses[:, target_i + 1, :, :])
            target_depths.append(sample["dmaps"][:, target_i + 1, :, :, :])
            target_masks.append(sample["dmasks"][:, target_i + 1, :, :, :])
            target_imgs.append(imgs[:, target_i + 1, :, :, :])

        outputs, cur_costs, cur_cam_poses = self.CostRegNet(cost_volumes,
                                                            semantic_features,
                                                            target_cam_poses,
                                                            cam_intr_stage1,
                                                            depth_values,
                                                            self.depth_min,
                                                            self.depth_interval,
                                                            pre_costs, pre_cam_poses,
                                                            mode)

        if mode == 'train':
            losses = self.depth_loss_scales(outputs, [0, 1, 2, 3], target_depths, target_masks, target_imgs, target_num)

            return outputs, losses
        elif mode == 'test':
            metrics = self.depth_metrics(outputs, [0, 2], target_depths, target_masks, target_num)
            return outputs, metrics
        else:
            return outputs, cur_costs, cur_cam_poses

    def depth_loss_scales(self, outputs, scales, depth_gt_ms, gt_masks, imgs, target_num, weight=0.8):
        losses = {}
        device = depth_gt_ms[0].device

        loss = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
        #l1_rel = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
        for scale in scales:
            losses['loss_{}'.format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
            # \sigma < 1.25, acccuracy;
            losses["a1_{}".format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device,
                                                            requires_grad=False)
            # abs_rel error
            losses["abs_rel_{}".format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device,
                                                            requires_grad=False)
            for target_i in range(target_num):
                pred = outputs[("depth", target_i, scale)]

                _, _, H, W = pred.shape

                gt = depth_gt_ms[target_i]
                mask = gt_masks[target_i]
                img = imgs[target_i]

                # have no effect
                # loss_w_ = SMOOTH_W * self.get_smooth_loss(pred, img) / (2 ** scale)
                # print(scale, loss_w_)
                losses['loss_{}'.format(scale)] += F.l1_loss(pred[mask], gt[mask])

                delta_i, thred_i = self.depth_stats(gt=gt.clone(), pr=pred.clone())  # here need to clone

                losses["a1_{}".format(scale)] += delta_i
                losses["abs_rel_{}".format(scale)] += thred_i

            losses["a1_{}".format(scale)] /= target_num
            losses["abs_rel_{}".format(scale)] /= target_num
            losses['loss_{}'.format(scale)] /= target_num
            loss = loss + (weight ** scale) * losses['loss_{}'.format(scale)]
            #l1_rel = l1_rel + losses['abs_rel_{}'.format(scale)]

        losses['loss'] = loss  # / len(scales)
        #losses['l1_rel'] = l1_rel / len(scales)# added by CCJ;

        return losses

    def get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    def depth_stats(self, gt, pr):

        mask = (gt > self.depth_min) & (gt < self.depth_max)
        pr[pr < self.depth_min] = self.depth_min
        pr[pr > self.depth_max] = self.depth_max

        gt = gt[mask] # ground truth
        pr = pr[mask] # prediction

        thresh, _ = torch.max(torch.stack([gt / pr, pr / gt], dim=0), dim=0, keepdim=False)
        delta = torch.mean((thresh < 1.25).to(torch.float32))
        abs_rel = torch.mean(torch.abs(gt - pr) / gt)

        return delta, abs_rel

    def depth_metrics(self, outputs, scales, depth_gt_ms, gt_masks, target_num):
        metrics = {}
        device = depth_gt_ms[0].device

        for scale in scales:
            metrics['a1_{}'.format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=False)
            metrics["a2_{}".format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device,
                                                          requires_grad=False)
            metrics["a3_{}".format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device,
                                                          requires_grad=False)

            metrics['abs_diff_{}'.format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device,
                                                                requires_grad=False)
            metrics["abs_rel_{}".format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device,
                                                               requires_grad=False)
            metrics["sq_rel_{}".format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device,
                                                              requires_grad=False)

            metrics["rmse_{}".format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device,
                                                            requires_grad=False)
            metrics["rmse_log_{}".format(scale)] = torch.tensor(0.0, dtype=torch.float32, device=device,
                                                                requires_grad=False)
            for target_i in range(target_num):
                pred = outputs[("depth", target_i, scale)]

                _, _, H, W = pred.shape

                gt = depth_gt_ms[target_i]
                mask = gt_masks[target_i]

                a1, a2, a3, abs_diff, abs_rel, sq_rel, rmse, rmse_log = self.metrics(gt[mask], pred[mask])

                metrics['a1_{}'.format(scale)] += (a1 / target_num)
                metrics["a2_{}".format(scale)] += (a2 / target_num)
                metrics["a3_{}".format(scale)] += (a3 / target_num)

                metrics['abs_diff_{}'.format(scale)] += (abs_diff / target_num)
                metrics["abs_rel_{}".format(scale)] += (abs_rel / target_num)
                metrics["sq_rel_{}".format(scale)] += (sq_rel / target_num)

                metrics["rmse_{}".format(scale)] += (rmse / target_num)
                metrics["rmse_log_{}".format(scale)] += (rmse_log / target_num)

        return metrics

    def metrics(self, gt, pred):
        thresh = torch.max((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        abs_diff = torch.mean(torch.abs(gt - pred))
        abs_rel = torch.mean(torch.abs(gt - pred) / gt)

        sq_rel = torch.mean(((gt - pred) ** 2) / gt)

        rmse = torch.sqrt(torch.mean((gt - pred) ** 2))

        rmse_log = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2))

        return a1, a2, a3, abs_diff, abs_rel, sq_rel, rmse, rmse_log
