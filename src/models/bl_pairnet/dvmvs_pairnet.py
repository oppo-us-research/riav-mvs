"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""
# ------------------------------------------------------------------------------------
# Modified from DeepVideoMVS (https://github.com/ardaduz/deep-video-mvs)
# MIT license.
# ------------------------------------------------------------------------------------

from collections import OrderedDict
import time
import math
import numpy as np
import random
import os
import sys
from path import Path

import torch


""" load modules from third_parties/DeepVideoMVS """
from third_parties.DeepVideoMVS.dvmvs.pairnet.model import(
    EncoderBlock,
    DecoderBlock,
    FeatureExtractor,
    FeatureShrinker,
)
from third_parties.DeepVideoMVS.dvmvs.layers import (
    conv_layer, depth_layer_3x3
)


""" load our own moduels """
from src.layers import match_features_fst
from src.models.mvsdepth.basics import SubModule


fpn_output_channels = 32
hyper_channels = 32

def _upsample(x):
    return torch.nn.functional.interpolate(x, 
                scale_factor=2,
                mode='bilinear',
                align_corners=True, 
                #recompute_scale_factor=True
                )


class CostVolumeEncoder(torch.nn.Module):
    def __init__(self, input_channels):
        super(CostVolumeEncoder, self).__init__()
        self.input_channels = input_channels
        #print ("[???] pairnet cv encoder input dim = ", self.input_channels)
        self.aggregator0 = conv_layer(
                    input_channels= self.input_channels + fpn_output_channels,
                    output_channels=hyper_channels,
                    kernel_size=5,
                    stride=1,
                    apply_bn_relu=True)

        self.encoder_block0 = EncoderBlock(
                    input_channels=hyper_channels,
                    output_channels=hyper_channels * 2,
                    kernel_size=5
                    )
        ###
        self.aggregator1 = conv_layer(
                    input_channels=hyper_channels * 2 + fpn_output_channels,
                    output_channels=hyper_channels * 2,
                    kernel_size=3,
                    stride=1,
                    apply_bn_relu=True
                    )
        self.encoder_block1 = EncoderBlock(input_channels=hyper_channels * 2,
                                           output_channels=hyper_channels * 4,
                                           kernel_size=3)
        ###
        self.aggregator2 = conv_layer(input_channels=hyper_channels * 4 + fpn_output_channels,
                                      output_channels=hyper_channels * 4,
                                      kernel_size=3,
                                      stride=1,
                                      apply_bn_relu=True)
        self.encoder_block2 = EncoderBlock(input_channels=hyper_channels * 4,
                                           output_channels=hyper_channels * 8,
                                           kernel_size=3)

        ###
        self.aggregator3 = conv_layer(
                    input_channels=hyper_channels * 8 + fpn_output_channels,
                    output_channels=hyper_channels * 8,
                    kernel_size=3,
                    stride=1,
                    apply_bn_relu=True)

        self.encoder_block3 = EncoderBlock(input_channels=hyper_channels * 8,
                                           output_channels=hyper_channels * 16,
                                           kernel_size=3)

    def forward(self, features_half, features_quarter,
                features_one_eight,
                features_one_sixteen, 
                cost_volume,
                ):
        
        inp0 = torch.cat([features_half, cost_volume], dim=1)

        inp0 = self.aggregator0(inp0)
        out0 = self.encoder_block0(inp0)

        inp1 = torch.cat([features_quarter, out0], dim=1)
        inp1 = self.aggregator1(inp1)
        out1 = self.encoder_block1(inp1)

        inp2 = torch.cat([features_one_eight, out1], dim=1)
        inp2 = self.aggregator2(inp2)
        out2 = self.encoder_block2(inp2)

        inp3 = torch.cat([features_one_sixteen, out2], dim=1)
        inp3 = self.aggregator3(inp3)
        out3 = self.encoder_block3(inp3)

        return inp0, inp1, inp2, inp3, out3


class CostVolumeDecoder(torch.nn.Module):
    def __init__(self,
        opt_min_depth,
        opt_max_depth,
        ):
        super(CostVolumeDecoder, self).__init__()

        self.inverse_depth_base = 1.0 / opt_max_depth
        self.inverse_depth_multiplier = 1.0/opt_min_depth - 1.0/opt_max_depth

        self.decoder_block1 = DecoderBlock(input_channels=hyper_channels * 16,
                                           output_channels=hyper_channels * 8,
                                           kernel_size=3,
                                           apply_bn_relu=True,
                                           plus_one=False)

        self.decoder_block2 = DecoderBlock(input_channels=hyper_channels * 8,
                                           output_channels=hyper_channels * 4,
                                           kernel_size=3,
                                           apply_bn_relu=True,
                                           plus_one=True)

        self.decoder_block3 = DecoderBlock(input_channels=hyper_channels * 4,
                                           output_channels=hyper_channels * 2,
                                           kernel_size=3,
                                           apply_bn_relu=True,
                                           plus_one=True)

        self.decoder_block4 = DecoderBlock(input_channels=hyper_channels * 2,
                                           output_channels=hyper_channels,
                                           kernel_size=5,
                                           apply_bn_relu=True,
                                           plus_one=True)

        self.refine = torch.nn.Sequential(conv_layer(input_channels=hyper_channels + 4,
                                                     output_channels=hyper_channels,
                                                     kernel_size=5,
                                                     stride=1,
                                                     apply_bn_relu=True),
                                          conv_layer(input_channels=hyper_channels,
                                                     output_channels=hyper_channels,
                                                     kernel_size=5,
                                                     stride=1,
                                                     apply_bn_relu=True))

        self.depth_layer_one_sixteen = depth_layer_3x3(hyper_channels * 8)
        self.depth_layer_one_eight = depth_layer_3x3(hyper_channels * 4)
        self.depth_layer_quarter = depth_layer_3x3(hyper_channels * 2)
        self.depth_layer_half = depth_layer_3x3(hyper_channels)
        self.depth_layer_full = depth_layer_3x3(hyper_channels)


    def forward(self, image, skip0, skip1, skip2, skip3, bottom):
        # work on cost volume
        decoder_block1 = self.decoder_block1(bottom, skip3, None)
        sigmoid_depth_one_sixteen = self.depth_layer_one_sixteen(decoder_block1)
        inverse_depth_one_sixteen = self.inverse_depth_multiplier * sigmoid_depth_one_sixteen + self.inverse_depth_base

        decoder_block2 = self.decoder_block2(decoder_block1, skip2, sigmoid_depth_one_sixteen)
        sigmoid_depth_one_eight = self.depth_layer_one_eight(decoder_block2)
        inverse_depth_one_eight = self.inverse_depth_multiplier * sigmoid_depth_one_eight + self.inverse_depth_base

        decoder_block3 = self.decoder_block3(decoder_block2, skip1, sigmoid_depth_one_eight)
        sigmoid_depth_quarter = self.depth_layer_quarter(decoder_block3)
        inverse_depth_quarter = self.inverse_depth_multiplier * sigmoid_depth_quarter + self.inverse_depth_base

        decoder_block4 = self.decoder_block4(decoder_block3, skip0, sigmoid_depth_quarter)
        sigmoid_depth_half = self.depth_layer_half(decoder_block4)
        inverse_depth_half = self.inverse_depth_multiplier * sigmoid_depth_half + self.inverse_depth_base

        scaled_depth = _upsample(sigmoid_depth_half)
        scaled_decoder = _upsample(decoder_block4)
        scaled_combined = torch.cat([scaled_decoder, scaled_depth, image], dim=1)
        scaled_combined = self.refine(scaled_combined)
        inverse_depth_full = self.inverse_depth_multiplier * self.depth_layer_full(scaled_combined) + self.inverse_depth_base

        depth_full = 1.0 / inverse_depth_full.squeeze(1)
        depth_half = 1.0 / inverse_depth_half.squeeze(1)
        depth_quarter = 1.0 / inverse_depth_quarter.squeeze(1)
        depth_one_eight = 1.0 / inverse_depth_one_eight.squeeze(1)
        depth_one_sixteen = 1.0 / inverse_depth_one_sixteen.squeeze(1)

        return depth_full, depth_half, depth_quarter, depth_one_eight, depth_one_sixteen


class DVMVS_PairNet_EncoderMatching(SubModule):
    def __init__(self, input_height, input_width, mode,
                 min_depth_bin,
                 max_depth_bin,
                 num_depth_bins,
                 adaptive_bins,
                 depth_binning,
                 device = torch.device("cuda"),
                 is_print_info_gpu0 = True,
                 pretrain_weights_dir = None # e.g., == 'pretrained_model_KITTI2015.tar'
                 ):

        super(DVMVS_PairNet_EncoderMatching, self).__init__()
        """
        # MnasNet, network with low-latency and lightweightness;
        # Input images are scaled down up to 1/32, and up scaled to 1/2;
        # feature channel = 32;
        """
        ## Specific to DVMVS_PairNet
        self.is_print_info_gpu0 = is_print_info_gpu0
        self.num_ch_enc = fpn_output_channels # 32 here;
        self.feature_extractor = FeatureExtractor()
        self.feature_shrinker = FeatureShrinker()
        self.weight_init()
        if pretrain_weights_dir is not None and pretrain_weights_dir != '':
            self.load_pretrained_ckpt(pretrain_weights_dir)
        ##---------------end of DVMVS_PairNet's feature layers

        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning # 'linear' or 'inverse'(i.e., disparity);
        self.num_depth_bins = num_depth_bins
        self.opt_min_depth = min_depth_bin
        self.opt_max_depth = max_depth_bin

        # we build the cost volume at 1/2 resolution
        self.matching_height, self.matching_width = input_height // 2, input_width // 2

        self.device = device
        self.is_training = (mode in ['train', 'resume'])
        self.depth_bins = self.compute_depth_bins(min_depth_bin, max_depth_bin, maintain_depth_order=False)

        if self.is_print_info_gpu0:
            print ("matching_height, matching_width = ", self.matching_height, self.matching_width)
            print ("[***] DVMVS_PairNet_EncoderMatching initialized !! Calling compute_depth_bins !!")
            print ("  ==> depth_bins: ", self.depth_binning, \
                self.depth_bins.device, self.depth_bins.shape, self.depth_bins.view(-1))

        assert self.depth_bins is not None, "depth_bins Cannot be None"
        self.is_dot_product = True
        kwargs = {
                'is_training': self.is_training,
                'num_ch_enc': self.num_ch_enc,
                'matching_height': self.matching_height,
                'matching_width': self.matching_width,
                'is_dot_product': self.is_dot_product, # dot product, not L1 distance;
                'set_missing_to_max' : False, # Do not set missing cost volume to its max_values
                'is_edge_mask': False, # do not consider pixels warped out of boundary;
                }
        self.my_match_func = match_features_fst(**kwargs)
        if self.is_print_info_gpu0:
            print ('[***] using layer match_features_fst()')


    def load_pretrained_ckpt(self, pretrain_weights_dir):
        # load pretrained checkpoint
        models_name = {
            "feature_extractor": "0_feature_extractor", 
            "feature_shrinker":  "1_feature_pyramid",
            }
        models_name2 = {
            "feature_extractor": "module_0_checkpoint.pth", 
            "feature_shrinker":  "module_1_checkpoint.pth",
            }
        all_checkpoints = sorted(Path(pretrain_weights_dir).files())
        #print ("[**] Found {} possible ckpts:\n{}".format(len(all_checkpoints), all_checkpoints))
        wanted_ckpts = {}
        for ckpt in all_checkpoints:
            for k, file_name in models_name.items():
                if file_name in ckpt or models_name2[k] in ckpt:
                    wanted_ckpts[k] = ckpt
        
        for k in models_name:
            try:
                #checkpoint = os.path.join(pretrain_weights_dir, file_name)
                checkpoint = wanted_ckpts[k]
                weights = torch.load(checkpoint)
                getattr(self, k).load_state_dict(weights)
                if self.is_print_info_gpu0:
                    print("Loaded weights for model {} from ckpt {}".format(k, checkpoint))
            except Exception as e:
                print(e)
                print("[XXXXXXXXXXXXX] Skipping ... model {}".format(k))


    def compute_depth_bins(self, min_depth_bin, max_depth_bin, maintain_depth_order=False):
        """Compute the depths bins used to build the cost volume. Bins will depend upon
        self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
        (inverse)"""

        if self.depth_binning == 'inverse':
            depth_bins = 1 / np.linspace(1 / max_depth_bin,
                                              1 / min_depth_bin,
                                              self.num_depth_bins)

            if maintain_depth_order:
                depth_bins = depth_bins[::-1]  # maintain depth order

        elif self.depth_binning == 'linear':
            depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)
        else:
            raise NotImplementedError

        depth_bins = torch.from_numpy(depth_bins).float().to(self.device).view(
            1, self.num_depth_bins)
        #self.register_buffer('depth_bins', depth_bins)
        return depth_bins


    def feature_extraction(self, image, is_scales_feats = False, is_freeze_fnet=False):
        """ Run feature extraction on an image, input images already normalized in (0, 1)"""
        # imagenet normalization
        mean_rgb = torch.tensor([0.485, 0.456, 0.406]).float().to(image.device).view(1,3,1,1)
        std_rgb = torch.tensor([0.229, 0.224, 0.225]).float().to(image.device).view(1,3,1,1)
        image = (image - mean_rgb) / std_rgb
        #print ("our normalized image = ", image[0,:,6:10,6:10])
        feats = {}
        if is_freeze_fnet:
            with torch.no_grad():
                tmp_feat_layers = self.feature_extractor(image)
        else:
            tmp_feat_layers = self.feature_extractor(image)

        feat_half, feat_quarter, feat_eighth, feat_sixteenth = self.feature_shrinker(*tmp_feat_layers)
        feats['half'] = feat_half
        if is_scales_feats:
            feats['quarter'] = feat_quarter
            feats['eighth'] = feat_eighth
            feats['sixteenth'] = feat_sixteenth
            feats['normalized_img'] = image
        return feats


    def indices_to_disparity(self, indices, 
            depth_bins ##[N,D], 2-D tensor
            ):
        """Convert cost volume indices to 1/depth for visualization"""
        epsilon = 1e-8 # to avoid divided by 0;
        batch, height, width = indices.shape
        depth = torch.gather(depth_bins, dim=1, index=indices.view(batch, -1))
        # our depth_bin is cuda tensor;
        depth = depth.view(batch, 1, height, width)
        disp = 1.0 / (depth + epsilon) # to avoid divided by 0;
        return disp


    def forward(self, current_image, lookup_images, poses, 
                Ks_src, invK_ref,
                min_depth_bin=None, max_depth_bin=None,
                src_projs = None,
                ref_proj = None,
                is_freeze_fnet=False
                ):
        output= {}
        # feature extraction for ref frame, we return multi-scale features
        # which will be used in the Cost volume regulization;
        #print ("ref feature extraction !!!")
        ref_feats = self.feature_extraction(current_image, is_scales_feats=True,
                                            is_freeze_fnet=is_freeze_fnet)
        #if is_imgnet_norm:
        #    print ("[******??????] feat ext debug")
        #    ref_feats = self.feature_extraction_debug(current_image, is_scales_feats=True)

        if self.adaptive_bins:
            assert min_depth_bin is not None and max_depth_bin is not None, \
                "adaptive_bins=True, requires non-None inputs"
            self.depth_bins = self.compute_depth_bins(min_depth_bin, max_depth_bin, maintain_depth_order=False)

        # feature extraction on lookup images - disable gradients to save memory
        #with torch.no_grad():
        batch_size, num_frames, chns_img, height_img, width_img = lookup_images.shape
        #print("[???] lookup_images = ", lookup_images.shape)
        lookup_images = lookup_images.reshape(batch_size * num_frames, chns_img, height_img, width_img)
        lookup_feat = self.feature_extraction(lookup_images, is_scales_feats=False,
                                is_freeze_fnet=is_freeze_fnet)['half']
        #if is_imgnet_norm:
        #    lookup_feat = self.feature_extraction_debug(lookup_images, is_scales_feats=False)['half']
        _, chns, height, width = lookup_feat.shape
        lookup_feat = lookup_feat.reshape(batch_size, num_frames, chns, height, width)
        #print("[???] lookup_feat = ", lookup_feat.shape)

        #if 0:
        #    print("[???] (our) ref_feat = ", ref_feats['half'].shape)
        #    print("[???] (our) ref_feat = \n", ref_feats['half'])
        #    print("[???] (our) 0 src_feats = ", lookup_feat.shape)
        #    print("[???] (our) 0 src_feat = \n", lookup_feat[:,0])
        depth_bins = self.depth_bins
        depth_bins = depth_bins.repeat(batch_size// depth_bins.size(0), 1).to(
            lookup_feat.device)

        cost_volume, missing_mask = self.my_match_func(
            depth_bins,
            ref_feats['half'], # use feature in hafl image size [H/2,W/2];
            lookup_feats = lookup_feat,
            relative_poses = poses,
            Ks_src = Ks_src,
            invK_ref = invK_ref,
            )
        #print ("Our cost_volume = ", cost_volume.shape, cost_volume[0,-1])


        # for visualisation - ignore 0s in cost volume for minimum
        viz_cost_vol = cost_volume.clone().detach()
        if self.is_dot_product:
            # should choose higt value, not min()
            mins, argbest = torch.max(viz_cost_vol, 1)
        else:
            viz_cost_vol[viz_cost_vol == 0] = 100
            mins, argbest = torch.min(viz_cost_vol, 1)
        lowest_cost = self.indices_to_disparity(argbest, depth_bins)

        output['cost_volume'] = cost_volume
        output['lowest_cost'] = lowest_cost
        output['depth_bins'] = self.depth_bins.detach()
        output['ref_features_list'] = ref_feats
        return output


"""
    Incuding:
    1) Cost Volume Encoder-Decoder: to spatially regularize the raw cost volume
       with a U-Net styel architecture;
    2) Depth regression and refinement
"""
class DVMVS_PairNet_DepthDecoder_2D(SubModule):
    def __init__(self,
                input_channels,
                opt_min_depth, opt_max_depth,
                is_print_info_gpu0 = True,
                pretrain_weights_dir = None,
                use_cost_augmentation = False
                ):

        super(DVMVS_PairNet_DepthDecoder_2D, self).__init__()

        ## Specific to DVMVS_PairNet
        self.num_ch_enc = fpn_output_channels # 32 here;
        self.cost_volume_encoder = CostVolumeEncoder(input_channels)
        self.cost_volume_decoder = CostVolumeDecoder(opt_min_depth, opt_max_depth)
        self.is_print_info_gpu0 = is_print_info_gpu0
        # apply random horizontal flips to the cost volume and the extracted
        # features during pair network training to increase the diversity of
        # the cases that the encoder and the decoder experience;
        self.use_cost_augmentation = use_cost_augmentation

        ##---------------end of DVMVS_PairNet's feature layers
        self.weight_init()
        if pretrain_weights_dir is not None and pretrain_weights_dir != '':
            self.load_pretrained_ckpt(pretrain_weights_dir)



    def load_pretrained_ckpt(self, pretrain_weights_dir):
        # load pretrained checkpoint
        models_name = {
            "cost_volume_encoder": "2_encoder",
            "cost_volume_decoder": "3_decoder",
            }

        models_name2 = {
            "cost_volume_encoder": "module_2_checkpoint.pth",
            "cost_volume_decoder": "module_3_checkpoint.pth",
            }
        
        all_checkpoints = sorted(Path(pretrain_weights_dir).files())
        #print ("[**] Found {} possible ckpts:\n{}".format(len(all_checkpoints), all_checkpoints))
        wanted_ckpts = {}
        for ckpt in all_checkpoints:
            for k, file_name in models_name.items():
                if file_name in ckpt or models_name2[k] in ckpt:
                    wanted_ckpts[k] = ckpt
        
        for k, file_name in models_name.items():
            try:
                #checkpoint = os.path.join(pretrain_weights_dir, file_name)
                checkpoint = wanted_ckpts[k]
                weights = torch.load(checkpoint)
                getattr(self, k).load_state_dict(weights)
                if self.is_print_info_gpu0:
                    print("Our Loaded weights for model {} from ckpt {}".format(k, checkpoint))
            except Exception as e:
                print(e)
                print("[XXXXXXXXXXXXX] Skipping ... model {}".format(k))


    def forward(self, inputs, is_train=False):
        outputs= {}
        feats = inputs['ref_features_list']
        
        # horizontal flips
        flipped = False
        if is_train and self.use_cost_augmentation and random.random() > 0.5:
            flipped = True   
        
        aug_fn = lambda x: torch.flip(x, dims=[-1]) if flipped else x
        skip0, skip1, skip2, skip3, bottom = self.cost_volume_encoder(
            features_half = aug_fn(feats['half']),
            features_quarter= aug_fn(feats['quarter']),
            features_one_eight= aug_fn(feats['eighth']),
            features_one_sixteen= aug_fn(feats['sixteenth']),
            cost_volume = aug_fn(inputs['cost_volume']),
            )
        
        # depth at multiscales;
        dep_full, dep_half, dep_quarter, dep_eighth, dep_sixteenth = self.cost_volume_decoder(
            image = aug_fn(feats['normalized_img']),
            skip0 = skip0,
            skip1 = skip1,
            skip2 = skip2,
            skip3 = skip3,
            bottom = bottom)
        
        if flipped:
            dep_full = aug_fn(dep_full)
            dep_half = aug_fn(dep_half)
            dep_quarter = aug_fn(dep_quarter)
            dep_eighth = aug_fn(dep_eighth)
            dep_sixteenth = aug_fn(dep_sixteenth)
        
        # key: ("depth", frame_id, scale)
        outputs[("depth", 0,  0)] = dep_full.unsqueeze(1)
        outputs[("depth", 0,  1)] = dep_half.unsqueeze(1)
        outputs[("depth", 0,  2)] = dep_quarter.unsqueeze(1)
        outputs[("depth", 0,  3)] = dep_eighth.unsqueeze(1)
        outputs[("depth", 0,  4)] = dep_sixteenth.unsqueeze(1)

        return outputs
