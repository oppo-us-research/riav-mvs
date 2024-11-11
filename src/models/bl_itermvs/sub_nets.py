"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""
# ------------------------------------------------------------------------------------
# Modified from IterMVS (https://github.com/FangjinhuaWang/IterMVS)
# MIT license.
# ------------------------------------------------------------------------------------


import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

""" load modules from third_parties/IterMVS """
from third_parties.IterMVS.models.net import Pipeline as Pipeline_base

""" load our own moduels """
from src.models.mvsdepth.att_gma import (
    Attention, FeatureAggregator 
)

class Pipeline(Pipeline_base):
    def __init__(self,  iteration=4, test=False):
        super(Pipeline, self).__init__(iteration=iteration, test=test)

    #-----------------------
    # updated forward 
    #-----------------------
    def forward(self, current_image, lookup_images, proj_matrices, depth_min, depth_max):
        
        combin_img = torch.cat((current_image.unsqueeze(1), lookup_images), dim=1) # [N,V,3,H,W]
        #print (combin_img.shape)
        features = self.feature_net(combin_img)
        ref_feature = {
                    "level3":features['level3'][0],
                    "level2":features['level2'][0],
                    "level1":features['level1'][0],
        }
        src_features = {
                    "level3": [src_fea for src_fea in features['level3'][1:]],
                    "level2": [src_fea for src_fea in features['level2'][1:]],
                    "level1": [src_fea for src_fea in features['level1'][1:]],
        }

        proj_matrices_1 = torch.unbind(proj_matrices['level_1'].float(), 1)
        proj_matrices_2 = torch.unbind(proj_matrices['level_2'].float(), 1)
        proj_matrices_3 = torch.unbind(proj_matrices['level_3'].float(), 1)
        
        ref_proj = {
                "level3": proj_matrices_3[0],
                "level2": proj_matrices_2[0],
                "level1": proj_matrices_1[0]
        }
        
        src_projs = {
                "level3": proj_matrices_3[1:],
                "level2": proj_matrices_2[1:],
                "level1": proj_matrices_1[1:]
        }
        
        depth_min = depth_min.float()
        depth_max = depth_max.float()

        if not self.test:
            depths, depths_upsampled, confidences, confidence_upsampled = self.iter_mvs(ref_feature, src_features,
                        ref_proj, src_projs, depth_min, depth_max)

            return {
                        "depths": depths, 
                        "depths_upsampled": depths_upsampled,
                        "confidences": confidences,
                        "confidence_upsampled": confidence_upsampled,
                    }
        else:
            depth, depths_upsampled, confidence, confidence_upsampled = self.iter_mvs(ref_feature, src_features,
                        ref_proj, src_projs, depth_min, depth_max)

            return {
                        "depths_upsampled": depths_upsampled,
                        "confidence_upsampled": confidence_upsampled,
                    }
        
#------------------------------------
# baseline backbone + attention 
#------------------------------------
class Pipeline_atten(Pipeline_base):
    def __init__(self,  iteration=4, test=False, 
                **kwargs
                ):
        super(Pipeline_atten, self).__init__(
            iteration=iteration, test=test
        )
        
        #---------------------------------
        # newly added methods and variables
        #--------------------------------- 
        self.atten_num_heads = kwargs.get('atten_num_heads', 4)
        self.levels = [1,2,3]
        self.fmap_dims = {
            'level1': 16, # 1/2 resolution
            'level2': 32, # 1/4 res
            'level3': 48,  # 1/8 res
            }
        self.skip_atten_levels = [1]
        # attention mechanism
        for l in self.levels:
            # skip 1/2 scale due to memory limit;
            if l in self.skip_atten_levels:
                setattr(self, f'f1_att_l{l}', None)
                setattr(self, f'f1_aggregator_l{l}', None)
            else:
                setattr(self, f'f1_att_l{l}',
                        Attention(
                            dim= self.fmap_dims[f'level{l}'], 
                            heads= self.atten_num_heads,
                            max_pos_size=160, 
                            dim_head= 128, 
                            #position_type = 'content_only'
                            position_type = 'position_and_content'
                            )
                        )
                setattr(self, f'f1_aggregator_l{l}',
                        FeatureAggregator(
                            input_dim = self.fmap_dims[f'level{l}'], 
                            head_dim =128, 
                            num_heads = self.atten_num_heads
                            )
                        )
        
    
    def forward(self, current_image, lookup_images, proj_matrices, depth_min, depth_max):
        
        combin_img = torch.cat((current_image.unsqueeze(1), lookup_images), dim=1) # [N,V,3,H,W]
        #print (combin_img.shape)
        features = self.feature_net(combin_img)
        ref_feature = {
                    "level3": features['level3'][0],
                    "level2": features['level2'][0],
                    "level1": features['level1'][0],
        }

        # apply attention to ref_feature
        for l in self.levels:
            ref_fea = ref_feature[f'level{l}']
            # attention mechanism to ref feature
            if l in self.skip_atten_levels:
                #print (f"[???] skip level{l}")
                continue
            else:
                attention = getattr(self, f'f1_att_l{l}')(ref_fea)
                ref_feat_global = getattr(self, f'f1_aggregator_l{l}')(attention, ref_fea)
                #print (f"[???] @ level{l} ref_feat_global = {ref_feat_global.shape}")
                ref_feature[f"level{l}"] =  ref_feat_global

        # do not apply attention to src_features
        src_features = {
                    "level3": [src_fea for src_fea in features['level3'][1:]],
                    "level2": [src_fea for src_fea in features['level2'][1:]],
                    "level1": [src_fea for src_fea in features['level1'][1:]],
        }

        proj_matrices_1 = torch.unbind(proj_matrices['level_1'].float(), 1)
        proj_matrices_2 = torch.unbind(proj_matrices['level_2'].float(), 1)
        proj_matrices_3 = torch.unbind(proj_matrices['level_3'].float(), 1)
        
        ref_proj = {
                "level3": proj_matrices_3[0],
                "level2": proj_matrices_2[0],
                "level1": proj_matrices_1[0]
        }
        
        src_projs = {
                "level3": proj_matrices_3[1:],
                "level2": proj_matrices_2[1:],
                "level1": proj_matrices_1[1:]
        }
        
        depth_min = depth_min.float()
        depth_max = depth_max.float()

        if not self.test:
            depths, depths_upsampled, confidences, confidence_upsampled = self.iter_mvs(ref_feature, src_features,
                        ref_proj, src_projs, depth_min, depth_max)

            return {
                        "depths": depths, 
                        "depths_upsampled": depths_upsampled,
                        "confidences": confidences,
                        "confidence_upsampled": confidence_upsampled,
                    }
        else:
            depth, depths_upsampled, confidence, confidence_upsampled = self.iter_mvs(ref_feature, src_features,
                        ref_proj, src_projs, depth_min, depth_max)

            return {
                        "depths_upsampled": depths_upsampled,
                        "confidence_upsampled": confidence_upsampled,
                    }



