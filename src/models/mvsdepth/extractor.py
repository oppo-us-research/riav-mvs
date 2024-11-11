"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from RAFT-Stereo (https://github.com/princeton-vl/RAFT-Stereo)
# MIT license.
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.model_utils import get_n_downsample
from src.utils.comm import print0

""" load modules from third_parties/RAFT_Stereo """
from third_parties.RAFT_Stereo.core.extractor import BasicEncoder as RAFTBasicEncoder
from third_parties.RAFT_Stereo.core.extractor import MultiBasicEncoder as RAFTMultiBasicEncoder


class BasicEncoder(RAFTBasicEncoder):
    def __init__(self, output_dim=128, norm_fn='batch', 
                dropout=0.0, 
                volume_scale = 'quarter'
                ):
        super(BasicEncoder, self).__init__(
            output_dim = output_dim, 
            norm_fn = norm_fn, 
            dropout = dropout, 
            )
        
        # ----------------------
        # below are new setups;
        # ---------------------- 
        self.volume_scale = volume_scale
        my_strides = {'layer1': 1}
        if self.volume_scale == 'half':
            my_strides['conv1'] = 1
            my_strides['layer2'] = 1
            my_strides['layer3'] = 2
        elif self.volume_scale == 'quarter':
            my_strides['conv1'] = 1
            my_strides['layer2'] = 2
            my_strides['layer3'] = 2
        elif self.volume_scale == 'eighth':
            my_strides['conv1'] = 2
            my_strides['layer2'] = 2
            my_strides['layer3'] = 2
        elif self.volume_scale == 'sixteenth':
            my_strides['conv1'] = 2
            my_strides['layer1'] = 2 # extra one;
            my_strides['layer2'] = 2
            my_strides['layer3'] = 2
        else:
            print ("Wrong scale! Cost volume should be in half, quarter or eighth scale!!!")
            raise NotImplementedError

        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=my_strides['conv1'], padding=3)

        self.in_planes = 64
        print0 ("  ==>  BasicEncoder: volume_scale = {}".format( self.volume_scale))
        self.layer1 = self._make_layer(64,  stride=my_strides['layer1'])
        self.layer2 = self._make_layer(96, stride=my_strides['layer2'])
        self.layer3 = self._make_layer(128, stride=my_strides['layer3'])
        
        #------------------------
        # Weight initialization
        #------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    # no update to forward fucntion 
    # def forward(self, x, dual_inp=False, num_layers=3):

class MultiBasicEncoder(RAFTMultiBasicEncoder):
    def __init__(self, 
                output_dim=[[128,128,128],[128,128,128],], # list of list
                norm_fn='batch', dropout=0.0, 
                volume_scale = 'quarter', 
                ):
        super(MultiBasicEncoder, self).__init__(
            output_dim = output_dim, 
            norm_fn = norm_fn, 
            dropout = dropout
            )
        
        # ----------------------
        # below are new setups;
        # ---------------------- 
        
        self.volume_scale = volume_scale
        n_downsample = get_n_downsample(volume_scale)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (n_downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (n_downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (n_downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    # no update to forward fucntion 
    # def forward(self, x, dual_inp=False, num_layers=3):