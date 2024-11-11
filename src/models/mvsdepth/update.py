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

""" load modules from third_parties/RAFT_Stereo """
from third_parties.RAFT_Stereo.core.update import (
    FlowHead, ConvGRU, SepConvGRU, 
    pool2x, interp
    )

""" load our own moduels """
from src.models.model_utils import get_n_downsample


""" Just 1D output, for Plance-Sweeping Depth Dimension """
class Flow1DHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(Flow1DHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1) # dim=1 for 1D index;
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ## delta flow 1D
        return self.conv2(self.relu(self.conv1(x)))


""" Updated in RAFT-Stereo: in- and out- planes number; """
class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius, convf1_in_dim):
        super(BasicMotionEncoder, self).__init__()
        # K = 2*r + 1
        # due to 1D indexing, here we use K (instead of K**2 for 2D indexing)
        cor_planes = corr_levels * (2*corr_radius + 1) # K
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(convf1_in_dim, 64, 3, padding=1) # 1D-flow
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv = nn.Conv2d(64+64, 128-convf1_in_dim, 3, padding=1)

    def forward(self, flow, corr):
        #print ("???", flow.shape, corr.shape)
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        #NOTE: always have output dim=128;
        out = torch.cat([out, flow], dim=1)
        return out


class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_levels, corr_radius, 
                hidden_dim, #e.g., == 128;
                head_type, # e.g., == 'depth'; 
                volume_scale, #e.g. == 'quarter'
                you_specify_upscale = None # if set, then disable "volume_scale"
                ):
        
        super(BasicUpdateBlock, self).__init__()
        self.head_type = str(head_type).lower()
        assert self.head_type in ['flow', 'depth', 'flow+depth'], \
            f"Wrong head_type={head_type} found!"
        if you_specify_upscale is None:
            assert volume_scale in ['half', 'quarter', 'eighth', 'sixteenth'], \
                f"cost volume in half, quarter, eighth or sixteenth scale!!!"
        else:
            assert you_specify_upscale in [1,2,4,8]

        if self.head_type == 'flow':
            assert volume_scale == 'eighth', "flowhead: cost volume in eighth scale!!!"
            self.encoder = BasicMotionEncoder(corr_levels, corr_radius, convf1_in_dim=2)
            self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim) # flow
            self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        
        elif self.head_type == 'depth':
            self.encoder = BasicMotionEncoder(corr_levels, corr_radius, convf1_in_dim=1)
            self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim) 
            self.flow_head = Flow1DHead(hidden_dim, hidden_dim=256)
        
        else:
            raise NotImplementedError
        
        if you_specify_upscale is not None:
            factor = you_specify_upscale
        else:
            factor = 2**get_n_downsample(volume_scale)
        #print ("mask: factor = ", factor)
        mask_out_dim = (factor**2)*9  # 9 is for 3x3 window patch;
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_out_dim, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net) # [N, H, W, 2 or 1]
        
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow



class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, corr_levels, corr_radius, 
                hidden_dims = [], head_type = 'depth', 
                volume_scale = 'quarter',
                n_gru_layers = 3 # number of hidden GRU levels;
                ):
        super(BasicMultiUpdateBlock, self).__init__()
        
        self.head_type = str(head_type).lower()
        assert self.head_type in ['flow', 'depth', 'flow+depth'], \
            f"Wrong head_type={head_type} found!"
        assert volume_scale in ['half', 'quarter', 'eighth'], \
            "cost volume in half, quarter or eighth scale!!!"

        self.volume_scale = volume_scale
        factor = 2**get_n_downsample(volume_scale) # 1,2, or 3;
        mask_out_dim = (factor**2)*9 # 9 is for 3x3 window patch;
        self.n_gru_layers = n_gru_layers
        if self.head_type == 'depth':
            encoder_output_dim = 128 # keep this 128, by changing others;
            self.encoder = BasicMotionEncoder(corr_levels, corr_radius, convf1_in_dim=1)
            self.flow_head = Flow1DHead(hidden_dims[2], hidden_dim=256)
            
            self.gru08 = ConvGRU(hidden_dim=hidden_dims[2], 
                                input_dim=encoder_output_dim + hidden_dims[1] * (n_gru_layers > 1))
            self.gru16 = ConvGRU(hidden_dim=hidden_dims[1], 
                                input_dim=hidden_dims[0] * (n_gru_layers == 3) + hidden_dims[2])
            self.gru32 = ConvGRU(hidden_dim=hidden_dims[0], 
                                input_dim=hidden_dims[1])
        else:
            raise NotImplementedError
        
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_out_dim, 1, padding=0))
    
    def forward(self, 
                net,# hidden list: [h8, h16, h32];
                    # h8,h16 and h32 are input hiddens of gru_08/16/32, respectively;     
                inp,# list of list:
                    # [  [cz0, cq0, cr0], for gru_08,
                    #    [cz1, cq1, cr1], for gru_16,
                    #    [cz2, cq2, cr2], for gru_32
                    # ]
                corr=None, flow=None,
                iter08=True, iter16=True, iter32=True, update=True):
        if iter32: # 3 grus
            net[2] = self.gru32(net[2], # hidden h32
                                *(inp[2]), # [cz, cr, cq] for gru_32;
                                # *x_list: downsampling of h16; 
                                pool2x(net[1]) 
                                )
        if iter16:
            if self.n_gru_layers > 2: # 3 grus
                net[1] = self.gru16(
                    net[1], # hidden h16
                    *(inp[1]), # [cz, cr, cq] for gru_16;
                    # *x_list: downsampling of h8 and 
                    # upsampling of gru_32_out (i.e., newly updated h32);
                    pool2x(net[0]), interp(net[2], net[1]) 
                    )
            else: # gru8 + gru16
                net[1] = self.gru16(net[1], *(inp[1]), 
                    # *x_list: downsampling of h8;
                    pool2x(net[0])
                    )
        if iter08:
            motion_features = self.encoder(flow, corr)
            if self.n_gru_layers > 1:
                net[0] = self.gru08(
                    net[0], # hidden h8
                    *(inp[0]), # [cz, cr, cq] for gru_08;
                    # *x_list: motion feature, and 
                    # upsampling of gru_16_out (i.e., newly updated h16);
                    motion_features, interp(net[1], net[0])
                    )
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0]) # [N, H, W, 2 or 1]

        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow