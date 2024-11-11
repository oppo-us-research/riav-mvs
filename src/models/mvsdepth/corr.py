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
import torch.nn.functional as F
from src.utils.comm import print0
from src.models.model_utils import bilinear_sampler

class CorrBlock1D(object):
    def __init__(self, cost_volume, num_levels=4, radius=4):
        """ 
        args: 
            cost_volume: MVS plane-sweeping cost volume, with size [N, H, W, D], 
                        here D means number of depth planes, 
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        
        # MVS plane-sweeping cost volume;
        batch, h1, w1, w2 = cost_volume.shape
        # due to F.grid_sample only supports spatial (4-D) and volumetric (5-D) input; 
        corr = cost_volume.reshape(batch*h1*w1, 1, 1, w2)#4D tensor [N,F,H,W]

        for i in range(self.num_levels):
            self.corr_pyramid.append(corr)
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2]) # at each time downsampled by 2;
     
    def __call__(self, coord_1D):
        """ 
        args: 
            corrd_1D: [N, 1, H, W]
        """
        r = self.radius
        coord_1D = coord_1D.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coord_1D.shape
        
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            # values are evenly spaced from start to end, inclusive
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(coord_1D.device) # [1, 1, K, 1], and K=2*r+1;
            x0 = dx + coord_1D.reshape(batch*h1*w1, 1, 1, 1) / 2**i # [N*H*W, 1, K, 1]
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)# [N*H*W, 1, K, 2]
            
            ##Note:
            # input: corr in size [N*H*W, F=1, H', W']
            # grid: coords_lvl in size [N*H*W, 1, K, 2]
            # then output will be in size [N*H*W, F=1, 1, K]
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        
        # [N, H, W, 1*K*L], L=pyramid level, say L=4;
        out = torch.cat(out_pyramid, dim=-1)
        # [N, 1*K*L, H, W]
        return out.permute(0, 3, 1, 2).contiguous().float()

