"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

""" updated for 1D flow """
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    #print ("H,W = ", H, W)
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    # This is a stereo problem
    assert torch.unique(ygrid).numel() == 1 and H == 1 

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def updepth(depth, scale, mode='bilinear'):
    new_size = (scale * depth.shape[2], scale * depth.shape[3])
    return  F.interpolate(depth, size=new_size, mode=mode, align_corners=True)

def upflow(flow, scale, value_scale, mode='bilinear', ):
    new_size = (scale * flow.shape[2], scale * flow.shape[3])
    return  value_scale*F.interpolate(flow, size=new_size, mode=mode, 
                                      align_corners=True)

""" integer coordinates """
def coords_grid(batch, ht, wd):
    # [start, end)
    x_range = torch.arange(start=0, end=wd, step=1)
    y_range = torch.arange(start=0, end=ht, step=1)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij') 
    coords = torch.stack((grid_x, grid_y), dim=0).float() # in size [2, H, W];
    return coords[None].expand(batch, -1, -1, -1) # in size [N, 2, H, W]

""" coordinates are normalized to [-1, 1] """
def coords_grid_normlized(batch, ht, wd, device):
    # [start, end]
    x_range = torch.linspace(-1, 1, wd, device=device)
    y_range = torch.linspace(-1, 1, ht, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij') 
    coords = torch.stack((grid_x, grid_y), dim=-1).float() # in size [H, W, 2];
    return coords[None].expand(batch, -1, -1, -1) # in size [N, 2, H, W]
    

def get_n_downsample(volume_scale):
    n_downsample_dict = {
        'half': 1, # 1/2^1
        'quarter': 2, # 1/2^2
        'eighth': 3, # 1/2^3
        'sixteenth': 4 # 1/2^4, 
    }
    return n_downsample_dict[volume_scale]

def depth_normalization(depth, inverse_depth_min, inverse_depth_max):
    '''convert depth map to the index in inverse range'''
    inverse_depth = 1.0 / (depth+1e-5)
    normalized_depth = (inverse_depth - inverse_depth_max) / (inverse_depth_min - inverse_depth_max)
    return normalized_depth

def depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max):
    '''convert the index in inverse range to depth map'''
    inverse_depth = inverse_depth_max + normalized_depth * (inverse_depth_min - inverse_depth_max) # [B,1,H,W]
    depth = 1.0 / inverse_depth
    return depth