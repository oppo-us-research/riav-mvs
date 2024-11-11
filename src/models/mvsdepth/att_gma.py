"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from GMA, ICCV'21 (https://github.com/zacjiang/GMA)
# WTFPL license.
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from CRAFT, CVPR'22 (https://github.com/askerlee/craft)
# WTFPL license.
# ------------------------------------------------------------------------------------

import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange

""" load modules from third_parties/RAFT_Stereo """
from third_parties.RAFT_Stereo.core.update import (
    FlowHead, ConvGRU, SepConvGRU, 
    pool2x, interp
)

""" load our own moduels """
from src.models.mvsdepth.update import BasicMotionEncoder,Flow1DHead
from src.models.model_utils import get_n_downsample
from src.models.mvsdepth.mvs_base import SubModule
from src.utils.comm import print0


class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)
        # torch.arange: [0, max_pos_size)
        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.max_pos_size = max_pos_size
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape

        if self.max_pos_size >= h:
            height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        else:
            # interpolation
            rel_ind_tmp = rearrange(self.rel_ind, 'l1 l2 -> 1 1 l1 l2')
            rel_ind_tmp = F.interpolate(rel_ind_tmp.float(),[h, h], mode="bilinear",
                    align_corners=True) #[1,1,h,h]
            height_emb = self.rel_height(rel_ind_tmp.long().reshape(-1))
            
        if self.max_pos_size >= w:
            width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))
        else:
            # interpolation
            rel_ind_tmp = rearrange(self.rel_ind, 'l1 l2 -> 1 1 l1 l2')
            rel_ind_tmp = F.interpolate(rel_ind_tmp.float(),[w, w], mode="bilinear",
                    align_corners=True) #[1,1,w,w]
            width_emb = self.rel_width(rel_ind_tmp.long().reshape(-1))

        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        return height_score + width_score


# code is adopted from CRAFT (https://github.com/askerlee/craft/blob/main/core/setrans.py;) 
class SlidingPosBiases2D(nn.Module):
    def __init__(self, pos_dim=2, 
                pos_bias_radius=7, 
                max_pos_size=(200, 200)
                ):
        super().__init__()
        self.pos_dim = pos_dim
        self.R = R = pos_bias_radius
        # biases: [15, 15]
        pos_bias_shape = [ pos_bias_radius * 2 + 1 for i in range(pos_dim)]
        self.biases = nn.Parameter(torch.zeros(pos_bias_shape))
        
        # Currently only feature maps with a 2D spatial shape 
        # (i.e., 2D images) are supported.
        if self.pos_dim == 2:
            all_h1s, all_w1s, all_h2s, all_w2s = [], [], [], []
            for i in range(max_pos_size[0]):
                i_h1s, i_w1s, i_h2s, i_w2s = [], [], [], []
                for j in range(max_pos_size[1]):
                    h1s, w1s, h2s, w2s = torch.meshgrid(
                        torch.tensor(i), 
                        torch.tensor(j), 
                        torch.arange(i, i+2*R+1), 
                        torch.arange(j, j+2*R+1)
                        )
                    i_h1s.append(h1s)
                    i_w1s.append(w1s)
                    i_h2s.append(h2s)
                    i_w2s.append(w2s)
                                                  
                i_h1s = torch.cat(i_h1s, dim=1)
                i_w1s = torch.cat(i_w1s, dim=1)
                i_h2s = torch.cat(i_h2s, dim=1)
                i_w2s = torch.cat(i_w2s, dim=1)
                all_h1s.append(i_h1s)
                all_w1s.append(i_w1s)
                all_h2s.append(i_h2s)
                all_w2s.append(i_w2s)
            
            all_h1s = torch.cat(all_h1s, dim=0)
            all_w1s = torch.cat(all_w1s, dim=0)
            all_h2s = torch.cat(all_h2s, dim=0)
            all_w2s = torch.cat(all_w2s, dim=0)
        else:
            #breakpoint()
            raise NotImplementedError

        # Put indices on GPU to speed up. 
        # But if without persistent=False, they will be saved to checkpoints, 
        # making the checkpoints unnecessarily huge.
        self.register_buffer('all_h1s', all_h1s, persistent=False)
        self.register_buffer('all_w1s', all_w1s, persistent=False)
        self.register_buffer('all_h2s', all_h2s, persistent=False)
        self.register_buffer('all_w2s', all_w2s, persistent=False)
        print0(f"Sliding-window Positional Biases, r: {R}, max size: {max_pos_size}")
        
    def forward(self, feat_shape, device):
        R = self.R
        spatial_shape = feat_shape[-self.pos_dim:]
        # [H, W, H, W] => [H+2R, W+2R, H+2R, W+2R].
        padded_pos_shape  = list(spatial_shape) + [ 
            2*R + spatial_shape[i] for i in range(self.pos_dim) 
            ]
        padded_pos_biases = torch.zeros(padded_pos_shape, device=device)
        
        if self.pos_dim == 2:
            H, W = spatial_shape
            all_h1s = self.all_h1s[:H, :W]
            all_w1s = self.all_w1s[:H, :W]
            all_h2s = self.all_h2s[:H, :W]
            all_w2s = self.all_w2s[:H, :W]
            padded_pos_biases[(all_h1s, all_w1s, all_h2s, all_w2s)] = self.biases
                
        # Remove padding. [H+2R, W+2R, H+2R, W+2R] => [H, W, H, W].
        pos_biases = padded_pos_biases[:, :, R:-R, R:-R]
            
        return pos_biases


class Attention(SubModule):
    def __init__(
        self,
        dim,
        max_pos_size = 100,
        heads = 4,
        dim_head = 128,
        position_type = 'content_only',
        **kwargs
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.position_type = position_type
        assert self.position_type in [
            'position_only', 'position_and_content', 
            'content_only',
            'bias',
            ]
        
        if self.position_type == 'position_only':
            self.pos_emb = RelPosEmb(max_pos_size, dim_head)
        elif self.position_type == 'position_and_content':
            self.pos_emb = RelPosEmb(max_pos_size, dim_head)
        elif self.position_type == 'bias':
            pos_bias_radius = kwargs.get('pos_bias_radius', 7)
            self.pos_emb = SlidingPosBiases2D(
                pos_dim = 2, 
                pos_bias_radius = pos_bias_radius
                )
        else:
            self.pos_emb = None
        
        self.pos_embed_weight = 1.0
        
        #------------------------
        # Weight initialization
        #------------------------
        self.weight_init()


    def forward(self, fmap):
        #b, c, h, w = *fmap.shape
        heads = self.heads

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        q = self.scale * q

        if self.position_type == 'position_only':
            sim = self.pos_emb(q)

        elif self.position_type == 'position_and_content':
            sim_content = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
            sim_pos = self.pos_emb(q)
            #E.g., sim_pos=[N,head, H, W, H, W] = [1, 4, 128, 160, 128, 160]
            sim = sim_content + self.pos_embed_weight*sim_pos

        else:
            sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)

        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)

        return attn


class FeatureAggregator(nn.Module):
    """Aggregates features from an attention map and feature map using a multi-head approach.

    This module applies multi-head attention to a feature map, aggregates the results, 
    and optionally projects the aggregated features back to the original dimension.

    Attributes:
        num_heads: Number of attention heads.
        value_conv: Convolutional layer to generate value vectors.
        scale_factor: Learnable scaling factor applied to the output.
        projection_layer: Optional convolutional layer for projecting back to original dimension.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        head_dim: int = 128,
    ) -> None:
        """Initializes the FeatureAggregator.

        Args:
            input_dim: Number of channels in the input feature map.
            num_heads: Number of attention heads.
            head_dim: Dimension of each attention head.
        """
        super().__init__()
        self.num_heads = num_heads
        inner_dim = num_heads * head_dim

        self.value_conv = nn.Conv2d(input_dim, inner_dim, kernel_size=1, bias=False)
        self.scale_factor = nn.Parameter(torch.zeros(1))

        if input_dim != inner_dim:
            self.projection_layer = nn.Conv2d(inner_dim, input_dim, kernel_size=1, bias=False)
        else:
            self.projection_layer = None

    def forward(self, attention_map: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """Forward pass for aggregating features.

        Args:
            attention_map: Attention tensor of shape (batch_size, num_heads, num_queries, num_keys).
            feature_map: Feature map tensor of shape (batch_size, channels, height, width).

        Returns:
            Aggregated feature map tensor of the same shape as the input feature_map.
        """
        num_heads = self.num_heads
        height, width = feature_map.shape[2], feature_map.shape[3]

        value = self.value_conv(feature_map)
        value = rearrange(value, 'b (h d) x y -> b h (x y) d', h=num_heads)
        aggregated_output = einsum('b h i j, b h j d -> b h i d', attention_map, value)
        aggregated_output = rearrange(
                    aggregated_output, 'b h (x y) d -> b (h d) x y', x=height, y=width)
        
        if self.projection_layer is not None:
            aggregated_output = self.projection_layer(aggregated_output)

        output_tensor = feature_map + self.scale_factor * aggregated_output
        #import pdb; pdb.set_trace()
        return output_tensor


# Single GRU layer
class GMAUpdateBlock(SubModule):
    def __init__(self, corr_levels, 
                 corr_radius, hidden_dim=128, 
                 head_type = 'depth', 
                 volume_scale = 'quarter', 
                 num_heads = 1
                ):
        super(GMAUpdateBlock, self).__init__()
        self.head_type = str(head_type).lower()
        assert self.head_type in ['depth', 'flow'], f"Wrong head_type={head_type} found!"
        assert volume_scale in ['half', 'quarter', 'eighth'], "cost volume in half, quarter or eighth scale!!!"

        if self.head_type == 'flow':
            assert volume_scale == 'eighth', "flowhead: cost volume in eighth scale!!!"
            self.encoder = BasicMotionEncoder(corr_levels, corr_radius, convf1_in_dim=2)
            self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+128+hidden_dim) # flow
            self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        
        elif self.head_type == 'depth':
            self.encoder = BasicMotionEncoder(corr_levels, corr_radius, convf1_in_dim=1)
            # + 128: for GMA Aggregate
            self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+128+hidden_dim) #already adjustment for 1D flow
            self.flow_head = Flow1DHead(hidden_dim, hidden_dim=256)
        else:
            raise NotImplementedError

        # attention mechanism
        self.aggregator = FeatureAggregator(input_dim=128, head_dim=128, 
                                            num_heads=num_heads)
        
        factor = 2**get_n_downsample(volume_scale)
        mask_out_dim = (factor**2)*9 # 9 for 3x3 window patch;
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_out_dim, 1, padding=0))

        #------------------------
        # Weight initialization
        #------------------------
        self.weight_init()


    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net) # [N, H, W, 2 or 1]

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow




# 3 GRU layers
class GMAMultiUpdateBlock(SubModule):
    def __init__(self, corr_levels, corr_radius, 
                hidden_dims=[], head_type = 'depth', 
                volume_scale = 'quarter', 
                n_gru_layers = 3, # number of hidden GRU levels;
                num_heads = 1
                ):
        super(GMAMultiUpdateBlock, self).__init__()
        self.head_type = str(head_type).lower()
        assert volume_scale in ['half', 'quarter', 'eighth'], "cost volume in 1/2, 1/4 or 1/8 scale!!!"
        
        if self.head_type == 'depth':
            self.encoder = BasicMotionEncoder(corr_levels, corr_radius, convf1_in_dim=1)
            self.flow_head = Flow1DHead(hidden_dims[2], hidden_dim=256)
            encoder_output_dim = 128
            # 2*encoder_output_dim: has 1 of dim=128 for GMA Aggregate
            self.gru08 = ConvGRU(hidden_dim=hidden_dims[2], 
                                input_dim= 2*encoder_output_dim + hidden_dims[1] * (n_gru_layers > 1))
            self.gru16 = ConvGRU(hidden_dim=hidden_dims[1], 
                                input_dim=hidden_dims[0] * (n_gru_layers == 3) + hidden_dims[2])
            self.gru32 = ConvGRU(hidden_dim=hidden_dims[0], 
                                input_dim=hidden_dims[1])
        else:
            raise NotImplementedError

        # attention mechanism
        self.aggregator = FeatureAggregator(input_dim=128, head_dim=128, num_heads=num_heads)
        
        factor = 2**get_n_downsample(volume_scale)
        mask_out_dim = (factor**2)*9 # 9 for 3x3 window patch;
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_out_dim, 1, padding=0))

        self.volume_scale = volume_scale
        self.n_gru_layers = n_gru_layers
        
        #------------------------
        # Weight initialization
        #------------------------
        self.weight_init()


    def forward(self, 
                net,# hidden list: [h8, h16, h32];
                    # h8,h16 and h32 are input hiddens of gru_08/16/32, respectively;     
                inp,# list of list:
                    # [  [cz0, cq0, cr0], for gru_08,
                    #    [cz1, cq1, cr1], for gru_16,
                    #    [cz2, cq2, cr2], for gru_32
                    # ]
                corr, 
                flow,
                attention,
                iter08 = True, iter16=True, iter32=True
                ):
        
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
            motion_features_global = self.aggregator(attention, motion_features)
            if self.n_gru_layers > 1:
                net[0] = self.gru08(
                    net[0], # hidden h8
                    *(inp[0]), # [cz, cr, cq] for gru_08;
                    # *x_list: motion feature, and 
                    # upsampling of gru_16_out (i.e., newly updated h16);
                    motion_features, motion_features_global, interp(net[1], net[0])
                    )
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, motion_features_global)

        delta_flow = self.flow_head(net[0]) # [N, H, W, 2 or 1]
        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow

