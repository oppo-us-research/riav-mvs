"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import os
import sys
import math
from path import Path

""" load our own moduels """
from .basics import SubModule, FeatureSPP, BasicConv
from .update import BasicUpdateBlock, BasicMultiUpdateBlock
from src.models.model_utils import bilinear_sampler
from src.loss_utils import dMap_to_indxMap
from .corr import CorrBlock1D
from .extractor import BasicEncoder, MultiBasicEncoder
from src.layers import match_features_fst
from src.utils.comm import print0
from src.models.bl_pairnet.dvmvs_pairnet import (
    FeatureExtractor, FeatureShrinker)
from .depth_bins import SCANNET_DEPTH_BINS

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

"""
Our base module: MVSDepth, with Plane-Sweeping Stereo design;
"""
class MVSDepth(SubModule):
    def __init__(self,
                input_height, 
                input_width, 
                mode, # 'train', 'resume', 'test'
                min_depth_bin,
                max_depth_bin,
                num_depth_bins,
                adaptive_bins,
                depth_binning, # 'linear' or 'inverse'(i.e., disparity)
                **kwargs
                ):
        super(MVSDepth, self).__init__()
        
        #------------------------
        # get values from kwargs 
        #------------------------
        self.is_verbose = kwargs.get('is_verbose', True)
        self.is_mixed_precision = kwargs.get('is_mixed_precision', False)
        if self.is_verbose:
            print0 ("[xxx] self.is_mixed_precision = ", self.is_mixed_precision)
        self.fnet_norm_fn = kwargs.get('fnet_norm_fn', 'instance')
        self.cnet_norm_fn = kwargs.get('cnet_norm_fn', 'batch')
        self.is_gt_flow_1D = kwargs.get('is_gt_flow_1D', False)
        if self.is_gt_flow_1D and self.is_verbose:
            print0 ("[**] is_gt_flow_1D = True, Using GT flow_1D " + \
                    "(i.e., frontal-parallel depth planes index) " + \
                    "for deep supervision !!!")
        self.freeze_raft_net = kwargs.get('freeze_raft_net', False)
        if self.is_verbose:
            print0 ("[***] self.freeze_raft_net = ", self.freeze_raft_net)

         
        self.raft_volume_scale = kwargs.get('raft_volume_scale', 'eighth')
        self.down_scale_int = {
            "sixteenth": 16, "eighth": 8, 'quarter': 4, 
            'half': 2,
            'full': 1,
            }[self.raft_volume_scale]
        if self.is_verbose:
            print0 ("[***]self.raft_volume_scale={}, self.down_scale_int={}".format(
                self.raft_volume_scale, self.down_scale_int))
        
        self.slow_fast_gru = kwargs.get('slow_fast_gru', False)
        self.n_gru_layers = kwargs.get('n_gru_layers', 1) # number of hidden GRU levels;
        self.is_multi_gru = (self.n_gru_layers > 1)
        self.is_max_corr_pixel_view = kwargs.get('is_max_corr_pixel_view', False)
        

        self.fnet_name = kwargs.get('fnet_name', 'raft_fnet')
        self.feature_fusion = None
        if self.fnet_name == 'raft_fnet':          
            assert self.fnet_norm_fn  == 'instance', 'Wrong fnet_norm_fn type'
            if self.is_verbose:
                print0 (f"[***] BasicEncoder fnet_norm_fn={self.fnet_norm_fn}")
            self.fmap_dim = 256
            self.fnet = BasicEncoder(output_dim=self.fmap_dim, 
                norm_fn=self.fnet_norm_fn, dropout= 0,
                volume_scale = self.raft_volume_scale
                )
        
        elif self.fnet_name == 'pairnet_fnet':
            self.feature_extractor = FeatureExtractor()
            self.feature_shrinker = FeatureShrinker()
            self.is_fusion_feats = kwargs.get('fusion_pairnet_feats', False)
            self.fmap_dim = 32
            if self.is_fusion_feats:
                # 4 scales: 1/2, 1/4, 1/8, and 1/16;
                self.feature_fusion = FeatureSPP(in_planes=4*self.fmap_dim, 
                                                out_planes=self.fmap_dim
                                                )

        assert self.cnet_norm_fn in ['batch', 'none'], 'Wrong cnet_norm_fn type'
        if self.is_verbose:
            print0 (f"[***] BasicEncoder cnet_norm_fn={self.cnet_norm_fn}")
        
        if self.is_multi_gru:
            if self.is_verbose:
                print0 ("### multi gru layers !!!")
            
            self.hidden_dims = hidden_dims = [128]*3
            self.context_dims = context_dims = hidden_dims
            self.corr_levels = 4
            self.corr_radius = 4
            
            self.cnet = MultiBasicEncoder(
                    output_dim=[hidden_dims, context_dims], # list of list;
                    norm_fn=self.cnet_norm_fn, 
                    volume_scale = self.raft_volume_scale
                    )
            self.update_block = BasicMultiUpdateBlock(
                self.corr_levels, self.corr_radius, 
                hidden_dims = hidden_dims,
                head_type ='depth',
                volume_scale = self.raft_volume_scale,
                n_gru_layers = self.n_gru_layers # number of hidden GRU levels;
                )
            self.context_zqr_convs = nn.ModuleList(
                # hidden_dims[i]*3 for cz, zr, and cq;
                [nn.Conv2d(context_dims[i], hidden_dims[i]*3, 3, padding=3//2) for i in range(self.n_gru_layers)])
            
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_levels = 4
            self.corr_radius = 4
            
            # context network
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn=self.cnet_norm_fn, dropout= 0, \
                track_norm_running_stats=self.track_norm_running_stats,
                volume_scale = self.raft_volume_scale
                )
            # update block
            self.update_block = BasicUpdateBlock(
                self.corr_levels, self.corr_radius, hidden_dim=hdim,
                head_type='depth',
                volume_scale = self.raft_volume_scale
                )
        
        self.is_training = (mode in ['train', 'resume'])
        #self.device = device
        
        # "correlation volume implementation"
        self.corr_block = CorrBlock1D
         
        # base model: we set no refine_net;
        self.refine_net_type = 'none'
        self.rfnet = None
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)  
        self.epsilon = 1e-8 # to avoid divided by 0;
        
        # ------ mvs depth module init -----
        self.raft_depth_init_type = kwargs.get('raft_depth_init_type', 'none')
        if self.is_verbose:
            print0 ("[***] self.raft_depth_init_type = ", self.raft_depth_init_type)
        if self.raft_depth_init_type in ['none', 'argmin', 'soft-argmin']:
            self.cost_agg = None
        elif self.raft_depth_init_type == 'soft-argmin-3dcnn':
            # NOTE: do not make sense, 3d cnn ???
            self.cost_agg = BasicConv(
                    in_channels= 1, 
                    out_channels= 1, 
                    deconv=False, 
                    is_3d=True,
                    bn=True, relu=True,
                    kernel_size=3,
                    stride=1, padding=1,
                )
        elif self.raft_depth_init_type == 'soft-argmin-2dcnn':
            self.cost_agg = BasicConv(
                    in_channels= num_depth_bins, 
                    out_channels= num_depth_bins, 
                    deconv=False, 
                    is_3d=False,
                    bn=True, relu=True,
                    l2= False,
                    kernel_size=3,
                    stride=1, padding=1,
                )
        else:
            raise NotImplementedError
        
        #------------------------
        # Weight initialization
        #------------------------
        self.weight_init()
        
        
        #---------------------------------------------
        #----- MVS Plane-Sweeping to get Cost Volume
        #---------------------------------------------
        self.depth_binning = depth_binning # 'linear' or 'inverse'(i.e., disparity);
        self.num_depth_bins = num_depth_bins
        # dense depth bins: e.g., 64 --> 64*4
        self.num_depth_bins_up = min(num_depth_bins*self.down_scale_int, 256)
        self.adaptive_bins = adaptive_bins
        self.opt_min_depth_bin = min_depth_bin
        self.opt_max_depth_bin = max_depth_bin
        
        # we build the cost volume at a lower resolution than the input images;
        self.matching_height = input_height // self.down_scale_int
        self.matching_width = input_width // self.down_scale_int
        if self.is_verbose:
            print0 ("[***] matching height_width= {}x{}".format(
                self.matching_height, self.matching_width))
        
        if not self.adaptive_bins:
            #[N, D] 
            depth_bins = self.compute_depth_bins(
                min_depth_bin, max_depth_bin, self.num_depth_bins)
            #[N, D_up] 
            depth_bins_up  = self.compute_depth_bins(
                min_depth_bin, max_depth_bin, self.num_depth_bins_up
                )
            #Register them to be part of the module's state
            # Used for tensors that need to be on the same device as the module.
            # persistent=False tells PyTorch to not add the buffer 
            # to the state dict (e.g. when we save the model)
            self.register_buffer('depth_bins', depth_bins, persistent=False)
            self.register_buffer('depth_bins_up', depth_bins_up, persistent=False)
            if self.is_verbose:
                print0 ("  ==> depth_bins: ", self.depth_binning, \
                    self.depth_bins.shape, self.depth_bins.view(-1))
                print0 ("  ==> depth_bins_up: ",self.depth_bins_up.shape, 
                    self.depth_bins_up.view(-1))
        else:
            # will be generated dynamically later given 
            # min_depth_bin and max_depth_bin;
            if self.is_verbose:
                print0 (" using daptive_bins, will be geneated dynamically later!")
            self.depth_bins = None
            self.depth_bins_up = None
        
        #assert self.depth_bins is not None, "depth_bins Cannot be None"



        self.is_dot_product = True
        match_kwargs = {
                'is_training': self.is_training,
                'num_ch_enc': self.fmap_dim,
                'matching_height': self.matching_height,
                'matching_width': self.matching_width,
                'is_dot_product': self.is_dot_product, # dot product, not L1 distance;
                'set_missing_to_max' : False, # Do not set missing cost volume to its max_values
                'is_edge_mask': False, # do not consider pixels warped out of boundary;
                'is_max_corr_pixel_view': self.is_max_corr_pixel_view, 
                }
        self.my_match_func = match_features_fst(**match_kwargs)
 
    
    # assuming min_depth_bin and max_depth_bin is the same for all the pixels 
    # in the current image; then we can smaple the bins by 1-D grid along 
    # the frontal-parallel depth planes direction;
    # otherwise, we have to sample the bins by 3-D grid, with (x,y,z) for 
    # (width, height, depth), i.e., the pixel (x, y) has depth bin index z;
    def compute_depth_bins(self, min_depth_bin, max_depth_bin, 
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
    

    """ fixed (i.e., not dynamically changed) depth bins for plane sweep stereo """
    # that is, depth bins are constant for all the pixels among images;
    def compute_depth_bins_fixed(self, min_depth_bin, 
                    max_depth_bin, num_depth_bins, 
                    maintain_depth_order=True):
        """
        Compute the depths bins used to build the cost volume. 
        Bins will depend upon self.depth_binning, to either 
        be linear in depth (linear) or linear in inverse depth
        (inverse)
        """

        if self.depth_binning == 'inverse':
            depth_bins = 1 / np.linspace(1 / max_depth_bin,
                                         1 / min_depth_bin,
                                        num_depth_bins
                                        )

            if maintain_depth_order:
                depth_bins = depth_bins[::-1] # maintain depth order
                depth_bins = np.ascontiguousarray(depth_bins)

        elif self.depth_binning == 'linear':
            depth_bins = np.linspace(min_depth_bin, max_depth_bin, num_depth_bins)
        
        elif self.depth_binning == 'merged':
            assert num_depth_bins == 96, "requires num_depth_bins=96"
            depth_bins = [float(i) for i in SCANNET_DEPTH_BINS]
            depth_bins = np.array(depth_bins).astype(np.float32)
            assert len(depth_bins) == num_depth_bins
            assert (max_depth_bin == depth_bins[-1]) and \
                (min_depth_bin == depth_bins[0])
        else:
            raise NotImplementedError
        depth_bins = torch.from_numpy(depth_bins).float()
        return depth_bins
    
    
    
    def indices_to_disparity_hard(self, indices, 
                                  depth_bins #[N,D], 2-D tensor
        ):
        """Convert cost volume indices to 1/depth for visualization"""
        batch, height, width = indices.shape
        #print0 ("[???] depth_bins = ", depth_bins.shape)
        depth = torch.gather(depth_bins, dim=1, index=indices.view(batch, -1))
        #print0 ("[???] indices = ", indices.shape, " depth = ", depth.shape)
        depth = depth.view(batch, 1, height, width)
        disp = 1.0 / (depth + self.epsilon) # to avoid divided by 0;
        return disp

    def indices_to_depth(self, 
            indices, depth_bins, #[N,D], 2-D tensor 
            maintain_depth_order = True
        ):
        """ Convert cost volume indices to depth by grid_sample"""
        indices = indices.permute(0, 2, 3, 1).contiguous() #to [N,H,W,1]
        batch, h1, w1, _ = indices.shape
        
        # due to F.grid_sample only supports 
        # spatial (4-D) and volumetric (5-D) input; 
        depth_bins = depth_bins.view(batch,1,1, -1) #[N,1,1,D]
        
        # grid: [N, H_out, W_out, 2]
        # here H_out=1, W_out=1;
        x0 = indices # [batch, h1, w1, 1]
        y0 = torch.zeros_like(x0)
        grid = torch.cat([x0,y0], dim=-1)# [N, H, W, 2]
        
        #grid_sample: output will have shape [N, C, H_out, W_out]
        depth = bilinear_sampler(depth_bins, grid) #[N, C=1, h1, w1]
        #print ("??? depth = ", depth.shape)
        if maintain_depth_order:
            depth_min = depth_bins[:,:,:,0:1]
            depth_max = depth_bins[:,:,:,-1:]
        else:
            depth_max = depth_bins[:,:,:,0:1]
            depth_min = depth_bins[:,:,:,-1:]
        
        depth = torch.clip(depth, min= depth_min, max= depth_max)
        return depth
 
    
    def get_raft_cost_idx_init_hard(self, cost_volume, outputs, depth_bins):
        """ 
        Args: 
            cost_volume: [N,D,H,W]
            depth_bins: [N,D]
        """
        # generate arg-min depth for visualization before 
        # the following cost aggregation;
        tmp_cost_vol = cost_volume.detach()
        if self.is_dot_product:
            # should choose high value, not min()
            _, argbest = torch.max(tmp_cost_vol, dim=1)
        else:
            tmp_cost_vol[tmp_cost_vol == 0] = 100 # set to a large enough value;
            _, argbest = torch.min(tmp_cost_vol, dim=1)
        
        # get depth by taking argmin(cost);
        outputs['lowest_cost'] = self.indices_to_disparity_hard(argbest, depth_bins)

        # generate soft-arg-min depth for visualization 
        # before the following cost aggregation;
        softargmin_depth = torch.sum(
            F.softmax(cost_volume, dim=1)*depth_bins.view(-1,self.num_depth_bins,1,1), 
            dim = 1, keepdim=True
            ) # [N, 1, H, W]
        outputs['softargmin_depth'] = softargmin_depth


        if self.raft_depth_init_type == 'none':
            cost_idx_init = None
         
        elif self.raft_depth_init_type in [
            'soft-argmin', 'soft-argmin-3dcnn', 'soft-argmin-2dcnn'
            ]:
            cost_idx_init = dMap_to_indxMap(softargmin_depth, depth_bins) # long type
        
        # in case in Long type;
        if cost_idx_init is not None:
            cost_idx_init = cost_idx_init.float()
        
        return cost_idx_init
    
    def get_raft_cost_idx_init(self, cost_volume, depth_bins):
        """
        Args:
            cost_volume: [N,D,H,W]
            depth_bins: [N,D]
        """
        num_depth_bins = depth_bins.size(1)
        if depth_bins.dim() == 2:
            depth_bins = rearrange(depth_bins, 'b d -> b d 1 1')
        prob = F.softmax(cost_volume, dim=1)
        softargmin_depth = reduce(prob*depth_bins, 'b d h w -> b 1 h w', 'sum')
        if self.raft_depth_init_type == 'none':
            cost_idx_init = None

        elif self.raft_depth_init_type == 'soft-argmin':
            ## soft-agrmin index;
            idx_values = torch.arange(num_depth_bins,
                                device=cost_volume.device,
                                dtype=torch.float).view(1,num_depth_bins,1,1)
            depth_idx = reduce(prob*idx_values, 'b d h w -> b 1 h w', 'sum')
            depth_idx = depth_idx.clamp(min=0, max=num_depth_bins-1)
            cost_idx_init = depth_idx
        else:
            raise NotImplementedError

        return cost_idx_init, softargmin_depth

    
    def load_pretrained_raft_encoder(self, raft_weights_path):
        if self.is_verbose:
            print0 (f"[***] MVSDepth try to load pretrained ckpt from {raft_weights_path}")
        if os.path.isfile(raft_weights_path):
            #my_pretrained_dict = torch.load(raft_weights_path, map_location=self.device)
            my_pretrained_dict = torch.load(raft_weights_path)
            if 'state_dict' in my_pretrained_dict:
                # e.g., 'module.feature_extraction.firstconv.0.0.weight'
                my_pretrained_dict = my_pretrained_dict['state_dict'] 

            """ fnet """
            #name_list = ['fnet', 'cnet', 'update_block']
            name_list = ['fnet']
            for name in name_list:
                if self.is_verbose:
                    print0 ("\t Trying {} ...".format(name))
                # E.g., change "module.fnet.conv1.weight" to "conv1.weight"
                pretrained_feat_dict = {
                    k[len(f'module.{name}.'):]: v for k,v in my_pretrained_dict.items() \
                        if f'module.{name}' in k }
                getattr(self, name).load_state_dict(pretrained_feat_dict, strict=True)
            if self.is_verbose:
                print0 ("\t ckpt loading done!")
        else:
            if self.is_verbose:
                print0 ("[!!] Not a valid .pth, skip the ckpt loading ..." )

    def load_pretrained_pairnet_ckpt(self, pretrain_weights_dir):
        # load pretrained checkpoint
        models_name = {
            "feature_extractor": "0_feature_extractor", 
            "feature_shrinker":  "1_feature_pyramid",
            }
        all_checkpoints = sorted(Path(pretrain_weights_dir).files())
        wanted_ckpts = {}
        for ckpt in all_checkpoints:
            for k, file_name in models_name.items():
                if file_name in ckpt:
                    wanted_ckpts[k] = ckpt
        
        for k in models_name:
            try:
                #checkpoint = os.path.join(pretrain_weights_dir, file_name)
                checkpoint = wanted_ckpts[k]
                weights = torch.load(checkpoint)
                getattr(self, k).load_state_dict(weights)
                if self.is_verbose:
                    print0("Loaded weights for model {} from ckpt {}".format(k, checkpoint))
            except Exception as e:
                if self.is_verbose:
                    print0(e)
                    print0("[XXXXXXXXXXXXX] Skipping ... model {}".format(k))
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
 
    def upsample_flow1d(self, flow1d, mask, channel_last=False):
        if channel_last:
            flow1d = flow1d.permute(0, 3, 1, 2).contiguous().float() #[N,1,H,W]
        """ Upsample depth [H/8, W/8, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = flow1d.shape # [N, 1, H, W]
        mask = mask.view(N, 9, self.down_scale_int, self.down_scale_int, H, W)
        mask = torch.softmax(mask, dim=1)
        scale = 1.0*self.num_depth_bins_up / self.num_depth_bins
        up_flow1d = F.unfold(scale*flow1d, [3,3], padding=1)
        up_flow1d = up_flow1d.view(N, 9, 1, 1, H, W)

        up_flow1d = torch.sum(mask * up_flow1d, dim=1) # [N, 9, 8, 8, H, W] --> [N, 8, 8, H, W]
        up_flow1d = up_flow1d.permute(0, 3, 1, 4, 2) #[N, 8, 8, H, W] --> [N, H, 8, W, 8]
        up_flow1d = up_flow1d.reshape(N, 1, self.down_scale_int*H, self.down_scale_int*W)
        return up_flow1d
    
    def image_normlization(self, 
                           image, # its values have been in range (0, 1)
                           is_imgnet_normalization
                           ):
        """ Run feature extraction on an image, input images already 
            normalized in (0, 1)"""
        #NOTE: make sure the input `image` has been normalized to range (0, 1),
        # E.g., it is loaded from dataloader using transforms.ToTensor() etc.
        
        if is_imgnet_normalization:
            # since we will use pretrained model;
            mean_rgb = torch.tensor([0.485, 0.456, 0.406]).float().to(image.device).view(1,3,1,1)
            std_rgb = torch.tensor([0.229, 0.224, 0.225]).float().to(image.device).view(1,3,1,1)
            image = (image - mean_rgb) / std_rgb

        else: # RAFT style normalization
            image = 2*image - 1.0 # to [-1, 1]
        
        return image.contiguous()

    def feature_extraction_raft_fnet(self, image):
        image = self.image_normlization(image, is_imgnet_normalization=False)
        #print0 ("our normalized image = ", image[0,:,6:10,6:10], image.device)
        with autocast(enabled=self.is_mixed_precision):
            feat = self.fnet(image)
            return feat
    
    
    def feature_extraction_pairnet(self, image):
        # imagenet normalization
        image = self.image_normlization(image, is_imgnet_normalization=True)
        #print0 ("our normalized image = ", image[0,:,6:10,6:10], image.device)
        #print0 ("image ", image.device, "feature ", self.feature_extractor.parameters())
        feats = {}
        feat_half, feat_quarter, feat_eighth, feat_sixteenth = \
            self.feature_shrinker(*self.feature_extractor(image))
        feats['half'] = feat_half
        feats['quarter'] = feat_quarter
        feats['eighth'] = feat_eighth
        feats['sixteenth'] = feat_sixteenth
        return feats
 

    def feature_extraction(self, image):
        raise NotImplementedError
    
    def context_extraction(self, image):
        raise NotImplementedError
    
    def context_extraction_single_gru(self, image):
        image = self.image_normlization(image, is_imgnet_normalization=False)
        
        with autocast(enabled=self.is_mixed_precision):
            cnet = self.cnet(image)
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net) # hidden state
            inp = torch.relu(inp) # context feature
            return net, inp
    
    def context_extraction_multi_gru(self, image, return_raw_context_inp = False):
        image = self.image_normlization(image, is_imgnet_normalization=False)
        
        with autocast(enabled=self.is_mixed_precision):
            cnet_list = self.cnet(image, num_layers=self.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list_raw = inp_list
            # Rather than running the GRU's conv layers on the context features 
            # multiple times, we do it once at the beginning 
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) \
                for i,conv in zip(inp_list, self.context_zqr_convs)]
            
            #NOTE:
            # inp_list looks like: (list of list)
            # [
            #    [cz0, cq0, cr0], for gru_08,
            #    [cz1, cq1, cr1], for gru_16,
            #    [cz2, cq2, cr2], for gru_32
            # ]
            if return_raw_context_inp:
                return net_list, inp_list, inp_list_raw
            else:
                return net_list, inp_list
     
    
    # run RAFT-backbone for MVS depth estimation;
    def run_raft_depth(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError