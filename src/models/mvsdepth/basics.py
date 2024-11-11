"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from IterMVS (https://github.com/FangjinhuaWang/IterMVS)
# MIT license.
# ------------------------------------------------------------------------------------
# Modified from SeparableFlow (https://github.com/feihuzhang/SeparableFlow)
# MIT license.
# ------------------------------------------------------------------------------------
# Modified from DeepVideoMVS (https://github.com/ardaduz/deep-video-mvs)
# MIT license.
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

""" This version works """
class ProbMaskBlock(torch.nn.Module):
    def __init__(self, down_scale_int, hidden_dim, output_dim):
        super(ProbMaskBlock, self).__init__()

        assert down_scale_int in [1, 2, 3], \
            f"Got {down_scale_int}, Should be in 1/2^i scale, i=1,2,3"
        self.factor = 2**down_scale_int
        self.D = output_dim
        out_num_samples = (self.factor**2)*self.D
        self.prob_mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_num_samples, 1, padding=0, bias=False)
            )

        #--------------------
        # initialization
        #--------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, hidden):
        mask = self.prob_mask(hidden)
        N, _, H, W = mask.shape # [N, 1, H, W]
        mask = mask.view(N, self.D, self.factor, self.factor, H, W)
        mask = mask.permute(0, 4, 2, 5, 3, 1).contiguous() #[N, D, 4, 4, H, W] --> [N, H, 4, W, 4, D]
        mask = mask.view(N, self.factor*H, self.factor*W, self.D) #[N, H', W', D]
        mask = torch.softmax(mask, dim=3)
        return mask

# To mimic SPP module in PSMNet
# For example: fuse 1/2, 1/4, 1/16, 1/32 features to one in 1/4 scale;
class FeatureSPP(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FeatureSPP, self).__init__()
        self.fusion_conv = nn.Sequential(
                    # conv + bn + relu
                    nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1,stride=1, bias=False),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU(inplace=True),
                    # conv only
                    nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                    )

    def forward_full_scale(self, feats):
        """
        Inputs: feats, including:
            feats['half'] = feat_half
            feats['quarter'] = feat_quarter
            feats['eighth'] = feat_eighth
            feats['sixteenth'] = feat_sixteenth
        """
        
        H_half, W_half = feats['half'].size()[-2:]
        H, W = 2*H_half, 2*W_half 
        to_cat_list = []
        for k, v in feats.items():
            out = F.interpolate(v, (H, W), mode='bilinear', align_corners=True)
            to_cat_list.append(out)
        x = torch.cat(to_cat_list, dim=1)
        #print ("[???] concat x = ", x.shape)
        out = self.fusion_conv(x)
        return out

    def forward(self, feats, target_scale='quarter'):
        """
        Inputs: feats, including:
            feats['half'] = feat_half
            feats['quarter'] = feat_quarter
            feats['eighth'] = feat_eighth
            feats['sixteenth'] = feat_sixteenth
        """
        if target_scale == 'full':
            return self.forward_full_scale(feats)
        else:
            target_feat = feats[target_scale]
            H, W = target_feat.size()[-2:]
            to_cat_list = [target_feat]
            for k, v in feats.items():
                if k != target_scale:
                    out = F.interpolate(v, (H, W), mode='bilinear', align_corners=True)
                    to_cat_list.append(out)
            x = torch.cat(to_cat_list, dim=1)
            #print ("[???] concat x = ", x.shape)
            out = self.fusion_conv(x)
            return out


""" Code is adopted from SeparableFlow repository """
class DomainNorm(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=True)
        self.l2 = l2
    def forward(self, x):
        if self.l2:
            x = F.normalize(x, p=2, dim=1)
        x = self.normalize(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, \
                is_3d=False, bn=True, 
                l2=True, 
                relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        self.l2 = l2
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = DomainNorm(channel=out_channels, l2=self.l2)
            #self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x



# deformable serach window
class offset_layer(torch.nn.Module):
    def __init__(self, dim_in, offset_range):
        super(offset_layer, self).__init__()
        self.conv_offset = nn.Conv2d(
                                dim_in, # (ref_feat, warped(src_feat))
                                2, # (offset_x, off_set_y)
                                kernel_size=3, 
                                stride=1, padding=1
                            )
        self.sigmoid = nn.Sigmoid()
        self.offset_range = offset_range
 
    def forward(self, x):
        y = self.conv_offset(x)
        y = 2.0*(self.sigmoid(y) - 0.5) # to range (-1,1)
        y = self.offset_range*y
        return y

""" Code is adopted from IterMVS repository """
class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, stride=stride, 
                              padding=pad, dilation=dilation, 
                              bias=False)

    def forward(self,x):
        return F.relu(self.conv(x), inplace=True)

# estimate pixel-wise view weight
class PixelViewWeight(nn.Module):
    def __init__(self, G):
        super(PixelViewWeight, self).__init__()
        self.conv = nn.Sequential(
            ConvReLU(G, 16),
            nn.Conv2d(16, 1, 1, stride=1, padding=0),
        )

    def forward(self, x):
        # x: [B, G, N, H, W]
        # where G: group #, N: depth planes #;
        batch, dim, num_depth, height, width = x.size()
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(batch*num_depth, dim, height, width) # [B*N,G,H,W]
        x = self.conv(x).view(batch, num_depth, height, width)
        x = torch.softmax(x,dim=1)
        x = torch.max(x, dim=1)[0]

        return x.unsqueeze(1)

class CorrNet(nn.Module):
    def __init__(self, G):
        super(CorrNet, self).__init__()
        self.conv0 = ConvReLU(G, 8)
        self.conv1 = ConvReLU(8, 16, stride=2)
        self.conv2 = ConvReLU(16, 32, stride=2)

        self.conv3 = nn.ConvTranspose2d(32, 16, 3, padding=1, output_padding=1,
                            stride=2, bias=False)

        self.conv4 = nn.ConvTranspose2d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False)

        self.conv5 = nn.Conv2d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        batch, dim, num_depth, height, width = x.size()
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(batch*num_depth, dim, height, width) # [B*N,G,H,W]
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        x = self.conv2(conv1)

        x = conv1 + self.conv3(x)
        del conv1
        x = conv0 + self.conv4(x)
        del conv0

        x = self.conv5(x).view(batch, num_depth, height, width)
        return x

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # He initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

class Conv3x3Block(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels, stride=1, pad=1):
        super(Conv3x3Block, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels, stride, pad)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, stride, pad, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(pad)
        else:
            self.pad = nn.ZeroPad2d(pad)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, stride=stride) # bias=True by default;
        #print ("conv : s=%d, pad=%d"%(stride, pad))

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def conv2d_leakyRelu(ch_in, ch_out, kernel_size, stride, pad, use_bias=True, dilation = 1):
    r'''
    Conv2d + leakyRelu
    '''
    return nn.Sequential( 
            nn.Conv2d(
                ch_in, ch_out, kernel_size=kernel_size, stride = stride,
                padding = dilation if dilation >1 else pad, 
                dilation = dilation, 
                bias= use_bias),
            nn.LeakyReLU()
            )

def convbn_2d_lrelu(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(kernel_size, kernel_size),
                  stride=(stride, stride), padding=(pad, pad), dilation=(dilation, dilation), bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True))


def up2xsample(x, interp_mode='bilinear'):
    """Upsample input tensor by a factor of 2
    """
    if interp_mode =='nearest':
        # align_corners option cannot be set with 'nearest'
        return F.interpolate(x, scale_factor=2, mode=interp_mode)
    else:
        return F.interpolate(x, scale_factor=2, mode=interp_mode, align_corners = True)


class RefineNet_Unet2D(nn.Module):
    '''
    The refinement block based on DemoNet
    This net takes input low res dpv statistics and high res rgb image
    It outputs the high res depth map
    ''' 
    def __init__(self, in_channels, is_depth=False, depth_max = 10, depth_min = 0.1):
        '''
        in_channels - for example, if we use some statistics from DPV, plus the raw rgb input image, then
                      in_channels = 3 + # of statistics we used from DPV
                      Statistics of DPV can includes {expected mean, variance, min_v, max_v etc. }
        '''
        super(RefineNet_Unet2D, self).__init__()

        self.conv0    = Conv3x3Block(in_channels,    out_channels=32,  stride=1, pad=1)
        self.conv0_1  = Conv3x3Block(in_channels=32, out_channels=32,  stride=1, pad=1)

        self.conv1    = Conv3x3Block(in_channels=32, out_channels=64,  stride=2, pad=1)
        self.conv1_1  = Conv3x3Block(in_channels=64, out_channels=64,  stride=1, pad=1)

        self.conv2    = Conv3x3Block(in_channels=64,  out_channels=128, stride=2, pad=1) 
        self.conv2_1  = Conv3x3Block(in_channels=128, out_channels=128,  stride=1, pad=1)
        
        self.up_conv0 = Conv3x3Block(in_channels=192, out_channels=64,  stride=1, pad=1)
        self.up_conv1 = Conv3x3Block(in_channels=96,  out_channels=32,  stride=1, pad=1)

        self.conv3    = Conv3x3Block(in_channels=32, out_channels=16,  stride=1, pad=1) 
        self.conv3_1  = Conv3x3Block(in_channels=16, out_channels=16,  stride=1, pad=1) 
        
        self.disp_conv  = Conv3x3(in_channels=16, out_channels=1, stride=1, pad=1)
        self.sigmoid = nn.Sigmoid()
        self.is_depth = is_depth # if depth or is disparity
        if self.is_depth:
            self.bn_depth = torch.nn.BatchNorm2d(1)
        self.depth_max = depth_max
        self.depth_min = depth_min
        self.apply(self.weight_init)
    
    def forward(self, depth_in, img_in, mono_disp = None):
        '''
        depth_in: [N,1,H,W], depth or disparity, depends on what you feed;
        img_in: NCHW image, normalized one in (0, 1)
        mono_disp: [N,1,H,W], disparity from others, shoud normazlied in (0, 1) range;
        '''
        if self.is_depth:
            # normalize to (0, 1) for easy training, due to img_in are normalized one;
            #depth_in = (depth_in - self.depth_min ) / (self.depth_max - self.depth_min + 1e-8)
            depth_in = self.bn_depth(depth_in)
        if mono_disp is not None:
            # adjust the in_channels accordingly, when defining conv layer;
            conv0_in = torch.cat([depth_in, img_in, mono_disp], dim=1)
        else:
            conv0_in = torch.cat([depth_in, img_in], dim=1)
        
        conv0_out= self.conv0(conv0_in) # 3/32
        conv0_1_out = self.conv0_1(conv0_out) #32/32

        conv1_out = self.conv1(conv0_1_out) #32/64, 1/2 scale;
        conv1_1_out = self.conv1_1(conv1_out) #64/64

        conv2_out = self.conv2(conv1_1_out) # 64/128, 1/4 scale;
        conv2_1_out = self.conv2_1(conv2_out) #128/128

        #import pdb
        #pdb.set_trace()
        
        up_conv0_in = torch.cat([up2xsample(conv2_1_out), conv1_1_out], 1) #192, 1/2 scale;
        up_conv0_out = self.up_conv0(up_conv0_in) #192/64

        up_conv1_in = torch.cat([up2xsample(up_conv0_out), conv0_1_out], 1) #96, 1/1 scale;
        y = self.up_conv1(up_conv1_in) #96/32
        
        y = self.conv3(y) #32/16
        y = self.conv3_1(y) #16/16
        # in (0, 1) range
        y = self.sigmoid(self.disp_conv(y)) # 16/1
        if self.is_depth: # if depth, sacle to (0, max_depth) range;
            y = (self.depth_max-self.depth_min)*y + self.depth_min
        return y

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            #print(' RefineNet_UNet2D: init conv2d')
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            #print(' init Batch2D')
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            #print(' init Linear')
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            #print(' init transposed 2d')
            n = m.kernel_size[1]
            factor = (n+1) // 2
            if n %2 ==1:
                center = factor - 1
            else:
                center = factor -.5 

            og = np.ogrid[:n, :n]
            weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            m.weight.data.copy_(torch.from_numpy(weights_np)) 

class RefinementNet_Residual(SubModule):
    def __init__(self, inplanes):
        super(RefinementNet_Residual, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_2d_lrelu(inplanes, 32, kernel_size=3, stride=1, pad=1),
            convbn_2d_lrelu(32, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            convbn_2d_lrelu(32, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            convbn_2d_lrelu(32, 16, kernel_size=3, stride=1, pad=2, dilation=2),
            convbn_2d_lrelu(16, 16, kernel_size=3, stride=1, pad=4, dilation=4),
            convbn_2d_lrelu(16, 16, kernel_size=3, stride=1, pad=1, dilation=1))

        self.classif1 = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.weight_init()

    def forward(self, input, disparity):
        """
        Refinement Block
        Description:    The network takes left image convolutional features from the second residual block
                        of the feature network and the current disparity estimation as input.
                        It then outputs the finetuned disparity prediction. The low-level feature
                        information serves as a guidance to reduce noise and improve the quality of the final
                        disparity map, especially on sharp boundaries.

        Args:
            :input: Input features composed of left image low-level features, cost-aggregator features, and
                    cost-aggregator disparity.

            :disparity: predicted disparity
        """

        output0 = self.conv1(input)
        output0 = self.classif1(output0)
        output = output0 + disparity
        #output = self.relu(output)
        output = self.sigmoid(output)
        return output


""" Code is adopted from DeepVideoMVS repository """
def conv_layer(input_channels, output_channels, kernel_size, stride, apply_bn_relu):
    if apply_bn_relu:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                bias=False),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True))
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                bias=False))


def depth_layer_3x3(input_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(input_channels, 1, 3, padding=1),
        torch.nn.Sigmoid())

# code adopated from manydepth/baselines/deep_video_mvs/dvmvs/pairnet/model.py;
class UpconvolutionLayer(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(UpconvolutionLayer, self).__init__()
        self.conv = conv_layer(input_channels=input_channels,
                               output_channels=output_channels,
                               stride=1,
                               kernel_size=kernel_size,
                               apply_bn_relu=True)

    def forward(self, inp):
        x = torch.nn.functional.interpolate(input=inp, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x

class DecoderBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, apply_bn_relu, plus_one):
        super(DecoderBlock, self).__init__()
        # Upsample the inpput coming from previous layer
        self.up_convolution = UpconvolutionLayer(
                                        input_channels=input_channels,
                                        output_channels=output_channels,
                                        kernel_size=kernel_size)

        if plus_one:
            next_input_channels = input_channels + 1
        else:
            next_input_channels = input_channels

        # Aggregate skip and upsampled input
        self.convolution1 = conv_layer(input_channels=next_input_channels,
                                       output_channels=output_channels,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       apply_bn_relu=True)

        # Learn from aggregation
        self.convolution2 = conv_layer(input_channels=output_channels,
                                       output_channels=output_channels,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       apply_bn_relu=apply_bn_relu)

    def forward(self, inp, skip, depth):
        inp = self.up_convolution(inp)
        to_combine = [inp]
        if skip is not None:
            to_combine.append(skip)
        if depth is not None:
            depth = torch.nn.functional.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True)
            to_combine.append(depth)

        x = torch.cat(to_combine, dim=1)

        x = self.convolution1(x)
        x = self.convolution2(x)
        return x


class RefineNet_PairNet_8th(SubModule):
    '''
    Eighth: inputs are in 1/8 spatial scale, i.e, [H/8, W/8], and HxW is the input image's dimension;
    The refinement block based on CostVolumeDecoder of PairNet in DeepVideoMVS (CVPR'21)
    This net takes input low res hidden feature from RAFT and high res rgb image
    It outputs the high res depth map
    ''' 
    def __init__(self, 
            hidden_dim=128,
            opt_min_depth=0.1,
            opt_max_depth = 10.0,
            ):
        
        super(RefineNet_PairNet_8th, self).__init__()
        assert hidden_dim % 4 == 0, "hidden_dim must be dividable by 4"
        hyper_channels = (hidden_dim // 4) # e.g, == 32
        #kernel_sizes = [3,5]
        kernel_sizes = [3,3]
        self.epsilon = 1e-8 # to avoid divided by 0;
        
        # See Eq. 4 in DeepVideoMVS paper (CVPR'21)
        self.inverse_depth_base = 1.0 / opt_max_depth
        self.inverse_depth_multiplier = 1.0/opt_min_depth - 1.0/opt_max_depth
        
        self.decoder_block1 = DecoderBlock(input_channels=hyper_channels * 4,
                                           output_channels=hyper_channels * 2,
                                           kernel_size=kernel_sizes[0],
                                           apply_bn_relu=True,
                                           plus_one=True)
        
        # added by CCJ: Upsample the inpput hidden, coming from previous layer
        self.up_conv_h0 = UpconvolutionLayer(input_channels=hyper_channels*4,
                                            output_channels=hyper_channels*2,
                                            kernel_size=kernel_sizes[0])
        
        self.up_conv_h1 = UpconvolutionLayer(input_channels=hyper_channels*2,
                                            output_channels=hyper_channels,
                                            kernel_size=kernel_sizes[1])

        self.decoder_block2 = DecoderBlock(input_channels=hyper_channels * 2,
                                           output_channels=hyper_channels,
                                           kernel_size=kernel_sizes[1],
                                           apply_bn_relu=True,
                                           plus_one=True)
        

        self.refine = torch.nn.Sequential(conv_layer(input_channels=hyper_channels +3+1, # 3 for RGB, 1 for depth;
                                                     output_channels=hyper_channels,
                                                     kernel_size=5,
                                                     stride=1,
                                                     apply_bn_relu=True),
                                          conv_layer(input_channels=hyper_channels,
                                                     output_channels=hyper_channels,
                                                     kernel_size=5,
                                                     stride=1,
                                                     apply_bn_relu=True))

        self.depth_layer_quarter = depth_layer_3x3(hyper_channels * 2)
        self.depth_layer_half = depth_layer_3x3(hyper_channels)
        self.depth_layer_full = depth_layer_3x3(hyper_channels)
        # -------------------
        self.weight_init()
    
    # See Eq. 4 in DeepVideoMVS paper (CVPR'21)
    def disp_to_depth(self, sigmoid_disp):
        inverse_depth = self.inverse_depth_multiplier * sigmoid_disp + self.inverse_depth_base
        depth = 1.0 / (inverse_depth + self.epsilon)
        return depth
    
    def forward(self, image, 
                hidden_inp, #say F=32, [N,4*F,H/8,W/8]
                cnet_inp, #feature dim F=32, [N,4*F,H/8,W/8]
                sigmoid_disp_one_eight, 
                is_return_depth = False # return depth or disparity;
                ):
        # hidden as skip, context feature as input here;
        skip1 = self.up_conv_h0(hidden_inp) #[N,4*F,H/8,W/8] --> [N,2*F,H/4,W/4]
        #[N,4*F,H/8,W/8] --> [N,2*F,H/4,W/4]
        decoder_block1 = self.decoder_block1(inp= cnet_inp, skip= skip1, 
                                            depth= sigmoid_disp_one_eight) 
        sigmoid_disp_quarter = self.depth_layer_quarter(decoder_block1)
        
        # decoder from last layer as input, and upsampled hidden as skip herer;
        skip2 = self.up_conv_h1(skip1) # 2*F,[H,W]/4 scale --> F, 1/2 scale;
        #[N,2*F,H/4,W/4] --> [N,F,H/2,W/2]
        decoder_block2 = self.decoder_block2(inp=decoder_block1, skip=skip2, depth=sigmoid_disp_quarter)
        sigmoid_disp_half = self.depth_layer_half(decoder_block2)
        
        scaled_depth = F.interpolate(sigmoid_disp_half, scale_factor=2, mode='bilinear', align_corners=True)#1
        scaled_decoder = F.interpolate(decoder_block2, scale_factor=2, mode='bilinear', align_corners=True)#2*F
        scaled_combined = torch.cat([scaled_decoder, scaled_depth, image], dim=1)
        scaled_combined = self.refine(scaled_combined)
        sigmoid_disp_full = self.depth_layer_full(scaled_combined)
        if is_return_depth:
            depth_full = self.disp_to_depth(sigmoid_disp_full)
            depth_half = self.disp_to_depth(sigmoid_disp_half)
            depth_quarter = self.disp_to_depth(sigmoid_disp_quarter)
            return depth_full, depth_half, depth_quarter
        else:
            return sigmoid_disp_full, sigmoid_disp_half, sigmoid_disp_quarter

