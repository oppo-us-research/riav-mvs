"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from DeepVideoMVS (https://github.com/ardaduz/deep-video-mvs)
# MIT license.
# ------------------------------------------------------------------------------------
# Modified from IterMVS (https://github.com/FangjinhuaWang/IterMVS)
# MIT license.
# ------------------------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

def dMap_to_indxMap(
    dmap, #[N,1, H, W] 
    d_bins, #[N, D],
    is_long_type = False
    ):
    assert d_bins.dim() == 2, "Requires 2D tensor"
    bat_size, ndepth = d_bins.size(0), d_bins.size(1)
    indxMap = []
    for bi in range(bat_size):
        # if right=False, return the first suitable location that is found;
        indxMap.append(torch.bucketize(dmap[bi], d_bins[bi,:], 
                        out_int32= (not is_long_type), right=False))
    indxMap = torch.stack(indxMap, dim=0)
    #indxMap = torch.clamp(indxMap, min=0, max=ndepth-1)
    #print ("??? indxMap = ", indxMap.shape)
    return indxMap

class LossMeter(object):
    def __init__(self):
        self.count = 0.0
        self.sum = 0.0
        self.avg = 0.0

        self.item_average = 0.0

    def update(self, loss, count):
        self.sum += loss
        self.count += count
        self.avg = self.sum / self.count

        self.item_average = loss / count

    def __repr__(self):
        return 'item_avg {:.4f}, avg {:.4f}, sum {:.4f}, count {:.4f}'.format(
            self.item_average, self.avg, self.sum, self.count)


def update_losses(
    predictions, # depth prediction list for different scales or stages;
    weights, # weight list
    groundtruth, valid_mask,
    is_training, l1_meter, l1_inv_meter, l1_rel_meter, loss_type):
    optimizer_loss = .0
    
    if is_training:
        for j, prediction in enumerate(predictions):
            sample_l1_loss, sample_l1_inv_loss, sample_l1_rel_loss, sample_valid_count = \
                calculate_loss_keepSum(groundtruth=groundtruth, prediction=prediction, valid_mask=valid_mask)

            if loss_type == "L1":
                optimizer_loss = optimizer_loss + weights[j] * (sample_l1_loss/sample_valid_count)
            else: # "L1-inv"
                optimizer_loss = optimizer_loss + weights[j] * (sample_l1_inv_loss/sample_valid_count)
    else:
        sample_l1_loss, sample_l1_inv_loss, sample_l1_rel_loss, sample_valid_count = \
            calculate_loss_keepSum(groundtruth=groundtruth, prediction=predictions[-1], # depth at last step;
                                valid_mask=valid_mask)
        # even though loss is not working for validation (i.e., no_grad), 
        # we report this loss still;
        if loss_type == "L1":
            optimizer_loss += weights[-1] * (sample_l1_loss/sample_valid_count)
        else: # "L1-inv"
            optimizer_loss += weights[-1] * (sample_l1_inv_loss/sample_valid_count)
    
    # only save the depth at full resolution to loss meters for tb log; 
    l1_meter.update(sample_l1_loss, sample_valid_count)
    l1_inv_meter.update(sample_l1_inv_loss, sample_valid_count)
    l1_rel_meter.update(sample_l1_rel_loss, sample_valid_count)
    #print ("[???] Updated loss: l1_meter", l1_meter)

    return optimizer_loss


def nll_losses(
    probabilities, # list of depth probability volume
    weights, # weight list
    depth_bins, #[N,D]
    groundtruth, #[N,1,H,W]
    valid_mask, #[N,1,H,W]
    ):
    ## digitize the depth map #
    loss = 0
    # assign invalid depth value to 0, which will be assigned to first bucket;
    # and ignore the bucket index when calculating NLL loss;
    valid_depth_gt = groundtruth.clone()
    valid_depth_gt[~valid_mask] = 0
    indxMap = dMap_to_indxMap(valid_depth_gt, depth_bins, is_long_type=True)
    # resize, due to nll_loss();
    indxMap = rearrange(indxMap, 'n 1 h w -> n h w').contiguous()
    valid_count = valid_mask.float().sum()
    assert len(probabilities) == len(weights)
    #print ("??? indxMap = ", indxMap.shape)
    for j, prob in enumerate(probabilities):
        prob_log = torch.log(torch.clamp(prob, min=1e-5)).contiguous()
        tmp_loss = F.nll_loss(input=prob_log, target=indxMap, ignore_index=0, reduction='sum')
        loss = loss + weights[j]*(tmp_loss/valid_count)
    return loss


def update_raft_losses(
    predictions, # depth prediction list for different scales or stages;
    weights, # weight list
    groundtruth, valid_mask,
    is_training, l1_meter, l1_inv_meter, l1_rel_meter, loss_type,
    ):
    
    optimizer_loss = .0
    unweighted_losses = {
        'l1_abs': [],
        'l1_rel': []
    }

    if is_training:
        for j, prediction in enumerate(predictions):
            sample_l1_loss, sample_l1_inv_loss, sample_l1_rel_loss, sample_valid_count = \
                calculate_loss_keepSum(groundtruth=groundtruth, prediction=prediction, valid_mask=valid_mask)

            if loss_type == "L1":
                tmp_loss = sample_l1_loss/sample_valid_count
            else: # "L1-inv"
                tmp_loss = sample_l1_inv_loss/sample_valid_count
                
            #unweighted_loss_list.append(tmp_loss)
            unweighted_losses['l1_abs'].append(sample_l1_loss/sample_valid_count)
            unweighted_losses['l1_rel'].append(sample_l1_rel_loss/sample_valid_count)
            optimizer_loss = optimizer_loss + weights[j] * tmp_loss
    
    else:
        sample_l1_loss, sample_l1_inv_loss, sample_l1_rel_loss, sample_valid_count = \
            calculate_loss_keepSum(groundtruth=groundtruth, prediction=predictions[-1], # depth at last step;
                                valid_mask=valid_mask)
        # even though loss is not working for validation (i.e., no_grad), 
        # we report this loss still;
        if loss_type == "L1":
            optimizer_loss += weights[-1] * (sample_l1_loss/sample_valid_count)
        else: # "L1-inv"
            optimizer_loss += weights[-1] * (sample_l1_inv_loss/sample_valid_count)
    
    # only save the depth at last iteration step to loss meters for tb log; 
    l1_meter.update(sample_l1_loss, sample_valid_count)
    l1_inv_meter.update(sample_l1_inv_loss, sample_valid_count)
    l1_rel_meter.update(sample_l1_rel_loss, sample_valid_count)

    return optimizer_loss, unweighted_losses




# prediction may have different sclaes
# resize the GT depth to align with the size of prediction;
def calculate_loss_keepSum(
    groundtruth, #[N,1,H,W]
    prediction, #[N,1,H,W]
    valid_mask = None #[N,1,H,W]
    ):
    assert prediction.size(1) == 1 and groundtruth.size(1) == 1
    batch, _, height_scaled, width_scaled = prediction.size()

    groundtruth_scaled = F.interpolate(groundtruth,
                                size=(height_scaled, width_scaled),
                                mode='nearest'
                            )

    if valid_mask is None:
        valid_mask = groundtruth_scaled != 0 # following pairnet training code;
    else:
        valid_mask = nn.functional.interpolate(
            valid_mask.float(),
            size=(height_scaled, width_scaled),
            mode='nearest'
            )
        valid_mask = valid_mask > 0.5 # change to bool tensors for indices;
    valid_count = valid_mask.float().sum()

    groundtruth_valid = groundtruth_scaled[valid_mask]
    prediction_valid = prediction[valid_mask]

    groundtruth_inverse_valid = 1.0 / groundtruth_valid
    prediction_inverse_valid = 1.0 / prediction_valid

    l1_diff = torch.abs(groundtruth_valid - prediction_valid)

    l1_loss = torch.sum(l1_diff)
    l1_inv_loss = torch.sum(torch.abs(groundtruth_inverse_valid - prediction_inverse_valid))
    l1_rel_loss = torch.sum(l1_diff / groundtruth_valid)
    #print ("[???] l1_loss = {}, valid_count = {}".format(l1_loss, valid_count))

    return l1_loss, l1_inv_loss, l1_rel_loss, valid_count

# prediction may have different sclaes
# resize the GT depth to align with the size of prediction;
def calculate_loss(
    groundtruth, #[N,1,H,W]
    prediction, #[N,1,H,W]
    ): 
    assert prediction.size(1) == 1 and groundtruth.size(1) == 1
    batch, _, height_scaled, width_scaled = prediction.size()
    
    groundtruth_scaled = F.interpolate(groundtruth,
                                size=(height_scaled, width_scaled),
                                mode='nearest')
    valid_mask = groundtruth_scaled != 0
    valid_count = valid_mask.nonzero().size()[0]

    groundtruth_valid = groundtruth_scaled[valid_mask]
    prediction_valid = prediction[valid_mask]

    groundtruth_inverse_valid = 1.0 / groundtruth_valid
    prediction_inverse_valid = 1.0 / prediction_valid
    #l1_diff 
    l1_diff = torch.abs(groundtruth_valid - prediction_valid)
    l1_loss = torch.sum(l1_diff) / valid_count
    
    l1_inv_diff = torch.abs(groundtruth_inverse_valid - prediction_inverse_valid)
    l1_inv_loss = torch.sum(l1_inv_diff) / valid_count
    
    l1_rel_loss = torch.sum(l1_diff / groundtruth_valid) / valid_count
    return l1_loss, l1_inv_loss, l1_rel_loss


def calculate_loss_with_mask(
    groundtruth, #[N,1,H,W]
    prediction, #[N,1,H,W]
    valid_mask, #[N,1,H,W]
    ): 
    assert prediction.size(1) == 1 and groundtruth.size(1) == 1
    batch, _, height_scaled, width_scaled = prediction.size()
     
    valid_count = valid_mask.nonzero().size()[0]

    groundtruth_valid = groundtruth[valid_mask]
    prediction_valid = prediction[valid_mask]

    groundtruth_inverse_valid = 1.0 / groundtruth_valid
    prediction_inverse_valid = 1.0 / prediction_valid
    #l1_diff 
    l1_diff = torch.abs(groundtruth_valid - prediction_valid)
    l1_loss = torch.sum(l1_diff) / valid_count
    
    l1_inv_diff = torch.abs(groundtruth_inverse_valid - prediction_inverse_valid)
    l1_inv_loss = torch.sum(l1_inv_diff) / valid_count
    
    l1_rel_loss = torch.sum(l1_diff / groundtruth_valid) / valid_count
    return l1_loss, l1_inv_loss, l1_rel_loss


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, torch.Tensor):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, torch.Tensor):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


# Bin centers regularizer used in AdaBins paper
class BinsChamferLoss(nn.Module):  
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, 
                depth_bins, # [N,D] 
                target_depth_maps,#[N,1,H,W]
                valid_mask, #[N,1,H,W]
               ):
        n, d = depth_bins.shape
        input_pts = rearrange(depth_bins, 'n d -> n d 1')
        h, w = target_depth_maps.shape[2:4]
        valid_count = valid_mask.nonzero().size()[0]
        target_depth_maps[~valid_mask] = 0 # set invalid ones to zero;
        gt_pts = rearrange(target_depth_maps, 'n c h w -> n c (h w)')
        min_bins = depth_bins[:,0].view(n, 1, 1)
        # set invalid depth to depth_bins_min, which will give 0 to Chamfer loss;
        gt_pts = torch.clamp(gt_pts, min=min_bins.expand(-1, -1, h*w))
        # [N,D,1] - [N,1,H*W]
        # bi-directional chamfer loss: direction 1
        diff = (gt_pts - input_pts).abs()
        loss1 = reduce(diff, 'n d l -> n l', 'min') # l = h*w
        loss1 = reduce(loss1, 'n l -> n', 'sum')
        # bi-directional chamfer loss: direction 2
        loss2 = reduce(diff, 'n d l -> n d', 'min')
        loss2 = reduce(loss2, 'n d -> n', 'sum')
        loss = (loss1 + loss2)/ valid_count
        return loss

