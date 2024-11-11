"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


""" load modules from third_parties/IterMVS """
from third_parties.IterMVS.models.module import (
    ConvBnReLU, ConvBn
)

""" load our own moduels """
from src.loss_utils import (
    update_losses, LossMeter
)
from src.layers import get_xy_homo_coords


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction = 'mean')
    return loss

def mvsnet_loss_L1(depth_est, depth_gt, mask):
    mask = mask > 0.5
    loss = (depth_est[mask] - depth_gt[mask]).abs().mean()
    return loss

def compute_mvsnet_losses(is_train, depth_est, depth_gt, mask, loss_type):
    # loss accumulator
    l1_meter = LossMeter()
    l1_inv_meter = LossMeter()
    l1_rel_meter = LossMeter()
    weights = [1]
    predictions = [depth_est]
    optimizer_loss = update_losses(
                        predictions=predictions,
                        weights=weights,
                        groundtruth= depth_gt,
                        valid_mask = mask,
                        is_training= is_train,
                        l1_meter = l1_meter,
                        l1_inv_meter = l1_inv_meter,
                        l1_rel_meter = l1_rel_meter,
                        loss_type = loss_type)
    #print ("after updated_losses: l1_meter ", l1_meter)
    return l1_meter, l1_inv_meter, l1_rel_meter, optimizer_loss


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class Depth_Decoder(nn.Module):
    def __init__(self, refine=True):
        super(Depth_Decoder, self).__init__()
        self.cost_regularization = CostRegNet()
        self.refine = refine
        if self.refine:
            self.refine_network = RefineNet()
            #self.refine_network_v2 = RefineNet_v2()
    
    def forward(self, cost_volume, depth_values, ref_frame_quarter=None, refine=False):
        # cost volume regularization
        outputs = {}
        
        cost_reg = self.cost_regularization(cost_volume)

        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1) #[N,1,H,W]
        
        depth = depth_regression(prob_volume, depth_values) #[N,1,H,W]
         
        outputs['depth'] = depth
        outputs['prob_volume'] = prob_volume
        # depth map refinement
        if refine and ref_frame_quarter is not None:
            refined_depth = self.refine_network(ref_frame_quarter, depth)
            outputs["refined_depth"] = refined_depth
        return outputs


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        h, w = depth_init.shape[2], depth_init.shape[3]
        concat = torch.cat((img, depth_init), dim=1)
        #print ("??? concat shape = ", concat.shape)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined

# to follow the original MVSNet paper (and its TensorFlow code);
class RefineNet_v2(nn.Module):
    def __init__(self):
        super(RefineNet_v2, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = nn.Conv2d(32, 1, 3, stride=1, padding=1, bias=True)

    def forward(self, img, depth_init, min_depth, max_depth):
        depth_scale = (max_depth - min_depth + 1.0e-8)
        depth_init_norm = (depth_init - min_depth)/depth_scale
        concat = torch.cat((img, depth_init_norm), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined_norm = depth_init_norm + depth_residual
        depth_refined = depth_refined_norm*depth_scale + min_depth
        return depth_refined

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1

def homography_warp(
    src_features: torch.Tensor, 
    src_projection: torch.Tensor, 
    ref_projection: torch.Tensor, 
    depth_values: torch.Tensor
) -> torch.Tensor:
    """
    Apply homography warping to source features based on depth values and projection matrices.

    Args:
        src_features: Tensor of shape [B, C, H, W] representing source features.
        src_projection: Tensor of shape [B, 4, 4] representing the source projection matrix.
        ref_projection: Tensor of shape [B, 4, 4] representing the reference projection matrix.
        depth_values: Tensor of shape [B, Ndepth] or [1, Ndepth] representing depth values.

    Returns:
        Warped source features of shape [B, C, Ndepth, H, W].
    """
    batch_size, channels = src_features.shape[0], src_features.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_features.shape[2], src_features.shape[3]

    # Compute relative projection matrix
    proj_matrix = torch.matmul(src_projection, torch.inverse(ref_projection))
    rotation = proj_matrix[:, :3, :3]  # [B, 3, 3]
    translation = proj_matrix[:, :3, 3:4]  # [B, 3, 1]

    # Get homogeneous coordinates for the image grid
    xyz_homogeneous = get_xy_homo_coords(height=height, width=width).to(src_features.device)
    xyz_homogeneous = xyz_homogeneous.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, H*W]

    # Apply rotation to the coordinates
    rotated_xyz = torch.matmul(rotation, xyz_homogeneous)  # [B, 3, H*W]
    
    # Multiply rotated coordinates by depth values
    rotated_depth_xyz = rotated_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(
                                batch_size, 1, num_depth, 1)  # [B, 3, Ndepth, H*W]

    # Apply translation
    projected_xyz = rotated_depth_xyz + translation.view(batch_size, 3, 1, 1)  # [B, 3, Ndepth, H*W]

    # Convert to normalized image coordinates
    projected_xy = projected_xyz[:, :2, :, :] / (projected_xyz[:, 2:3, :, :] + 1.0e-8)  # [B, 2, Ndepth, H*W]
    projected_x_normalized = projected_xy[:, 0, :, :] / ((width - 1) / 2) - 1
    projected_y_normalized = projected_xy[:, 1, :, :] / ((height - 1) / 2) - 1
    projected_grid = torch.stack((projected_x_normalized, projected_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]

    # Sample from the source features using the computed grid
    warped_features = F.grid_sample(
        src_features, 
        projected_grid.view(batch_size, num_depth * height, width, 2), 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    # Reshape the warped features to match the output format
    warped_features = warped_features.view(batch_size, channels, num_depth, height, width)

    return warped_features


# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D] or [1,D];
def depth_regression(p, depth_values):
    #print ("//// depth_values shape = ", depth_values.shape)
    N, D = depth_values.shape
    depth = torch.sum(p * depth_values.view(N,D,1,1), dim=1, keepdim=True)#[B,1,H,W]
    return depth


if __name__ == "__main__":
    # some testing code, just IGNORE it
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2

    MVSDataset = find_dataset_def("dtu_yao")
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 256)
    dataloader = DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4].cuda()
    proj_matrices = item["proj_matrices"].cuda()
    mask = item["mask"].cuda()
    depth = item["depth"].cuda()
    depth_values = item["depth_values"].cuda()

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    warped_imgs = homo_warping(src_imgs[0], src_projs[0], ref_proj, depth_values)

    cv2.imwrite('../tmp/ref.png', ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        cv2.imwrite('../tmp/tmp{}.png'.format(i), img_np[:, :, ::-1] * 255)


    # generate gt
    def tocpu(x):
        return x.detach().cpu().numpy().copy()


    ref_img = tocpu(ref_img)[0].transpose([1, 2, 0])
    src_imgs = [tocpu(x)[0].transpose([1, 2, 0]) for x in src_imgs]
    ref_proj_mat = tocpu(ref_proj)[0]
    src_proj_mats = [tocpu(x)[0] for x in src_projs]
    mask = tocpu(mask)[0]
    depth = tocpu(depth)[0]
    depth_values = tocpu(depth_values)[0]

    for i, D in enumerate(depth_values):
        height = ref_img.shape[0]
        width = ref_img.shape[1]
        xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
        print("yy", yy.max(), yy.min())
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        X = np.vstack((xx, yy, np.ones_like(xx)))
        # D = depth.reshape([-1])
        # print("X", "D", X.shape, D.shape)

        X = np.vstack((X * D, np.ones_like(xx)))
        X = np.matmul(np.linalg.inv(ref_proj_mat), X)
        X = np.matmul(src_proj_mats[0], X)
        X /= X[2]
        X = X[:2]

        yy = X[0].reshape([height, width]).astype(np.float32)
        xx = X[1].reshape([height, width]).astype(np.float32)

        warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
        # warped[mask[:, :] < 0.5] = 0

        cv2.imwrite('../tmp/tmp{}_gt.png'.format(i), warped[:, :, ::-1] * 255)
