"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from DeepVideoMVS (https://github.com/ardaduz/deep-video-mvs)
# MIT license.
# ------------------------------------------------------------------------------------

import torch



""" load modules from third_parties/DeepVideoMVS """
from third_parties.DeepVideoMVS.dvmvs.layers import(
    down_conv_layer,
    up_conv_layer,
    conv_layer,
    depth_layer_3x3
)


def calculate_cost_volume_by_warping(
        image1, image2, pose1, pose2, K, warp_grid, 
        min_depth, max_depth, n_depth_levels, 
        device, dot_product, 
        relative_pose = None
    ):
    """
    # Pay attention here: Scannet itself provides camera-to-world pose (not extrinsic matrix!),
    # so before passing to this function, make sure your pose already be extrinsic matrix !!!
    # pose1: Nx4x4, reference frame, world_to_camera pose (i.e., extrinsic matrix!)
    # pose2: Nx4x4, source frame, world_to_camera pose (i.e., extrinsic matrix!)
    # relative_pose = pose2.bmm(torch.inverse(pose1))
    # K: Nx4x4
    """

    batch_size, channels, height, width = image1.size()
    warp_grid = torch.cat(batch_size * [warp_grid.unsqueeze(dim=0)])

    cost_volume = torch.empty(size=(batch_size, n_depth_levels, height, width), dtype=torch.float32).to(device)
    if relative_pose is not None:
        extrinsic2 = relative_pose
    else:
        extrinsic2 = torch.inverse(pose2).bmm(pose1)
    
    #print ("[???] extrinsic2 \n", extrinsic2)
    R = extrinsic2[:, 0:3, 0:3]
    t = extrinsic2[:, 0:3, 3].unsqueeze(-1)
    #print ("[???] K, t ", K, ", " , t)
    
    Kt = K.bmm(t)
    K_R_Kinv = K.bmm(R).bmm(torch.inverse(K))
    K_R_Kinv_UV = K_R_Kinv.bmm(warp_grid)

    inverse_depth_base = 1.0 / max_depth
    inverse_depth_step = (1.0 / min_depth - 1.0 / max_depth) / (n_depth_levels - 1)

    width_normalizer = width / 2.0
    height_normalizer = height / 2.0

    depth_bins = []
    for depth_i in range(n_depth_levels):
        this_depth = 1 / (inverse_depth_base + depth_i * inverse_depth_step)
        depth_bins.append(this_depth)
    
    #print ("their dpeths = ", depth_bins)
    for depth_i in range(n_depth_levels):
        this_depth = 1 / (inverse_depth_base + depth_i * inverse_depth_step)

        warping = K_R_Kinv_UV + (Kt / this_depth)
        warping = warping.transpose(dim0=1, dim1=2)
        warping = warping[:, :, 0:2] / (warping[:, :, 2].unsqueeze(-1) + 1e-8)
        warping = warping.view(batch_size, height, width, 2)
        warping[:, :, :, 0] = (warping[:, :, :, 0] - width_normalizer) / width_normalizer
        warping[:, :, :, 1] = (warping[:, :, :, 1] - height_normalizer) / height_normalizer
        warped_image2 = torch.nn.functional.grid_sample(input=image2,
                                                        grid=warping,
                                                        mode='bilinear',
                                                        padding_mode='zeros',
                                                        align_corners=True)
        #if depth_i == 0:
        #    print ("[???******] depth={}, warping idx_x = \n{}".format(this_depth, \
        #        warping[...,0]))
        #    print ("[???] warped_volume = ", warped_image2.shape)
        #    print ("[???] warped_volume = ", warped_image2[0,0])

        if dot_product:
            cost_volume[:, depth_i, :, :] = torch.sum(image1 * warped_image2, dim=1) / channels
        else:
            cost_volume[:, depth_i, :, :] = torch.sum(torch.abs(image1 - warped_image2), dim=1)

    return cost_volume

def cost_volume_fusion(image1, image2s, pose1, pose2s, K, warp_grid, min_depth, \
    max_depth, n_depth_levels, device, dot_product, 
    relative_poses = None
    ):
    batch_size, channels, height, width = image1.size()
    fused_cost_volume = torch.zeros(size=(batch_size, n_depth_levels, height, width), dtype=torch.float32).to(device)
    
    if relative_poses is None:
        relative_poses = len(pose2s)*[None]
    else:
        pose2s = len(relative_poses)*[None]
    #for pose2, image2 in zip(pose2s, image2s):
    for pose2, image2, relative_pose in zip(pose2s, image2s, relative_poses):
        cost_volume = calculate_cost_volume_by_warping(
                                image1=image1,
                                image2=image2,
                                pose1=pose1,
                                pose2=pose2,
                                K=K,
                                warp_grid=warp_grid,
                                min_depth=min_depth,
                                max_depth=max_depth,
                                n_depth_levels=n_depth_levels,
                                device=device,
                                dot_product=dot_product,
                                relative_pose = relative_pose
                            )
        fused_cost_volume += cost_volume
    fused_cost_volume /= len(pose2s)
    
    return fused_cost_volume
