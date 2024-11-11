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
from typing import Tuple, Optional


class Im2ColLayer(nn.Module):
    def __init__(self, k: int, d: int = 2, is5D: bool = True) -> None:
        """
        Initialize the im2col layer.

        Args:
            k (int): Kernel size.
            d (int, optional): Dilation. Defaults to 2.
            is5D (bool, optional): Flag to indicate if output should be 5D. Defaults to True.
        """
        super(Im2ColLayer, self).__init__()
        self.k = k  # Kernel size
        self.d = d  # Dilation
        self.padding = self.d * (self.k - 1) // 2
        self.im2col = nn.Unfold(kernel_size=self.k, dilation=self.d, padding=self.padding, stride=1)
        self.is5D = is5D

        """
        # NOTE:
        # PyTorch im2col (i.e., nn.Unfold) flattens each k by k
        # block into a column which contains C*(k*k) values, 
        # where k*k is a continuous chunk, with C being the 
        # Channel dimension.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the im2col layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after im2col operation.
        """
        N, C, H, W = x.size()
        if self.is5D:
            # a 5D tensor
            return self.im2col(x).view(N, C, self.k * self.k, H, W)
        else:
            # a 4D tensor
            return self.im2col(x).view(N, C * self.k * self.k, H, W)

def get_coord_feat(img: torch.Tensor) -> torch.Tensor:
    """
    Append (x, y) coordinates to the input image tensor.

    Args:
        img (torch.Tensor): A 4D tensor of shape [B, C, H, W] representing the input image features.

    Returns:
        torch.Tensor: A tensor with shape [B, C + 2, H, W] where (x, y) coordinates are appended
                      to the input image features along the channel dimension.
    """
    # Extract height and width from the input tensor
    _, _, height, width = img.shape

    # Generate coordinate ranges
    x_range = torch.linspace(-1, 1, width, device=img.device)
    y_range = torch.linspace(-1, 1, height, device=img.device)

    # Create coordinate grids
    y, x = torch.meshgrid(y_range, x_range, indexing='ij')
    # older versions of PyTorch w/o arg indexing='ij'
    #y, x = torch.meshgrid(y_range, x_range)

    # Expand coordinates to match the batch size
    bsize = img.shape[0]
    y = y[None,None].repeat(bsize, 1, 1, 1)
    x = x[None,None].repeat(bsize, 1, 1, 1)

    # Concatenate coordinates with the original image features
    coord_feat = torch.cat([img, x, y], dim=1)
    
    return coord_feat

class SSIM(nn.Module):
    """
    Layer to compute the Structural Similarity (SSIM) index between a pair of images.
    
    The SSIM index measures the similarity between two images, considering luminance, contrast, and structure.
    """

    def __init__(self) -> None:
        """
        Initialize the SSIM layer with necessary pooling and padding operations.
        """
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.mu_y_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.sig_x_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.sig_y_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.sig_xy_pool = nn.AvgPool2d(kernel_size=3, stride=1)

        self.refl = nn.ReflectionPad2d(padding=1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute SSIM loss between two images.
        
        Args:
            x (torch.Tensor): The first input image tensor of shape [B, C, H, W].
            y (torch.Tensor): The second input image tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: The SSIM loss tensor of shape [B, C, H, W].
        """
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def disp_to_depth(disp: torch.Tensor, min_depth: float, max_depth: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert the network's sigmoid output into depth prediction.

    Args:
        disp (torch.Tensor): The disparity tensor output from the network with sigmoid activation, typically in the range [0, 1].
        min_depth (float): The minimum depth value for scaling.
        max_depth (float): The maximum depth value for scaling.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - scaled_disp (torch.Tensor): The scaled disparity tensor, converted to the range corresponding to depth values.
            - depth (torch.Tensor): The depth tensor computed from the scaled disparity.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def get_xy_homo_coords(height: int, width: int) -> torch.Tensor:
    """
    Generate homogeneous coordinates for each pixel in a 2D image.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        torch.Tensor: A tensor of shape [3, H*W] containing homogeneous coordinates for each pixel,
                      where the last row consists of ones to represent homogeneous coordinates.
    """
    # Create coordinate grids for x and y
    y, x = torch.meshgrid(
        [torch.arange(0, height), torch.arange(0, width)],
        indexing='ij'
    )
    y, x = y.contiguous(), x.contiguous()
    
    # Flatten the grids and concatenate them with ones
    y = y.view(1, height * width)
    x = x.view(1, height * width)
    xy = torch.cat((x, y, torch.ones_like(x))).float()  # Shape [3, H*W]

    return xy

class BackprojectDepth(nn.Module):
    """
    Layer to transform a depth image into a point cloud.

    This layer converts a depth image into a 3D point cloud using the inverse camera intrinsics matrix.
    """

    def __init__(self, height: int, width: int) -> None:
        """
        Initialize the BackprojectDepth layer.

        Args:
            height (int): The height of the depth image.
            width (int): The width of the depth image.
        """
        super(BackprojectDepth, self).__init__()

        # get homo coords
        pix_coords = get_xy_homo_coords(height, width).unsqueeze(0)  # [N=1, 3, H*W]
        ones = torch.ones(1, height*width).float() # [1, H*W]
        
        # Register buffers to ensure they are part of the module's state
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer 
        # to the state dict (e.g. when we save the model)
        self.register_buffer('pix_coords', pix_coords, persistent=False)
        self.register_buffer('ones', ones, persistent=False)

    def forward(self, depth: torch.Tensor, inv_K: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to transform depth image into point cloud.

        Args:
            depth (torch.Tensor): The depth image tensor of shape [B, 1, H, W].
            inv_K (torch.Tensor): The inverse camera intrinsics matrix of shape [B, 3, 3].

        Returns:
            torch.Tensor: The point cloud tensor of shape [B, 4, H*W].
        """
        
        bs, _, h, w = depth.size()
        
        # here batch_size can also be passed by bs = D = num_depth_bins;
        # So here we use B or D to represent the shapes.
        # Matrix multiplication to get camera points
        # [1, 3, 3] (broadcasting) x [D, 3, H*W] ==> [D, 3, H*W]
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.repeat(bs, 1, 1).to(depth.device))
        
        # Compute the point cloud
        cam_points = depth.view(bs, 1, h * w) * cam_points  # [D, 3, H*W]
        ones_new = self.ones.unsqueeze(0).repeat(bs, 1, 1)  # [D, 1, H*W]
        cam_points = torch.cat([cam_points, ones_new], dim=1)  # [B, 4, H*W]
        
        return cam_points


class Project3D(nn.Module):
    """
    Layer that projects 3D points into a 2D camera plane using intrinsic 
    and extrinsic matrices.

    This layer transforms 3D points into 2D image coordinates given 
    the camera intrinsics and extrinsics.
    """

    def __init__(self, height: int, width: int, eps: float = 1e-6) -> None:
        """
        Initialize the Project3D layer.

        Args:
            height (int): The height of the image.
            width (int): The width of the image.
            eps (float, optional): A small constant to prevent division by zero. Defaults to 1e-6.
        """
        super(Project3D, self).__init__()
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points: torch.Tensor, K: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to project 3D points into 2D image coordinates.
        
        # Here batch_size is actually passed by D = num_depth_bins;
        # So here we use D instead of B to represent the shapes.

        Args:
            points (torch.Tensor): The 3D points tensor of shape [D, 4, H*W].
            K (torch.Tensor): The intrinsic matrix of shape [1, 4, 4].
            E (torch.Tensor): The extrinsic matrix of shape [1, 4, 4].

        Returns:
            torch.Tensor: The 2D pixel coordinates of shape [D, H, W, 2], normalized to the range [-1, 1].
        """
        bs = points.shape[0]
        
        # Compute the projection matrix
        P = torch.matmul(K, E)[:, :3, :]  # Shape [1, 3, 4]
        
        # Project 3D points into the 2D camera plane
        # matrix multiply: [1,3,4](broadcast) x [D,4,H*W] ==> [D, 3, H*W]
        cam_points = torch.matmul(P, points)  # Shape [D, 3, H*W]

        # Normalize the projected points
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2:3, :] + self.eps)
        
        # Reshape and normalize pixel coordinates to [-1, 1]
        pix_coords = pix_coords.view(bs, 2, self.height, self.width)  # Shape [D, 2, H, W]
        pix_coords = pix_coords.permute(0, 2, 3, 1)  # Shape [D, H, W, 2]
        pix_coords[..., 0] /= (self.width - 1)
        pix_coords[..., 1] /= (self.height - 1)
        pix_coords = (pix_coords - 0.5) * 2  # Normalize to [-1, 1], which can be used in grid_sample() func;
        
        return pix_coords


def create_translation_matrix(translation_vec: torch.Tensor) -> torch.Tensor:
    """Convert a translation vector into a 4x4 transformation matrix.

    Args:
        translation_vec: Tensor of shape (batch_size, 3) representing translation vectors.

    Returns:
        A tensor of shape (batch_size, 4, 4) representing the transformation matrices.
    """
    batch_size = translation_vec.shape[0]
    transformation_matrix = torch.zeros(batch_size, 4, 4, device=translation_vec.device)

    translation_vec = translation_vec.contiguous().view(-1, 3, 1)

    transformation_matrix[:, 0, 0] = 1
    transformation_matrix[:, 1, 1] = 1
    transformation_matrix[:, 2, 2] = 1
    transformation_matrix[:, 3, 3] = 1
    transformation_matrix[:, :3, 3, None] = translation_vec

    return transformation_matrix

def rotation_matrix_from_axis_angle(axis_angle_vec: torch.Tensor) -> torch.Tensor:
    """Convert an axis-angle rotation into a 4x4 transformation matrix.

    Args:
        axis_angle_vec: Tensor of shape (batch_size, 1, 3) representing axis-angle vectors.

    Returns:
        A tensor of shape (batch_size, 4, 4) representing the rotation matrices.
    """
    angle = torch.norm(axis_angle_vec, p=2, dim=2, keepdim=True)
    axis = axis_angle_vec / (angle + 1.0e-8)

    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    one_minus_cos = 1 - cos_angle

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    x_sin = x * sin_angle
    y_sin = y * sin_angle
    z_sin = z * sin_angle
    x_cos = x * one_minus_cos
    y_cos = y * one_minus_cos
    z_cos = z * one_minus_cos
    xy_cos = x * y_cos
    yz_cos = y * z_cos
    zx_cos = z * x_cos

    rotation_matrix = torch.zeros((axis_angle_vec.shape[0], 4, 4), device=axis_angle_vec.device)

    rotation_matrix[:, 0, 0] = torch.squeeze(x * x_cos + cos_angle)
    rotation_matrix[:, 0, 1] = torch.squeeze(xy_cos - z_sin)
    rotation_matrix[:, 0, 2] = torch.squeeze(zx_cos + y_sin)
    rotation_matrix[:, 1, 0] = torch.squeeze(xy_cos + z_sin)
    rotation_matrix[:, 1, 1] = torch.squeeze(y * y_cos + cos_angle)
    rotation_matrix[:, 1, 2] = torch.squeeze(yz_cos - x_sin)
    rotation_matrix[:, 2, 0] = torch.squeeze(zx_cos - y_sin)
    rotation_matrix[:, 2, 1] = torch.squeeze(yz_cos + x_sin)
    rotation_matrix[:, 2, 2] = torch.squeeze(z * z_cos + cos_angle)
    rotation_matrix[:, 3, 3] = 1

    return rotation_matrix




def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rotation_matrix_from_axis_angle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = create_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def warp_frame_depth(
    depth_ref: torch.Tensor,
    src_fea: torch.Tensor,
    relative_pose: torch.Tensor,
    K_src: torch.Tensor,
    invK_ref: torch.Tensor,
    is_edge_mask: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warps source feature map using depth map and camera intrinsics and extrinsics.

    Args:
        depth_ref (torch.Tensor): The reference depth map with shape [B, 1, H, W].
        src_fea (torch.Tensor): The source feature map with shape [B, C, H, W].
        relative_pose (torch.Tensor): The relative pose, T^{src}_{ref}, from reference 
                                    frame to source frame with shape [B, 4, 4].
                                    That is, P^{src} = T^{src}_{ref} * P^{ref}.
        K_src (torch.Tensor): The intrinsic matrix of the source frame with shape [B, 4, 4].
        invK_ref (torch.Tensor): The inverse intrinsic matrix of the reference frame with shape [B, 4, 4].
        is_edge_mask (bool, optional): Whether to apply edge masking. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - warped_src_fea (torch.Tensor): The warped source feature map with shape [B, C, H, W].
            - edge_mask (torch.Tensor): The edge mask indicating valid regions with shape [B, 1, H, W].
    """
    batch, channels, height, width = src_fea.shape[:]
    
    if not isinstance(depth_ref, torch.Tensor):
        raise TypeError(f"Input depth_ref type is not a torch.Tensor. Got {type(depth_ref)}.")
    
    if not (depth_ref.ndim == 4 and depth_ref.shape[1] == 1):
        raise ValueError(f"Input depth_ref must have a shape (N, 1, H, W). Got: {depth_ref.shape}")
    

    # Compute the projection matrix
    proj = K_src.bmm(relative_pose).bmm(invK_ref) # [B,4,4]
    

    # Get homogeneous coordinates
    xy = get_xy_homo_coords(height, width).to(depth_ref.device)  # [3, H*W]
    
    # Prepare coordinate tensors
    xy = xy[None, ...].repeat(batch, 1, 1)  # [B, 3, H*W]
    xyz = xy * depth_ref.view(batch, 1, height * width)  # [B, 3, H*W]
    X = torch.cat((xyz, torch.ones(batch, 1, height * width, device=depth_ref.device)), 
                  dim=1)  # [B, 4, H*W]

    # Project 3D points into 2D space
    proj_xyz = torch.matmul(proj, X) #[B,4,4]@[B,4,H*W]=[B,4,H*W]
    proj_xy = proj_xyz[:, :2, :] / (proj_xyz[:, 2:3, :] + 1e-8)  # [B, 2, H*W]
    #check_nan_inf(inp=proj, name='proj')
    #check_nan_inf(inp=proj_xyz, name='proj_xyz')
    #check_nan_inf(inp=proj_xy, name='proj_xy')


    # Normalize to [-1, 1], for grid_sample() func;
    width_normalizer = width / 2.0
    height_normalizer = height / 2.0
    proj_x_normalized = (proj_xy[:, 0, :] - width_normalizer) / width_normalizer  # [B, H*W]
    proj_y_normalized = (proj_xy[:, 1, :] - height_normalizer) / height_normalizer #[B, H*W]

    # Create masks for out-of-bounds values
    X_mask = ((proj_x_normalized > 1) | (proj_x_normalized < -1)).detach()
    proj_x_normalized[X_mask] = 2
    Y_mask = ((proj_y_normalized > 1) | (proj_y_normalized < -1)).detach()
    proj_y_normalized[Y_mask] = 2
    invalid_mask = X_mask | Y_mask  # [B, H*W]
    valid_mask = (~invalid_mask) # [B, H*W]

    grid_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # [B, H*W, 2]

    # Warp the source feature map using the computed grid
    warped_src_fea = F.grid_sample(
        src_fea,
        grid=grid_xy.view(batch, height, width, 2),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, height, width)

    if is_edge_mask:
        # Create edge mask to ignore border areas
        x_vals = proj_xy[:, 0:1, :].detach()  # [N, 1, H*W]
        x_vals = x_vals.view(batch, 1, height, width)
        y_vals = proj_xy[:, 1:2, :].detach() # [N, 1, H*W]
        y_vals = y_vals.view(batch, 1, height, width)
        border_pxls = 2
        edge_mask = valid_mask.view(batch, 1, height, width) * \
                    (x_vals >= border_pxls) * (x_vals <= width - border_pxls) * \
                    (y_vals >= border_pxls) * (y_vals <= height - border_pxls)
        edge_mask = edge_mask.float()  # [N, 1, H, W]
    else:
        edge_mask = valid_mask.view(batch, 1, height, width).float()

    return warped_src_fea, edge_mask


def grid_normlize(proj_xy: torch.Tensor, 
                  width: int, 
                  height: int
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize projection coordinates to the range [-1, 1] for use in grid_sample().

    Args:
        proj_xy (torch.Tensor): The projected coordinates with shape [B, D, 2, H*W].
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - grid_xy (torch.Tensor): The normalized coordinates with shape [B, D, H*W, 2].
            - valid_mask (torch.Tensor): A mask indicating valid pixels with shape [B, D, H, W].
    """
    batch, num_depth = proj_xy.size(0), proj_xy.size(1)
    
    wd_normalizer = width / 2.0
    ht_normalizer = height / 2.0
    
    # Normalize coordinates to range [-1, 1]
    proj_x_normalized = (proj_xy[:, :, 0, :] - wd_normalizer) / wd_normalizer  # [B, D, H*W]
    proj_y_normalized = (proj_xy[:, :, 1, :] - ht_normalizer) / ht_normalizer

    # Create masks for out-of-bounds values
    X_mask = (proj_x_normalized > 1) | (proj_x_normalized < -1).detach()  # [B, D, H*W]
    Y_mask = (proj_y_normalized > 1) | (proj_y_normalized < -1).detach()  # [B, D, H*W]
    
    proj_x_normalized[X_mask] = 2
    proj_y_normalized[Y_mask] = 2
    invalid_mask = X_mask | Y_mask  # [B, D, H*W]
    valid_mask = (~invalid_mask).view(batch, num_depth, height, width)  # [B, D, H, W]
    
    grid_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, D, H*W, 2]
    
    return grid_xy, valid_mask


def homo_warping(depth_bins: torch.Tensor,
                src_fea: torch.Tensor,
                relative_pose: torch.Tensor,
                K_src: torch.Tensor,
                invK_ref: torch.Tensor,
                is_edge_mask: bool = True,
                offset_layer: Optional[torch.nn.Module] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp a frame based on depth and projection parameters.

    Args:
        depth_bins (torch.Tensor): Depth bins with shape [B, D] or [B, D, H, W], where D is the number of depth planes.
        src_fea (torch.Tensor): Source features with shape [B, C, H, W].
        relative_pose (torch.Tensor): Relative pose from reference frame to 
                                     source frame with shape [B, 4, 4]. It maps a point in ref frame 
                                     to a point in src, i.e., P^{src} = T^{src}_{ref} * P^{ref};
        K_src (torch.Tensor): Intrinsic matrix of source frame with shape [B, 4, 4].
        invK_ref (torch.Tensor): Inverse intrinsic matrix of reference frame with shape [B, 4, 4].
        is_edge_mask (bool): Whether to apply an edge mask to avoid out-of-bounds values.
        offset_layer (Optional[torch.nn.Module]): An optional offset layer for refining the pixel coordinates.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - warped_src_fea (torch.Tensor): Warped source features with shape [B, C, D, H, W].
            - edge_mask (torch.Tensor): Mask indicating valid pixels with shape [B, D, H, W].
    """
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    assert depth_bins.dim() in {2, 4}, "Input depth_bins should be a 2D or 4D tensor."
    assert batch == depth_bins.size(0), "Batch size of depth_bins and src_fea should match."
    
    device = src_fea.device
    num_depth = depth_bins.size(1)
    depth_values = depth_bins.view(batch, num_depth, 1, -1).to(device)  # [B, D, 1, H*W]

    height, width = src_fea.shape[2], src_fea.shape[3]
    proj = K_src.bmm(relative_pose).bmm(invK_ref)  # [B, 4, 4]
    xy = get_xy_homo_coords(height, width).to(device)  # [3, H*W]
    xy = xy[None, None, ...].repeat(batch, num_depth, 1, 1)  # [B, D, 3, H*W]
    xyz = xy * depth_values  # [B, D, 3, H*W]
    
    X = torch.cat(
        (
            xyz,
            torch.ones(1, 1, 1, height * width).repeat(batch, num_depth, 1, 1).to(device)
        ), dim=2
    )  # [B, D, 4, H*W]

    proj_xyz = torch.matmul(proj.unsqueeze(1), X)
    proj_xy = proj_xyz[:, :, :2, :] / (proj_xyz[:, :, 2:3, :] + 1e-8)  # [B, D, 2, H*W]
    
    # Normalize grids
    grid_xy, valid_mask = grid_normlize(proj_xy, width, height)

    warped_src_fea = F.grid_sample(
        src_fea,
        grid=grid_xy.view(batch, num_depth * height, width, 2),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
        ).view(batch, channels, num_depth, height, width)

    if offset_layer is not None:
        # Apply offset layer to generate coordinate shifts
        offset = []
        for i in range(num_depth):
            combined = torch.cat([src_fea, warped_src_fea[:, :, i, :, :]], dim=1)  # [N, 2F, H, W]
            offset.append(offset_layer(combined))  # [N, 2, H, W]
        offset_xy = torch.stack(offset, dim=1)  # [N, D, 2, H, W]
        offset_xy = offset_xy.view(batch, num_depth, 2, -1)  # [B, D, 2, H*W]
        proj_xy = proj_xy + offset_xy  # [B, D, 2, H*W]
        
        # Second round of warping with new pixel coordinates
        grid_xy, valid_mask = grid_normlize(proj_xy, width, height)
        warped_src_fea = F.grid_sample(
            src_fea,
            grid=grid_xy.view(batch, num_depth * height, width, 2),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
            ).view(batch, channels, num_depth, height, width)

    if is_edge_mask:
        # Mask out-of-bounds values and near border values
        x_vals = proj_xy[:, :, 0, :].detach()  # [B, D, H*W]
        x_vals = x_vals.view(batch, num_depth, height, width)
        y_vals = proj_xy[:, :, 1, :].detach()
        y_vals = y_vals.view(batch, num_depth, height, width)
        
        edge_mask = valid_mask * (x_vals >= 0.0) * (x_vals <= width - 1) * \
                    (y_vals >= 0.0) * (y_vals <= height - 1)
        edge_mask = edge_mask.float()  # [B, D, H, W]
    else:
        edge_mask = valid_mask.float()

    return warped_src_fea, edge_mask


class match_features_fst(nn.Module):
    """
    Layer which computes a cost volume based on feature concatenation.

    We backwards warp the lookup_feats into the current frame using the
    estimated relative pose, known intrinsics and using hypothesized
    depths self.warp_depths (which are either linear in depth or linear
    in inverse depth).
    """
    def __init__(self, **kwargs):
        super(match_features_fst, self).__init__()
        self.num_ch_enc = kwargs.get('num_ch_enc', None)
        self.matching_height = kwargs.get('matching_height', None)
        self.matching_width = kwargs.get('matching_width', None)
        self.set_missing_to_max = kwargs.get('set_missing_to_max', False)
        self.is_dot_product = kwargs.get("is_dot_product", True)  # Dot product or L1 distance
        self.is_edge_mask = kwargs.get("is_edge_mask", False)  # Mask out invalid warping region
        assert self.is_edge_mask == False, "Currently we do not use self.is_edge_mask!!!"
        
        # Normalization coefficient
        self.scale_same_to_transformer = kwargs.get("scale_same_to_transformer", False)
        self.is_max_corr_pixel_view = kwargs.get("is_max_corr_pixel_view", False)  # Find max or avg correlation among views

    def forward(self, depth_bins: torch.Tensor,
                ref_feat: torch.Tensor,
                lookup_feats: torch.Tensor,
                relative_poses: torch.Tensor,
                Ks_src: torch.Tensor,
                invK_ref: torch.Tensor,
                pixel_wise_net: Optional[nn.Module] = None,
                offset_layer: Optional[nn.Module] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If relative_pose == 0 then this indicates that the lookup frame is missing
        (i.e. we are at the start of a sequence), and so we skip it.

        Note: N is batch size, F is feature channel, V is number of frames or views,
        and H for height, W for width.

        Args:
            depth_bins (torch.Tensor): Depth bins with shape [B, D] or [B, D, H, W], 
                                       where D is the number of depth planes.
            ref_feat (torch.Tensor): Reference features with shape [N, F, H, W].
            lookup_feats (torch.Tensor): Lookup features with shape [N, V, F, H, W].
            relative_poses (torch.Tensor): Relative poses with shape [N, V, 4, 4].
            Ks_src (torch.Tensor): Intrinsic matrices for source frames with shape [N, V, 4, 4].
            invK_ref (torch.Tensor): Inverse intrinsic matrix of reference frame with shape [N, 4, 4].
            pixel_wise_net (Optional[nn.Module]): Optional network for pixel-wise operations.
            offset_layer (Optional[nn.Module]): Optional offset layer for refining pixel coordinates.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - L1_cost_volume (torch.Tensor): Cost volume with shape [N, D, H, W].
                - missing_value_mask (torch.Tensor): Mask indicating missing values with shape [N, D, H, W].
        """
        bat_size, feat_size, height, width = ref_feat.size()
        # cf attention in Transformer;
        scale = feat_size ** -0.5 if self.scale_same_to_transformer else feat_size ** -1.0
        
        num_depth = depth_bins.size(1)
        
        if self.num_ch_enc:
            assert feat_size == self.num_ch_enc, \
                f"ref feature: channel dimension != {self.num_ch_enc}"

        if self.matching_height and self.matching_width:
            assert (height, width) == (self.matching_height, self.matching_width), \
                f"ref feature: HxW != {self.matching_height}x{self.matching_width}"

        # differentiable homography, 
        # Build cost volume
        L1_volume_shape = (bat_size, num_depth, height, width)
        L1_cost_volume = torch.zeros(L1_volume_shape, dtype=torch.float, device=ref_feat.device)
        diffs_all = []

        # Reference cost volume
        ref_cost_volume = ref_feat[:, :, None, ...].repeat([1, 1, num_depth, 1, 1])  # [N, F, D, H, W]

        num_views = lookup_feats.shape[1]
        for lookup_idx in range(num_views):
            lookup_feat = lookup_feats[:, lookup_idx]  # [N, F, H, W]
            lookup_pose = relative_poses[:, lookup_idx]
            K_src = Ks_src[:, lookup_idx]

            # Warp source features
            warped_volume, edge_mask = homo_warping(
                depth_bins=depth_bins,
                src_fea=lookup_feat,
                relative_pose=lookup_pose,
                K_src=K_src,
                invK_ref=invK_ref,
                is_edge_mask=self.is_edge_mask,
                offset_layer=offset_layer
            )
            
            if self.is_dot_product:
                diffs = reduce(scale * warped_volume * ref_cost_volume, 'b c d h w -> b d h w', 'sum')
            else:
                diffs = torch.abs(warped_volume - ref_cost_volume).mean(1)  # [N, D, H, W]

            # Masking of current image
            if self.is_edge_mask:
                diffs = diffs * edge_mask  # [N, D, H, W]
            
            if self.is_max_corr_pixel_view:
                diffs_all.append(diffs)
            else:
                L1_cost_volume = L1_cost_volume + diffs

            del warped_volume
        
        if self.is_max_corr_pixel_view:
            diffs_all = torch.stack(diffs_all, 1)
            L1_cost_volume, _ = torch.max(diffs_all, dim=1, keepdim=False)
        else:
            L1_cost_volume = L1_cost_volume / num_views
        
        # Set missing values to max of existing values
        missing_value_mask = (L1_cost_volume == 0).float()  # [N, D, H, W]
        if self.set_missing_to_max:
            L1_cost_volume = L1_cost_volume * (1 - missing_value_mask) + \
                L1_cost_volume.max(dim=1, keepdim=True)[0] * missing_value_mask

        return L1_cost_volume, missing_value_mask


def down_conv_layer(input_channels, output_channels, kernel_size):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=1,
            bias=False),
        torch.nn.BatchNorm2d(output_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=2,
            bias=False),
        torch.nn.BatchNorm2d(output_channels),
        torch.nn.ReLU())


def up_conv_layer(input_channels, output_channels, kernel_size):
    return torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False),
        torch.nn.BatchNorm2d(output_channels),
        torch.nn.ReLU())


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
