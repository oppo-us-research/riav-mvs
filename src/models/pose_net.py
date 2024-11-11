"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from torchvision resnet (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
# ------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
from typing import List, Optional, Tuple

class ResNetMultiImageInput(models.ResNet):
    """ ResNet model capable of handling varying numbers of 
        input images or channels.
    """
    def __init__(
        self, 
        block: nn.Module, 
        layers: List[int], 
        num_input_images: int = 1, 
        num_channels: Optional[int] = None
    ) -> None:
        """
        Initializes the ResNetMultiImageInput model.
        
        Args:
            block: Residual block type to be used (BasicBlock or Bottleneck).
            layers: Number of layers for each ResNet stage.
            num_input_images: Number of input images (each contributing 3 channels).
            num_channels: Number of input channels (optional).
        """
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64

        if num_channels is not None:
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def resnet_multiimage_input(
    num_layers: int, 
    pretrained: bool = False, 
    num_input_images: int = 1, 
    num_channels: Optional[int] = None
) -> ResNetMultiImageInput:
    """
    Constructs a ResNet model with multi-image input.
    
    Args:
        num_layers: Number of ResNet layers (18 or 50).
        pretrained: If True, returns a model pre-trained on ImageNet.
        num_input_images: Number of frames stacked as input.
        num_channels: Number of input channels (optional).
    
    Returns:
        A ResNetMultiImageInput model instance.
    """
    assert num_layers in [18, 50], "Only ResNet 18 and 50 are supported."
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, num_channels=num_channels)

    if pretrained:
        resnet_model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            }
        model_weights = model_zoo.load_url(resnet_model_urls[f'resnet{num_layers}'])
        if num_channels is not None:
            # if num_channels cannot divied by 3
            tmp_num_input_imgs = (num_channels + 3 // 2) // 3
            tmp_weight = torch.cat([model_weights['conv1.weight']] * tmp_num_input_imgs, 1) / tmp_num_input_imgs
            # E.g., module.fnet.conv1.weight: torch.Size([64, 3, 7, 7])
            model_weights['conv1.weight'] = tmp_weight[:, :num_channels, :, :].contiguous()
        else:
            model_weights['conv1.weight'] = torch.cat([model_weights['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(model_weights)

    return model


class ResnetEncoder(nn.Module):
    """ResNet-based encoder for feature extraction."""
    
    def __init__(
        self, 
        num_layers: int, 
        pretrained: bool, 
        num_input_images: int = 1, 
        num_channels: Optional[int] = None, 
        **kwargs
    ) -> None:
        """
        Initializes the ResnetEncoder.

        Args:
            num_layers: Number of ResNet layers.
            pretrained: If True, loads pretrained weights.
            num_input_images: Number of input images (optional, default is 1).
            num_channels: Number of input channels (optional).
        """
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        if num_layers not in resnets:
            raise ValueError(f"{num_layers} is not a valid number of ResNet layers.")

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, num_channels)
        else:
            self.encoder = resnets[num_layers](weights="IMAGENET1K_V2" if pretrained else None)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image: torch.Tensor) -> List[torch.Tensor]:
        """Extracts features from the input image using the ResNet encoder."""
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features


class PoseDecoder(nn.Module):
    """Decoder network for predicting pose (rotation and translation) from feature maps."""
    
    def __init__(
        self, 
        num_ch_enc: List[int], 
        num_input_features: int, 
        num_frames_to_predict_for: Optional[int] = None, 
        stride: int = 1
    ) -> None:
        """
        Initializes the PoseDecoder.

        Args:
            num_ch_enc: Number of channels at each encoding layer.
            num_input_features: Number of input feature maps.
            num_frames_to_predict_for: Number of frames to predict the pose for.
            stride: Stride for the convolution layers.
        """
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.squeeze = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.pose_0 = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.pose_1 = nn.Conv2d(256, 256, 3, stride, 1)
        self.pose_2 = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

    def forward(self, input_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict pose (rotation and translation) for each input frame.

        Args:
            input_features: List of feature maps from the encoder.

        Returns:
            axisangle: Tensor of predicted rotation (axis-angle representation).
            translation: Tensor of predicted translation.
        """
        last_features = [f[-1] for f in input_features]
        cat_features = [self.relu(self.squeeze(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = getattr(self, f"pose_{i}")(out)
            if i != 2:
                out = self.relu(out)

        out = out.mean([2, 3])
        # Note: 
        # To predict the rotation using an axis-angle representation,
        # and scale the rotation and translation outputs by 0.01;
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
