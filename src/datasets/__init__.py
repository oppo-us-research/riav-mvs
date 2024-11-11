"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

from .scannet_npz import MVSDataset as MVS_Scannet
from .scannet_eval import MVSDataset as MVS_Scannet_eval
from .dtu_yao import MVSDataset as MVS_DTU_Yao
from .dtu_yao_eval import MVSDataset as MVS_DTU_Yao_eval
from .seven_scenes_eval import MVSDataset as MVS_7scenes_eval
from .tum_rgbd_eval import MVSDataset as MVS_tumrgbd_eval
from .rgbdscenesv2_eval import MVSDataset as MVS_rgbdscenesv2_eval
