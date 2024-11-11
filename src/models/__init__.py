"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""


from .pose_net import ResnetEncoder, PoseDecoder
from .riav_mvs import RIAV_MVS, RIAV_MVS_Eval
from .riav_mvs_cascade import RIAV_MVS_CAS, RIAV_MVS_CAS_Eval

""" baseline methods """
# The model is named following this pattern: 
# xxx: is the model work in the train mode;
# xxx_eval: is the model work in the evaluation mode;
# xxx_pose: baseline + our pose net module, for the ablation study;
# xxx_atten: baseline + our attention module, for the ablation study;

from .bl_pairnet import baseline_pairnet, baseline_pairnet_eval
from .bl_itermvs import (
    baseline_itermvs,
    baseline_itermvs_eval,
    baseline_itermvs_atten,
    baseline_itermvs_atten_eval, 
    baseline_itermvs_pose,
    baseline_itermvs_pose_eval
)

from .bl_mvsnet import (
    baseline_mvsnet,
    baseline_mvsnet_eval,
    baseline_mvsnet_pose,
    baseline_mvsnet_pose_eval,
    baseline_mvsnet_atten,
    baseline_mvsnet_atten_eval
)

from .bl_estdepth import (
    baseline_estdepth,
    baseline_estdepth_eval,
    baseline_estdepth_atten,
    baseline_estdepth_atten_eval
)


""" baseline pipeline """
__models__ = {
    # you can comment those baselines;
    # just used in experiments for comparison
    "bl_pairnet" :  baseline_pairnet,
    "bl_pairnet_eval" :  baseline_pairnet_eval,

    "bl_itermvs" :  baseline_itermvs,
    "bl_itermvs_eval" :  baseline_itermvs_eval,
    "bl_itermvs_pose" :  baseline_itermvs_pose,
    "bl_itermvs_pose_eval" :  baseline_itermvs_pose_eval,
    "bl_itermvs_atten" :  baseline_itermvs_atten,
    "bl_itermvs_atten_eval" :  baseline_itermvs_atten_eval,

    "bl_estdepth" :  baseline_estdepth,
    "bl_estdepth_eval" :  baseline_estdepth_eval,
    "bl_estdepth_atten" :  baseline_estdepth_atten,
    "bl_estdepth_atten_eval" :  baseline_estdepth_atten_eval,

    "bl_mvsnet" :  baseline_mvsnet,
    "bl_mvsnet_eval" :  baseline_mvsnet_eval,
    "bl_mvsnet_pose" :  baseline_mvsnet_pose,
    "bl_mvsnet_pose_eval" :  baseline_mvsnet_pose_eval,
    "bl_mvsnet_atten" :  baseline_mvsnet_atten,
    "bl_mvsnet_atten_eval" :  baseline_mvsnet_atten_eval,


    # main model pipeline:
    # our riav-mvs: including pose module, depth module etc
    "riav_mvs": RIAV_MVS,
    "riav_mvs_eval": RIAV_MVS_Eval,
    "riav_mvs_cas": RIAV_MVS_CAS,
    "riav_mvs_cas_eval": RIAV_MVS_CAS_Eval,
}