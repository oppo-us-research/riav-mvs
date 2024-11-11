"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""


from . import (
    raft_mvs, 
    raft_mvs_asyatt,
    raft_mvs_sysatt,
    raft_mvs_casBins,
    raft_mvs_gma,
    raft_mvs_adaBins
)

""" submodule: our raft-backbone based MVS depth module """
__mvs_depth_models__ = {
    "raft_mvs": raft_mvs.RAFT_MVS, 
    "raft_mvs_gma": raft_mvs_gma.RAFT_MVS,
    "raft_mvs_asyatt_f1_att": raft_mvs_asyatt.RAFT_MVS,
    "raft_mvs_sysatt_f1_f2_att": raft_mvs_sysatt.RAFT_MVS,
    "raft_mvs_casbins": raft_mvs_casBins.RAFT_MVS,
    "raft_mvs_adabins": raft_mvs_adaBins.RAFT_MVS,
}