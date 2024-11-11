"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import sys
import time
from datetime import datetime
import warnings
import json
import torch
import torch.distributed as dist

# this project related #
from src.utils.comm import is_main_process, synchronize, print0

class Config(object):
    
    def __init__(self, options):

        self.opt = options

        """ DDP related configuration """
        if self.opt.gpu_id >= 0:
            warnings.warn(
                '[***] You have chosen a specific GPU. This will completely '
                'disable data parallelism.')

        if not torch.cuda.is_available():
            print('[***] using CPU, this will be slow')

        self.num_node = self.opt.num_node
        
        if self.opt.multiprocessing_distributed and self.opt.dist_url == "env://":
            num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
            self.WORLD_SIZE = num_gpus
            self.ngpus_per_node = num_gpus // self.num_node
            # NOTE:
            # `torchrun` provides a superset of the functionality as `torch.distributed.launch` 
            # with the following additional functionalities:
            # - Worker failures are handled gracefully by restarting all workers.
            # - Worker RANK and WORLD_SIZE are assigned automatically.
            # - Number of nodes is allowed to change between minimum and maximum sizes (elasticity).
            
            # If your training script reads local rank from a --local_rank cmd argument. 
            # Change your training script to read from the LOCAL_RANK environment variable 
            # as `local_rank = int(os.environ["LOCAL_RANK"])`;
            self.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        else:
            self.ngpus_per_node = torch.cuda.device_count()
            num_gpus = self.ngpus_per_node * self.num_node
            self.WORLD_SIZE = num_gpus
            self.LOCAL_RANK = self.opt.local_rank
        
        print0 ("[***] num_gpus = ", num_gpus) 
        self.DISTRIBUTED = (num_gpus >= 1) and self.opt.multiprocessing_distributed
        print0 ("[***] self.DISTRIBUTED = ", self.DISTRIBUTED) 
        if self.DISTRIBUTED:
            local_rank = self.LOCAL_RANK
            local_world_size = self.WORLD_SIZE
            n = torch.cuda.device_count() // local_world_size
            device_ids = list(range(local_rank * n, (local_rank + 1) * n))
            dist.init_process_group(
                #backend="nccl", init_method="env://"
                backend = self.opt.dist_backend, 
                init_method = self.opt.dist_url,
               )
            torch.cuda.set_device(local_rank)
            
            print(
                f"[{os.getpid()}] rank = {dist.get_rank()}, " +
                f"world_size = {dist.get_world_size()}, device_ids = {device_ids}"
                )
            print0(
                f'Config: number of gpus: {num_gpus}, ' + 
                f'number of nodes: {self.num_node}, ' +
                f'world_size (i.e., gpus_per_node * node_num): {self.WORLD_SIZE}'
                )
            synchronize()

        if not self.opt.seed:
            self.opt.seed = datetime.now()

        
        # this batch_size will not be adjusted w.r.t #GPUs;
        self.batch_size_orig = options.batch_size
        
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        self.BATCH_SIZE = int(self.batch_size_orig / self.ngpus_per_node)
        self.NUM_WORKERS = int((self.opt.num_workers + self.ngpus_per_node - 1) / self.ngpus_per_node)
        self.SUMMARY_FREQ = self.opt.log_frequency
        self.PRINT_FREQ = self.opt.print_freq
        #self.SAVE_CKPT_FREQ = self.opt.save_ckpt_freq


        # If true, do validation after one epoch training
        self.opt.train_validate = True
        #self.opt.train_validate = False

        self.network_class_name = options.network_class_name

        """ raft backbone related """
        # 'normalization function type for fnet in RAFT: 
        # e.g., 'group', 'instance'
        self.opt.raft_fnet_norm_fn = "instance"
        self.opt.raft_cnet_norm_fn = "batch"

        """ baseline iter-mvs related """
        self.opt.itermvs_iters = 4
        #self.opt.adaptive_bins = True # dynamic depth range if True;
        

        #'track_norm_running_stats or not for Batch Normalization'
        self.opt.track_norm_running_stats = True
        self.raft_mvs_type = options.raft_mvs_type
        self.opt.is_resnet50_raft = False
        self.opt.radius_atten_raft = 2
        self.opt.is_key_attention_raft = False
        self.opt.raft_disp_nonlinear_func = 'sigmoid'
        
        # for the feature extractor, we use 
        # raft_fnet (from RAFT, ECCV 2020) 
        # or pairnet_fnet (from DeepVideoMVS, CVPR 2021);
        assert self.opt.fnet_name in ['raft_fnet', 'pairnet_fnet']

        assert self.opt.n_gru_layers in [1,3]
        # consider a window, with half size = radius
        self.opt.raft_mvs_pss_prob_radius = 4
        
        # use GT optical flow for deep supervision to each RAFT iteration
        self.opt.raft_gt_flow = False
        # if raft_gt_flow=True, use this hyper-parameter weight to
        # balance flow loss and depth loss;
        # if this weight < 0, means to disable this flow loss;
        self.opt.flow_loss_weight = -1

        #If true, average the raft correlation among reference frame
        # vs different source views.
        # Otherwise, will be taking max, not average
        self.opt.is_avg_corr_raft = True
        self.opt.refine_net_type = None
        #'iteration number for depth/flow update'
        self.raft_iters = options.raft_iters
        # use customized cuda Correlation Block CorrBlock
        self.opt.is_raft_alternate_corr = False
        self.opt.is_raft_mixed_precision = False # 'use mixed precision'


        """ training configuration """
        self.train_image_width = options.width
        self.train_image_height = options.height

        self.train_min_depth = options.min_depth
        self.train_max_depth = options.max_depth
        self.train_n_depth_levels = options.num_depth_bins

        self.opt.train_minimum_pose_distance = 0.125
        self.opt.train_maximum_pose_distance = 0.325
        self.opt.train_crawl_step = 3
        self.opt.train_subsequence_length = None
        self.opt.train_predict_two_way = True
        self.opt.use_cost_augmentation = True
        self.opt.train_freeze_batch_normalization = False
        self.train_data_pipeline_workers = options.num_workers

        self.train_epochs = options.num_epochs
        self.dataset_type = options.dataset


        #self.opt.train_seed = int(round(time.time()))
        self.opt.train_seed = 1234


        self.opt.scales ={
            'riav_mvs': [0],
            'bl_pairnet': [0],
            'bl_itermvs': [0,1,2,3],
            'bl_estdepth': [0,1,2,3],
            'bl_mvsnet': [0,1,2], # return depth at 1/4 scale;
        }[self.network_class_name] # just use one scale first;

        self.opt.scannet_sub_test_type = None
        self.opt.six_digit = True

        self.opt.adaptive_bins = False # dynamic depth range if True;
        
        """ get dataset info """
        if self.dataset_type == 'scannet_mea1_npz_sml':
            self.data_dir = "datasets_link/Scannet/scans_train_npz/"
            self.dataset_str = 'scanpz1'
            self.opt.split='scannetv2_small' # set manually;
            self.split_str = {
                'scannetv2_small' : 'sml',
                'scannetv2': '' # defualt
            }[self.opt.split]
            self.opt.mvsdata_nviews= 2 # 1 ref frame + 1 source frames
            self.opt.frame_ids=[0,1] # only 1 source frame

            self.opt.eval_valid_depth_min = 0.25 # meter
            self.opt.eval_valid_depth_max = 20 # meter

        elif self.dataset_type == 'scannet_mea1_npz':
            self.data_dir = "datasets_link/Scannet/scans_train_npz/"
            self.dataset_str = 'scanpz1'
            self.opt.split='scannetv2' # set manually;
            self.split_str = {
                'scannetv2_small' : 'sml',
                'scannetv2': '' # defualt
            }[self.opt.split]
            self.opt.mvsdata_nviews= 2 # 1 ref frame + 1 source frames
            self.opt.frame_ids=[0,1] # only 1 source frame

            self.opt.eval_valid_depth_min = 0.25 # meter
            self.opt.eval_valid_depth_max = 20 # meter

        elif self.dataset_type == 'scannet_mea2_npz':
            self.data_dir = "datasets_link/Scannet/scans_train_npz/"
            
            self.dataset_str = 'scanpz2'
            #self.opt.split='scannetv2' # set manually;
            self.opt.split='scannetv2_small' # for debugging only;
            self.split_str = {
                'scannetv2_small' : 'sml',
                'scannetv2': '' # defualt
            }[self.opt.split]
            self.opt.mvsdata_nviews= 3 # 1 ref frame + 2 source frames
            self.opt.frame_ids=[0,1,2] # 1 ref frame + 2 source frames

            self.opt.eval_valid_depth_min = 0.25 # meter
            self.opt.eval_valid_depth_max = 20 # meter

        # scannet test set: toy set
        elif self.dataset_type == 'scannet_n3_eval_sml':
            self.data_dir = "datasets_link/Scannet/scans_test/"
            self.opt.mvsdata_nviews= 3 # 1 ref frame + 2 source frames
            self.opt.frame_ids=[0,1,2] # only 1 source frame
            self.dataset_str = f'scantstn{self.opt.mvsdata_nviews}sml'
            self.opt.split='scannetv2_small' # set manually;
            self.split_str = {
                'scannetv2_small' : 'sml',
                'scannetv2': '' # defualt
            }[self.opt.split]
            txt_file_dir = f'splits/{self.opt.split}/test'
            filename_txt_list = ['test_files_iter_00.txt']
            self.opt.filename_txt_list = [
                os.path.join(txt_file_dir, i) for i in filename_txt_list
                ]
            self.opt.scannet_sub_test_type = None

            #self.opt.test_image_width = 320
            #self.opt.test_image_height = 256
            self.opt.eval_valid_depth_min = 0.25 # meter
            self.opt.eval_valid_depth_max = 20 # meter
        
        # scannet test set
        elif self.dataset_type == 'scannet_n3_eval':
            self.data_dir = "datasets_link/Scannet/scans_test/"
            self.opt.mvsdata_nviews= 3 # 1 ref frame + 2 source frames
            self.opt.frame_ids=[0,1,2] # only 1 source frame
            self.dataset_str = f'scantstn{self.opt.mvsdata_nviews}'
            scannet_sub_test_type = self.opt.scannet_eval_sampling_type
            txt_file_subdir = {
                "e-s10n3" : "test_iters_estdepth",
                "d-kyn3" : "test_deepvideo_keyframe-nmeas2",
                "d-s10n3": "test_deepvideo_simple10-nmeas2",

                # "sml validation set for quick model selection"
                "val-3k": "val3k_deepvideo_keyframe-nmeas2", 
                "val-all": "val_deepvideo_keyframe-nmeas2", 
                }
            if scannet_sub_test_type in ["val-3k", "val-all"]:
                self.data_dir = self.data_dir.replace("scans_test", "scans_val")
            
            self.opt.scannet_sub_test_type = txt_file_subdir[scannet_sub_test_type]

            self.opt.split='scannetv2' # set manually;
            self.split_str = {
                'scannetv2_small' : 'sml',
                'scannetv2': '' # defualt
            }[self.opt.split]

            txt_file_dir = f'splits/{self.opt.split}/test/{self.opt.scannet_sub_test_type}'
            filename_txt_list = [
                'test_files_iter_00.txt',
                'test_files_iter_01.txt',
                'test_files_iter_02.txt',
                'test_files_iter_03_not_used.txt',
                'val_files.txt',
                ]
            self.opt.filename_txt_list = [
                os.path.join(txt_file_dir, i) for i in filename_txt_list \
                if os.path.isfile(os.path.join(txt_file_dir, i))
                ]

            self.opt.eval_valid_depth_min = 0.25 # meter
            self.opt.eval_valid_depth_max = 20 # meter


        # scannet test set
        elif self.dataset_type == 'scannet_n2_eval':
            self.data_dir = "datasets_link/Scannet/scans_test/"
            self.opt.mvsdata_nviews= 2 # 1 ref frame + 1 source frames
            self.opt.frame_ids=[0,1] # only 1 source frame
            self.dataset_str = f'scantstn{self.opt.mvsdata_nviews}'
            scannet_sub_test_type = self.opt.scannet_eval_sampling_type
            txt_file_subdir = {
                "d-kyn2" : "test_deepvideo_keyframe-nmeas1",
                }
            self.opt.scannet_sub_test_type = txt_file_subdir[scannet_sub_test_type]

            self.opt.split='scannetv2' # set manually;
            self.split_str = {
                'scannetv2_small' : 'sml',
                'scannetv2': '' # defualt
            }[self.opt.split]

            txt_file_dir = f'splits/{self.opt.split}/test/{self.opt.scannet_sub_test_type}'
            filename_txt_list = [
                'test_files_iter_00.txt',
                'test_files_iter_01.txt',
                'test_files_iter_02.txt',
                ]
            self.opt.filename_txt_list = [
                os.path.join(txt_file_dir, i) for i in filename_txt_list
                ]

            self.opt.eval_valid_depth_min = 0.25 # meter
            self.opt.eval_valid_depth_max = 20 # meter

        # 7scenes_eval test set
        elif self.dataset_type in ['7scenes_n3_eval', '7scenes_n5_eval']:
            self.data_dir = "datasets_link/seven-scenes"
            if self.dataset_type == '7scenes_n3_eval':
                self.opt.mvsdata_nviews= 3 # 1 ref frame + 2 source frames
                self.opt.frame_ids=[0,1,2]
            if self.dataset_type == '7scenes_n5_eval':
                self.opt.mvsdata_nviews= 5 # 1 ref frame + 4 source frames
                self.opt.frame_ids=[0,1,2,3,4]
            self.dataset_str = f'7sceneststn{self.opt.mvsdata_nviews}'

            eval_sampling_type = self.opt.scannet_eval_sampling_type
            txt_files = {
                "d-kyfn3" : [ "test_keyframe-nmeas2.txt" ],
                "e-s10n3":  [ "7-scenes-n3-simple10-seqinter10.txt" ],
                # when compared with ESTDepth baseine;
                "e-s10n5":  [ "7-scenes-n5-simple10-seqinter10.txt" ], 
                }
            self.opt.split='7scenes' # set manually;
            self.split_str = {
                '7scenes_small' : 'sml',
                '7scenes': '' # defualt
            }[self.opt.split]

            txt_file_dir = f'splits/{self.opt.split}/test'
            self.opt.filename_txt_list = [
                os.path.join(txt_file_dir, i) for i in txt_files[eval_sampling_type]
                ]

            self.opt.eval_valid_depth_min = 0.25 # meter
            self.opt.eval_valid_depth_max = 20 # meter

        
        # tum-rgbd test set
        elif self.dataset_type in ['tumrgbd_n3_eval', 'tumrgbd_n5_eval']:
            self.data_dir = "datasets_link/tum_rgbd/exported_test"
            if self.dataset_type == 'tumrgbd_n3_eval':
                self.opt.mvsdata_nviews= 3 # 1 ref frame + 2 source frames
                self.opt.frame_ids=[0,1,2]
            if self.dataset_type == 'tumrgbd_n5_eval':
                self.opt.mvsdata_nviews= 5 # 1 ref frame + 4 source frames
                self.opt.frame_ids=[0,1,2,3,4]
            self.dataset_str = f'tumrgbdtstn{self.opt.mvsdata_nviews}'

            eval_sampling_type = self.opt.scannet_eval_sampling_type
            txt_files = {
                "e-s10n3":  [ "tumrgbd-n3-simple10-seqinter10.txt" ],
                # when compared with ESTDepth baseine;
                "e-s10n5":  [ "tumrgbd-n5-simple10-seqinter10.txt" ], 
                "e-s5n3":  [ "tumrgbd-n3-simple5-seqinter5.txt" ], 
                "e-s5n5":  [ "tumrgbd-n5-simple5-seqinter5.txt" ], 
                "e-s20n3":  [ "tumrgbd-n3-simple20-seqinter20.txt" ], 
                "e-s20n5":  [ "tumrgbd-n5-simple20-seqinter20.txt" ], 
                }
            self.opt.split='tumrgbd' # set manually;
            self.split_str = {
                'tumrgbd_sml' : 'sml',
                'tumrgbd': '' # defualt
            }[self.opt.split]

            txt_file_dir = f'splits/{self.opt.split}/test'
            self.opt.filename_txt_list = [
                os.path.join(txt_file_dir, i) for i in txt_files[eval_sampling_type]
                ]

            self.opt.eval_valid_depth_min = 0.25 # meter
            self.opt.eval_valid_depth_max = 20 # meter
        
        # rgbd-scenesv2 test set
        elif self.dataset_type in ['rgbdscenesv2_n3_eval', 'rgbdscenesv2_n5_eval']:
            self.data_dir = "datasets_link/rgbd-scenes-v2/exported_test"
            if self.dataset_type == 'rgbdscenesv2_n3_eval':
                self.opt.mvsdata_nviews= 3 # 1 ref frame + 2 source frames
                self.opt.frame_ids=[0,1,2]
            if self.dataset_type == 'rgbdscenesv2_n5_eval':
                self.opt.mvsdata_nviews= 5 # 1 ref frame + 4 source frames
                self.opt.frame_ids=[0,1,2,3,4]
            self.dataset_str = f'rgbdscenesv2tstn{self.opt.mvsdata_nviews}'

            eval_sampling_type = self.opt.scannet_eval_sampling_type
            txt_files = {
                "e-s10n3":  [ "rgbd-scenesv2-n3-simple10-seqinter10.txt" ],
                # when compared with ESTDepth baseine;
                "e-s10n5":  [ "rgbd-scenesv2-n5-simple10-seqinter10.txt" ], 
                "e-s5n3":  [ "rgbd-scenesv2-n3-simple5-seqinter5.txt" ], 
                "e-s5n5":  [ "rgbd-scenesv2-n5-simple5-seqinter5.txt" ], 
                "e-s20n3":  [ "rgbd-scenesv2-n3-simple20-seqinter20.txt" ], 
                "e-s20n5":  [ "rgbd-scenesv2-n5-simple20-seqinter20.txt" ], 
                }
            self.opt.split='rgbdscenesv2' # set manually;
            self.split_str = {
                'rgbdscenesv2_sml' : 'sml',
                'rgbdscenesv2': '' # defualt
            }[self.opt.split]

            txt_file_dir = f'splits/{self.opt.split}/test'
            self.opt.filename_txt_list = [
                os.path.join(txt_file_dir, i) for i in txt_files[eval_sampling_type]
                ]

            self.opt.eval_valid_depth_min = 0.25 # meter
            self.opt.eval_valid_depth_max = 20 # meter

        # only used for training and validation;
        elif self.dataset_type == 'dtu_yao':
            self.opt.train_validate = False
            self.opt.adaptive_bins = False # dynamic depth range if True;
            if self.opt.raft_mvs_type == 'raft_mvs_adabins':
                self.opt.adaptive_bins = True # adaptive depth bins;

            self.data_dir = "datasets_link/DTU/dtu_patchmatchnet"
            self.dataset_str = 'dtu'
            self.opt.split='dtu' # set manually;
            self.split_str = {
                'dtu_small' : 'sml',
                'dtu': '' # defualt
            }[self.opt.split]
            
            #self.opt.mvsdata_nviews=5 # 1 ref frame + 4 source frames
            if self.opt.mvsdata_nviews < 0:
                self.opt.mvsdata_nviews=5 # 1 ref frame + 4 source frames
            else:
                print (f"!!!{self.dataset_type} has self.opt.mvsdata_nviews = {self.opt.mvsdata_nviews}")
            self.opt.frame_ids=list(range(self.opt.mvsdata_nviews))
            self.opt.num_scales = 4 # i.e., [2^0, 2^{-1}, 2^{-2}, 2^{-3}]

        elif self.dataset_type == 'dtu_yao_eval':
            if self.opt.raft_mvs_type == 'raft_mvs_adabins':
                self.opt.adaptive_bins = True # adaptive depth bins;
            self.data_dir = "datasets_link/DTU/dtu_patchmatchnet_test" 
            self.dataset_str = 'dtu'
            self.opt.split='dtu' # set manually;
            self.split_str = {
                'dtu_small' : 'sml',
                'dtu': '' # defualt
            }[self.opt.split]
            if self.opt.mvsdata_nviews < 0:
                self.opt.mvsdata_nviews=5 # 1 ref frame + 4 source frames
            else:
                print (f"!!! {self.dataset_type} has self.opt.mvsdata_nviews = {self.opt.mvsdata_nviews}")
            self.opt.frame_ids=list(range(self.opt.mvsdata_nviews))
            
            self.opt.eval_valid_depth_min = 0.425 # meter
            self.opt.eval_valid_depth_max = 0.935 # meter
            self.opt.num_scales = 4

        # check `data_dir`, make sure the dataset has been
        # soft-linked into correct local dir;
        assert os.path.exists(self.data_dir), f"Wrong!!! Cannot find {self.data_dir} for {self.dataset_type}"

        if "small" in self.opt.split:
            # for debugging only;
            self.opt.train_validate = True
            print ("For debugging , set self.opt.train_validate = ", self.opt.train_validate)
        
        self.opt.disable_median_scaling = True
        self.opt.data_path = self.data_dir
                        
        exp_name = self.opt.exp_name = self.get_exp_name()
        print ("[/////////] exp_name = ", exp_name)
        run_dir = self.opt.run_dir
        if not os.path.exists(run_dir):
            run_dir = self.opt.run_dir2
        if not os.path.exists(run_dir):
            run_dir = self.opt.run_dir3
        
        self.run_dir = self.opt.run_dir = run_dir
        assert os.path.exists(run_dir), "Use a valid run_dir!!!"
        self.opt.log_dir = os.path.join(run_dir, "logs", exp_name)
        self.opt.checkpts_dir = os.path.join(run_dir, "checkpoints", exp_name)
        self.opt.result_dir = os.path.join(run_dir, "results", exp_name)
        self.opt.csv_dir = os.path.join(run_dir, "results")
        self.opt.model_name = exp_name
        
        # if self.opt.load_weights_path:
        #     input_ckpt_name = self.opt.load_weights_path.split("/")[-2]
        #     self.verify_our_method_name(input_ckpt_name)

        
        if 'bl_estdepth' == self.network_class_name:
            self.opt.mvsdata_nviews=5 # 1 ref frame + 4 source frames
            self.opt.frame_ids=list(range(self.opt.mvsdata_nviews))
        
        if 'bl_mvsnet' == self.network_class_name:
            for name in [
                'pretrain_mvsnet_path', 
                'our_pretrain_mvsnet_path',
                'pretrain_residual_pose_path',
                'pretrain_atten_path',
                ]:
                if hasattr(self.opt, name) and getattr(self.opt, name):
                    tmp_path = os.path.join(run_dir, "checkpoints", getattr(self.opt, name))
                    setattr(self.opt, name, tmp_path)
                    print (f" ~~~ Reset self.opt.{name} to {tmp_path}")
        
        if 'bl_itermvs' == self.network_class_name:
            for name in [
                'pretrain_itermvs_path',
                'our_pretrain_itermvs_path',
                'pretrain_residual_pose_path',
                'pretrain_atten_path',
                ]:
                if hasattr(self.opt, name) and getattr(self.opt, name):
                    tmp_path = os.path.join(run_dir, "checkpoints", getattr(self.opt, name))
                    setattr(self.opt, name, tmp_path)
                    print (f" ~~~ Reset self.opt.{name} to {tmp_path}")

    def save_opts(self):
        """
        Save options to disk so we know what we ran this experiment with
        """
        models_dir = self.opt.checkpts_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def get_our_method_name(self):
        raft_mvs_type = self.opt.raft_mvs_type

        raft_volume_scale = {
            'half': 'H',
            'quarter': 'Q',
            'eighth': '', # default
            'sixteenth': 'S',
            }[self.opt.raft_volume_scale]

        #TMP_MODEL_NAME="exp77A-ddp-raftpsQr4f1a4G3pairspf-Dil1inv-scanpz2-Z0.25to20-D64-epo20-LR1e-4-p-epo4-8-15-gam0.5-bs56-h256xw256-task-20220417191800-57718-grtwm"
        if raft_mvs_type == 'raft_mvs_asyatt_f1_att':
            raft_name = "raftps{}r{}f1a{}".format(
                raft_volume_scale,
                self.opt.raft_mvs_pss_prob_radius,
                self.opt.gma_atten_num_heads
                )
 
        elif raft_mvs_type == 'raft_mvs_adabins':
            raft_name = "raftcas{}r{}".format(
                raft_volume_scale,
                self.opt.raft_mvs_pss_prob_radius)
        
        elif raft_mvs_type == 'raft_mvs':
            raft_name = "raftps{}r{}".format(
                raft_volume_scale,
                self.opt.raft_mvs_pss_prob_radius)
 
        else:
            print (f"Wrong raft_mvs_type={raft_mvs_type}")
            raise NotImplementedError

        if self.opt.n_gru_layers > 1:
            raft_name += 'G{}'.format(self.opt.n_gru_layers)
        if isinstance(self.raft_iters, list):
            raft_iters = sum(self.raft_iters)
        else:
            raft_iters = self.raft_iters
        if raft_iters > 12:
            raft_name += 'Itr{}'.format(raft_iters)

        # fnet
        if self.opt.fnet_name == 'pairnet_fnet':
            if self.opt.fusion_pairnet_feats:
                raft_name += 'pairspf' # 'spf': spatial pyramid fusion module
            else:
                raft_name += 'pair'
        return raft_name


    def get_exp_name(self):
        if "riav_mvs" == self.network_class_name:
            model_name = self.get_our_method_name()
        else:
            model_name = self.network_class_name

        if self.opt.mode == 'test':
            lr_str = ''
        else:
            # considering LR rate etc
            # remove zeros to get neaty name: 1.00e-04 --> 1e-4
            lr = "{0:.2e}".format(self.opt.learning_rate)
            num1, num2 = lr.split('e')
            num1, num2 = float(num1), float(num2)
            lr_str = "-LR" + f'{num1:g}' + 'e' + f'{num2:g}'
            # say
            if self.opt.lr_scheduler == 'constant':
                lr_str += '-c'
            elif self.opt.lr_scheduler == 'piecewise_epoch':
                lr_str += "-p-epo" + ("{}-"*len(self.opt.scheduler_step_size)).format(*self.opt.scheduler_step_size) + \
                    "gam{:g}".format(self.opt.lr_gamma)
            else:
                print (f"[!!!] Wrong LR_SCHEDULER type: {self.opt.lr_scheduler} !!!")
                raise NotImplementedError

        depth_binning_str = {
            'linear': 'Dl',
            'inverse': 'Di',
            'merged': 'Dm',
        }[self.opt.depth_binning]

        loss_type_str = {
            'L1-inv': 'l1inv',
            'L1': 'l1',
            }[self.opt.loss_type]

        # f'{3.140:g}': Formatting floats without trailing zeros
        # e.g., f'{3.1400:g}' --> 3.14;
        min_dep = self.train_min_depth
        max_dep = self.train_max_depth
        if self.opt.mode == 'test':
            min_dep = self.opt.eval_valid_depth_min
            max_dep = self.opt.eval_valid_depth_max

        exp_name = f"{self.opt.exp_idx}-{model_name}-{depth_binning_str}{loss_type_str}" + \
                f"-{self.dataset_str}-Z{min_dep:.2f}to{max_dep:g}" + \
                f"-D{self.train_n_depth_levels:d}-epo{self.train_epochs}" + \
                f"{lr_str}" + \
                f"-bs{self.opt.batch_size}" + \
                f"-h{self.train_image_height}xw{self.train_image_width}" + \
                f"-{self.opt.machine_name}"
        if self.opt.mode == 'test':
            exp_name += f'/depth-epo-{self.opt.eval_epoch:03d}'
            if self.opt.scannet_sub_test_type:
                exp_name += "/" + self.opt.scannet_sub_test_type
        return exp_name

    def print_paths(self):
        print (f"exp_name = {self.opt.exp_name}")
        print (f"log_dir = {self.opt.log_dir}")
        print (f"result_dir = {self.opt.result_dir}")
        print (f"checkpts_dir = {self.opt.checkpts_dir}")

    def verify_our_method_name(self, input_ckpt_name):
        if "riav_mvs" == self.network_class_name:
            model_name = self.get_our_method_name()
        else:
            model_name = self.network_class_name
            if self.opt.network_sub_class_name_str:
                model_name = self.opt.network_sub_class_name_str

        # E.g.,:
        #"exp55B-ddp-raftpsQr4a1-Dil1inv-scanpz2-Z0.25to20-D96-epo10-LR1e-4-p-epo5-7-8-gam0.5-bs44-h256xw256-rtxA6ks3"
        print ("Checking input_ckpt_name={}".format(input_ckpt_name))
        for tmp in input_ckpt_name.split("-")[:]:
            if "exp" in tmp:
                continue
            elif "ddp" in tmp:
                continue
            decoded_model_name = tmp # "raftpsQr4a1Itr24pairspf"
            break

        # check NUMDEPTH
        num_depth_str= f"-D{self.train_n_depth_levels:d}"
        assert num_depth_str in input_ckpt_name, f"Wrong NUMDEPTH used! {num_depth_str} not in {input_ckpt_name}"


        # ignore Itr24
        #print ("???????", model_name)
        pos = model_name.find("Itr")
        if pos > 0:
            model_name = model_name[:pos] + model_name[pos+len("Itr24"):]
        # e.g.,
        print ("checking hyper-parameters based on the naming rule:")
        assert decoded_model_name == model_name, \
                f" ??? to load {decoded_model_name}, do not match current {model_name}"
        print (" ==> Passed")
