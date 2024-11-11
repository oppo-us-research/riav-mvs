"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import time
import matplotlib.pyplot as plt
try:
    from collections.abc import defaultdict
except ImportError:
    from collections import defaultdict
import os
import sys
from datetime import datetime
import random
import json

from os.path import join as pjoin
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

""" DDP related """
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.cuda.amp import GradScaler

### In order to correctly call third_party/* ###
### modules, and do not destory the original import format ###
### inside the maskrcnn_benchmark python files ###
sys.path.append('third_parties/DeepVideoMVS')
sys.path.append('third_parties/ESTDepth')
sys.path.append('third_parties/IterMVS')
sys.path.append('third_parties/RAFT_Stereo')

""" load modules from third_parties/RAFT-Stereo """
from third_parties.RAFT_Stereo.core.utils import frame_utils

""" load our own moduels """
from src.utils.utils import (tensor2numpy, count_parameters, tocuda)
from src.utils.utils import change_sec_to_hm_str as sec_to_hm_str
from src.utils.utils import monodepth_colormap as colormap
from src import datasets
from src.utils import pfmutil as pfm
from src.utils import flow_viz
from src.loss_utils import LossMeter, DictAverageMeter
from src.utils.comm import (is_main_process, print0)
from src.models import __models__

#This flag allows you to enable the inbuilt cudnn auto-tuner to
# find the best algorithm to use for your hardware.
cudnn.benchmark = True

def _get_optimizer(network_class_name, is_print_info_gpu0):
    optimizer_dict = {
        'optim.Adam': optim.Adam,
        'optim.AdamW': optim.AdamW,
    }

    if network_class_name in ['bl_pairnet', 'bl_itermvs', 
                              'bl_estdepth', 'bl_mvsnet']:
        optimizer_type = "optim.Adam"
    else:
        optimizer_type = "optim.AdamW"

    optimizer = optimizer_dict[optimizer_type]
    if is_print_info_gpu0:
        print ("[!!!] using {} for {}".format(optimizer_type, network_class_name))
    return optimizer

""" Multi-GPU Version:  DistributedDataParallel """
def run_train(cfg):
    # create logger
    is_print_info_gpu0 = False

    if is_main_process():
        is_print_info_gpu0 = True
        print ("[!!!] checking Config for {}".format(cfg.opt.network_class_name))
        cfg.print_paths()
        cfg.save_opts()
        for tmp_dir in [cfg.opt.log_dir, cfg.opt.checkpts_dir]:
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
                print ("==> mkdirs ", tmp_dir)

        current_time_str = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        logfile_path = os.path.join(cfg.opt.log_dir, f'{current_time_str}_{cfg.opt.mode}.log')
        print('creating log file', logfile_path)
        logger.add(logfile_path, format="{time} {level} {message}", level="INFO")

    # find which model definition;
    if cfg.opt.network_class_name in ['riav_mvs', 'riav_mvs_cas']:
        assert cfg.opt.raft_mvs_type in [
            "raft_mvs", 'raft_mvs_gma',
            'raft_mvs_asyatt_f1_att',
            'raft_mvs_sysatt_f1_f2_att',
            'raft_mvs_casbins',
            'raft_mvs_adabins'
            ], f"Wrong raft_mvs_type={cfg.opt.raft_mvs_type} found!"
        class_name = cfg.opt.network_class_name
    
    elif cfg.opt.network_class_name in [
            "bl_pairnet", "bl_mvsnet",
            "bl_itermvs", "bl_estdepth"]:
        class_name = cfg.opt.network_class_name
        if cfg.opt.network_sub_class_name:
            class_name = cfg.opt.network_sub_class_name

    else:
        print0 ("Wrong class_name {}".format(class_name))
        raise NotImplementedError
    
    if is_print_info_gpu0:
        logger.info("Will load our model {}".format(class_name))
    # model, optimizer
    model = __models__[class_name](cfg.opt)

    if cfg.DISTRIBUTED:
        model = model.to('cuda')
        
        if cfg.opt.sync_bn:
            #convert all BatchNorm*D layers in the model to torch.nn.SyncBatchNorm layers
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        try:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[cfg.LOCAL_RANK],
                output_device=cfg.LOCAL_RANK,
                static_graph= True
                )
            if is_main_process():
                logger.info("[***] using static_graph for DDP")
        except TypeError:
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[cfg.LOCAL_RANK],
                    output_device=cfg.LOCAL_RANK,
                    find_unused_parameters= cfg.opt.find_unused_parameters,
                    )
            if is_main_process():
                logger.info("[***] this machine does not support static_graph=True for DDP") 
    else:
        # DP training;
        logger.info("[***] DP train ...")
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.cuda()



    count_valid, count_all = count_parameters(model)
    if is_main_process():
        logger.info("[!!!!!xxxx] modle {}: Parameter Count = valid/all = {}/{}".format(
            cfg.opt.network_class_name, count_valid, count_all))

    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        # pointer label
        p_model = model.module
    else:
        p_model = model

    model_names_to_train = []

    if p_model.pose_encoder is not None:
        tmp_count = count_parameters(p_model.pose_encoder)
        if is_print_info_gpu0:
            logger.info("[!!!!!xxxx] modle {}.pose_encoder: Parameter Count = valid/all = {}/{}".format(
                cfg.opt.network_class_name,
                tmp_count[0], tmp_count[1]))
        model_names_to_train.append('pose_encoder')

    if p_model.pose is not None:
        tmp_count = count_parameters(p_model.pose)
        if is_print_info_gpu0:
            logger.info("[!!!!!xxxx] modle {}.pose_decoder: Parameter Count = valid/all = {}/{}".format(
                cfg.opt.network_class_name,
                tmp_count[0],  tmp_count[1] ))
        model_names_to_train.append('pose')

    if p_model.encoder is not None:
        # e.g., manydepth_raft has no separate encoder;
        model_names_to_train.append('encoder')
    if p_model.depth is not None:
        model_names_to_train.append('depth')

    if is_print_info_gpu0:
        logger.info("[***] MultiStepLR: milestones = {opt.scheduler_step_size}")
        logger.info("[***] Freeze_layers = {opt.freeze_layers}")

    # different learning rate for different modules;
    cond1 = 'riav_mvs' in cfg.opt.network_class_name
    if cond1 and cfg.opt.freeze_layers != '':
        if is_print_info_gpu0:
            logger.info("calling get_ourmethod_parameters_to_train() ...")
        parameters_to_train = get_ourmethod_parameters_to_train(p_model,
                                        model_names_to_train,
                                        cfg.opt,
                                        freeze_layers= cfg.opt.freeze_layers
                                        )
    else:
        parameters_to_train = get_parameters_to_train(
                                        p_model, 
                                        model_names_to_train, 
                                        cfg.opt,
                                        freeze_layers = cfg.opt.freeze_layers
                                        )


    optimizer = _get_optimizer(cfg.opt.network_class_name, is_print_info_gpu0)
    model_optimizer = optimizer(
        parameters_to_train,
        cfg.opt.learning_rate,
        weight_decay= cfg.opt.raft_wdecay
        )

    scaler = GradScaler(enabled= cfg.opt.is_raft_mixed_precision)

    # load model
    load_weights_path = cfg.opt.load_weights_path
    resume_path = cfg.opt.resume_path

    # optionally resume from a checkpoint
    if resume_path:
        assert os.path.isfile(resume_path), f"=> no checkpoint found at {resume_path}"
        assert not load_weights_path, "Should set load_weights_path empty or None"
        
        if is_main_process():
            logger.info(f"resuming {resume_path}")
        if cfg.DISTRIBUTED:
            dist.barrier()
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(cfg.LOCAL_RANK)
        checkpoint = torch.load(resume_path, map_location=loc)
        
        cfg.opt.start_epoch = checkpoint['epoch']
        
        model.load_state_dict(
            checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint['model'])

        logger.info(f"=> [***] GPU rank = {dist.get_rank()}, map_loc = {loc}, " \
                    f"successfully resumed from {resume_path}, start_epoch={cfg.opt.start_epoch}")
        
        try:
            if is_main_process():
                logger.info("Loading Adam(W) weights")
            model_optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError:
            if is_main_process():
                logger.info("Can't load Adam(W) - using random")
    
    elif load_weights_path != "":
        assert not resume_path, "Should set cfg.opt.resume_path empty or None"
        assert os.path.isfile(load_weights_path), f"cannot find {load_weights_path}"
        #print0(f"=> [***]Loading checkpoint {load_weights_path}")
        if is_main_process():
            logger.info(f"=> [***]Loading checkpoint {load_weights_path}")
        
        if cfg.DISTRIBUTED:
            dist.barrier()
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(cfg.LOCAL_RANK)
        checkpoint = torch.load(load_weights_path, map_location=loc)

        model.load_state_dict(
            checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint['model'])
        
        logger.info(f"=> [***] GPU rank = {cfg.LOCAL_RANK}, map_loc = {loc}, " \
                    f"successfully loaded checkpoint {load_weights_path}")
        
        try:
            if is_main_process():
                logger.info("Loading Adam(W) weights")
            model_optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError:
            if is_main_process():
                logger.info("Can't load Adam(W) - using random")


    # Data loading code
    datasets_dict = {
        "scannet_mea1_npz": datasets.MVS_Scannet, # 1 measurement (i.e, source) frame;
        # toy data for network debug
        "scannet_mea1_npz_sml": datasets.MVS_Scannet, # 1 measurement (i.e, source) frame;
        "scannet_mea2_npz": datasets.MVS_Scannet, # 2 measurement (i.e, source) frame;
        "scannet_mea2_npz_sml": datasets.MVS_Scannet, # 2 measurement (i.e, source) frame;
        "scannet_mea4_npz": datasets.MVS_Scannet, # 4 measurement (i.e, source) frame;
        "dtu_yao": datasets.MVS_DTU_Yao,
        }

    frames_to_load = cfg.opt.frame_ids.copy()
    if is_print_info_gpu0:
        logger.info('Loading frames: {}'.format(frames_to_load))

    dataset = datasets_dict[cfg.opt.dataset]

    # mvs dataset
    if cfg.opt.dataset in ["scannet_mea1_npz", "scannet_mea1_npz_sml"]:
        # 1 ref + 1 measurement (i.e., source) frame;
        fpath = os.path.join("splits", cfg.opt.split, "scannet_mea1_npz/{}_files.txt")
    elif cfg.opt.dataset in ["scannet_mea2_npz", "scannet_mea2_npz_sml"]:
        # 1 ref + 2 measurement (i.e., source) frames;
        fpath = os.path.join("splits", cfg.opt.split, "scannet_mea2_npz/{}_files.txt")
    elif cfg.opt.dataset in ["scannet_mea4_npz", "scannet_mea4_npz_sml"]:
        # 1 ref + 2 measurement (i.e., source) frames;
        fpath = os.path.join("splits", cfg.opt.split, "scannet_simple10_npz/{}_files.txt")
    elif cfg.opt.dataset in ["dtu_yao"]:
        fpath = os.path.join("splits", cfg.opt.split, "{}.txt")
    else:
        fpath = os.path.join("splits", cfg.opt.split, "{}_files.txt")

    train_filename_txt = fpath.format("train")
    val_filename_txt = fpath.format("val")

    val_dataset = None
    val_loader = None


    #--------------
    # dataloader
    #--------------
    kwargs = {}
    kwargs['load_depth_path'] = False
    kwargs['load_image_path'] = False
    if cfg.opt.dataset in ["scannet_mea1_npz", "scannet_mea1_npz_sml", 
                       "scannet_mea2_npz", "scannet_mea2_npz_sml", 
                       "scannet_mea4_npz",]:
        kwargs['geometric_scale_augmentation'] = True
        kwargs['distortion_crop'] = 10
        kwargs['splitfile_dir'] = None
        kwargs['seed'] = cfg.opt.seed
        train_dataset = dataset(
                data_path = cfg.opt.data_path,
                filename_txt = train_filename_txt,
                split = 'train',
                height = cfg.opt.height,
                width = cfg.opt.width,
                nviews = cfg.opt.mvsdata_nviews,
                #robust_train = True, # scale to adjust depth;
                depth_min= cfg.opt.min_depth,
                depth_max= cfg.opt.max_depth,
                **kwargs
                )
        if cfg.opt.train_validate:
            val_dataset = dataset(
                data_path = cfg.opt.data_path,
                filename_txt = val_filename_txt,
                split = 'val',
                height = cfg.opt.height,
                width = cfg.opt.width,
                nviews = cfg.opt.mvsdata_nviews,
                #robust_train = False, # scale to adjust depth;
                depth_min= cfg.opt.min_depth,
                depth_max= cfg.opt.max_depth,
                **kwargs
                )

    elif cfg.opt.dataset in ["dtu_yao"]:
        kwargs['data_mode'] = 'train'
        kwargs['ndepths'] = cfg.opt.num_depth_bins

        train_dataset = dataset(
            data_path = cfg.opt.data_path,
            filenames = train_filename_txt,
            height = cfg.opt.height,
            width = cfg.opt.width,
            nviews = cfg.opt.mvsdata_nviews, # ref img + source imgs
            num_scales = cfg.opt.num_scales,
            is_train= True,
            robust_train = True,
            load_depth=True,
            depth_min= cfg.opt.min_depth,
            depth_max= cfg.opt.max_depth,
            **kwargs
            )

        if cfg.opt.train_validate:
            kwargs['data_mode'] = 'val' # adjust this arg;
            val_dataset = dataset(
                data_path = cfg.opt.data_path,
                filenames = val_filename_txt,
                height = cfg.opt.height,
                width = cfg.opt.width,
                nviews = cfg.opt.mvsdata_nviews, # ref img + source imgs
                num_scales = cfg.opt.num_scales,
                is_train=False,
                robust_train=False,
                load_depth=True,
                depth_min= cfg.opt.min_depth,
                depth_max= cfg.opt.max_depth,
                ** kwargs
            )

    else:
        logger.info("Wrong dataset type: {}".format(cfg.opt.dataset))
        raise NotImplementedError

    num_train_samples = len(train_dataset)
    num_steps_per_epoch = num_train_samples // cfg.batch_size_orig
    num_total_steps = num_steps_per_epoch * (cfg.opt.num_epochs - cfg.opt.start_epoch)

    # save to object of class TrainerConfig;
    cfg.num_train_samples = num_train_samples
    cfg.num_steps_per_epoch = num_steps_per_epoch
    cfg.num_total_steps = num_total_steps

    
    #----------------------------
    if cfg.DISTRIBUTED:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    # dataset, dataloader 
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = cfg.BATCH_SIZE,
        sampler = train_sampler,
        shuffle = (train_sampler is None),
        num_workers = cfg.NUM_WORKERS, 
        pin_memory=True, 
        drop_last=True,
        )
    cfg.train_loader = train_loader

    if is_main_process():
        logger.info("[*Train] Read {} samples from {}, dataloader len={}, bs_per_gpu={}".format(
            len(train_dataset), cfg.opt.dataset, 
            len(train_loader), cfg.BATCH_SIZE))
 
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = cfg.BATCH_SIZE,
            shuffle=False,
            num_workers = cfg.NUM_WORKERS, 
            pin_memory=True, 
            drop_last=True
            )
        cfg.val_loader = val_loader
        if is_main_process():
            logger.info("[*Validation] Read {} samples from {}, dataloader len={}, bs_per_gpu={}".format(
                len(val_dataset), cfg.opt.dataset, 
                len(val_loader), cfg.BATCH_SIZE))

    if is_print_info_gpu0:
        logger.info(f"\tTraining model named:\n  {cfg.opt.model_name}\n" + \
                    f"\tModels and tensorboard events files are saved to:\n  {cfg.opt.log_dir}" + \
                    f"\tUsing split:\n  {cfg.opt.split}" + \
                    f"\tThere are {len(train_dataset)} training items + {len(val_dataset)} validation items"
                    )

    writers = {}
    if is_main_process():
        logger.info(f"[***GPU rank = {dist.get_rank()},] saving opts")
        cfg.save_opts()
        for mode in ["train", "val"]:
            writers[mode] = SummaryWriter(os.path.join(cfg.opt.log_dir, mode))

        # check the lr
        if 0:
            logger.info('[**info check**] lr')
            for epoch in range(cfg.opt.start_epoch, cfg.opt.num_epochs):
                logger.info('[**info check**] @Epoch-{:02d}'.format(epoch))
                cur_lrs = adjust_learning_rate_per_param(model_optimizer, epoch, opt)
            logger.info('[**info check**] lr done!!!')

    start_time = time.time()
    cfg.start_time = start_time
    ##run the entire training pipeline
    for epoch in range(cfg.opt.start_epoch, cfg.opt.num_epochs):
        if cfg.DISTRIBUTED:
            train_sampler.set_epoch(epoch)

        cur_lrs = adjust_learning_rate_per_param(model_optimizer, epoch, cfg.opt)

        # save for TB log;
        cfg.opt.cur_lrs = cur_lrs

        # train for one epoch
        step = run_epoch(cfg, model, model_optimizer, epoch, writers, scaler)

        if is_main_process() and (epoch + 1) % cfg.opt.save_frequency == 0:
            save_path = save_model(
                            model, model_optimizer,
                            checkpts_path = cfg.opt.checkpts_dir,
                            epoch = epoch,
                            step = step,
                        )
            logger.info("[***] Just saved model at {}".format(save_path))


def save_model(model, optimizer, checkpts_path, epoch, step, scheduler = None, is_warmup = False):
    """
    Save model weights to disk
    """
    save_folder = checkpts_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if is_warmup:
        save_path = os.path.join(save_folder, "ckpt_epoch_%03d_warmup_step_%06d.pth.tar" % (epoch, step))
    else:
        save_path = os.path.join(save_folder, "ckpt_epoch_%03d.pth.tar" % epoch)

    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        height = model.module.opt.height
        width  = model.module.opt.width
    else:
        height = model.opt.height
        width  = model.opt.width

    to_save = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'step': step + 1,
        'height': height,
        'width': width,
    }

    if scheduler is not None:
        to_save.update({'scheduler': scheduler.state_dict()})
    torch.save(to_save, save_path)
    return save_path

#------------------------------------------------
#--------------------- main part ----------------
#------------------------------------------------
def run_epoch(
        trainer_config, model, model_optimizer, 
        epoch_idx, tb_writer, scaler
        ):
    """
    Run a single epoch of training and validation
    """
    if is_main_process():
        logger.info("[***] Training")
        is_print_info_gpu0 = True
    else:
        is_print_info_gpu0 = False
    
    # switch to train mode
    model.train()

    num_total_steps = trainer_config.num_total_steps
    num_steps_per_epoch = trainer_config.num_steps_per_epoch
    # the time when we start first epoch;
    start_time = trainer_config.start_time

    step = epoch_idx*num_steps_per_epoch

    ## running log loss
    loss_running = defaultdict(float)

    args = trainer_config.opt

    #for batch_idx, inputs in enumerate(tqdm(trainer_config.train_loader)):
    for batch_idx, inputs in enumerate(trainer_config.train_loader):
        before_op_time = time.time()

        inputs = tocuda(inputs)

        # only for gpu-0
        my_is_verbose = False 
        if batch_idx % (4*args.print_freq) == 0 and is_print_info_gpu0:
            my_is_verbose = True

        mykargs = {
            'inputs': inputs,
            'is_train': True,
            'is_verbose': my_is_verbose,
            }

        # save imgs every 2*args.log_frequency batches;
        is_log_imgs = (step > 2*args.log_frequency) and batch_idx % (2*args.log_frequency) == (2*args.log_frequency - 1)
        is_log_scalar = (step > 2*args.log_frequency) and batch_idx % args.log_frequency == args.log_frequency - 1

        if args.network_class_name in ['riav_mvs', 'riav_mvs_cas']:
            ## freeze RAFT's FNet and CNet if needed;
            if args.warmup_raft_gru_training_step.startswith('e'):
                freeze_raft_fnet_cnet = epoch_idx < int(args.warmup_raft_gru_training_step[1:])
            elif args.warmup_raft_gru_training_step.startswith('s'):
                freeze_raft_fnet_cnet = step < int(args.warmup_raft_gru_training_step[1:])
            else:
                freeze_raft_fnet_cnet= False
            if freeze_raft_fnet_cnet:
                assert args.raft_pretrained_path or args.load_weights_path, "Cannot be empty. Need ckpt for fnet and cnet!"
            mykargs.update({'freeze_raft_fnet_cnet': freeze_raft_fnet_cnet })

        elif args.network_class_name == 'bl_pairnet':
            ## freeze PairNet's FNet if needed;
            mykargs.update({'is_freeze_fnet': epoch_idx < 2 }) # with warmup
            #mykargs.update({'is_freeze_fnet': epoch_idx < -1 }) # no freeze

        elif args.network_class_name == 'bl_itermvs':
            mykargs.update({
                # to exclude regress and conf loss, to warm up the classification;
                'is_regress_loss': epoch_idx < 2,
                'do_tb_summary' : is_log_scalar,
                'do_tb_summary_image' : is_log_imgs,
                'val_avg_meter': None # only for validation, to accumulate the evluation metrics;
                })
        elif args.network_class_name == 'bl_estdepth':
            mykargs.update({
                'pre_costs' : None, # for train, just use 5-frame input;
                'pre_cam_poses' : None,
                }) 
        
        elif args.network_class_name == 'bl_mvsnet':
            mykargs.update({
                'min_depth_bin': None, # dummy one;
                'max_depth_bin': None, # dummy one;
                }) 

        else:
            raise NotImplementedError

        #------ run the model ---
        outputs, losses = model(**mykargs)

        # save graph

        model_optimizer.zero_grad()

        loss = losses["loss"].mean()
        scaler.scale(loss).backward()
        scaler.unscale_(model_optimizer)
        clip_grad_value = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= clip_grad_value, norm_type=2.0)
        scaler.step(model_optimizer)
        scaler.update()

        # measure elapsed time
        duration = time.time() - before_op_time

        # principal GPU to print some info;
        if is_print_info_gpu0 and batch_idx % args.print_freq == 0:
            my_info = log_time(batch_idx, epoch_idx, duration, losses["loss"].mean().item(), \
                num_total_steps, num_steps_per_epoch,
                step - args.start_epoch*num_steps_per_epoch, # if resuming from some existing ckpt;
                start_time, args.batch_size,
                # added .mean() for DP code;
                epe=losses['loss/last_epe'].mean() if 'loss/last_epe' in losses else None)
            logger.info("@train " +  my_info)

        if is_print_info_gpu0 and step in [500, 1000, 2000, 4000, 5000, 8000]:
            #warmup training
            save_path = save_model(model, model_optimizer,
                        checkpts_path = args.checkpts_dir,
                        epoch = epoch_idx,
                        step = step,
                        is_warmup=True)
            logger.info(f"[***] Just saved model at {save_path}")

        # running loss accumulation
        with torch.no_grad():
            if is_print_info_gpu0 and step in [1, 50, 500, 1000, 5000, 8000,
                        10000, 15000, 20000, 22000,
                        25000, 26000, 27000, 28000, 29000, 30000,
                        32000, 34000, 35000, 36000,
                        40000, 45000, 46000, 47000, 48000, 49000,
                        50000, 70000, 100000, 150000, 200000]:

                save_log_to_pfm(
                    step,
                    args.log_dir,
                    # remove str variable, e.g., image_path
                    {k: v for k,v in inputs.items() if not isinstance(v, str)},
                    outputs,
                    args)

            for l, v in losses.items():
                loss_running[l] += v

            if is_log_scalar:
                for l, v in loss_running.items():
                    loss_running[l] = v/args.log_frequency

                # log training loss (running loss, i.e, averaged loss)
                # do not save first args.log_frequency steps, to avoid very large losses;
                if tb_writer and (tb_writer['train'] is not None):
                    log_tb(step, tb_writer['train'], inputs, outputs, args, loss_running, is_log_imgs)

                # reset running loss to zeros
                for l, v in loss_running.items():
                    loss_running[l] = .0


        step += 1 # update it;

    if args.train_validate: # validation
        if is_print_info_gpu0:
            logger.info("[***] Validating")

        if args.network_class_name == 'bl_itermvs':
            val_per_epoch_bl_itermvs(epoch_idx, step, model, trainer_config, tb_writer)
        else:
            val_per_epoch(epoch_idx, step, model, trainer_config, tb_writer)

    # switch back to train mode !!!
    model.train()

    return step


def val_per_epoch(epoch_idx, step, model, trainer_config, tb_writer):
    """
    Validate the model on an epoch
    """
    args = trainer_config.opt

    # used to identify principal GPU to print or save;
    if is_main_process():
            is_print_info_gpu0 = True
    else:
        is_print_info_gpu0 = False

    info_printer = tqdm(total=0, position=1, bar_format='{desc}')

    loss_meter = LossMeter()
    validation_l1_meter = LossMeter()
    validation_l1_inv_meter = LossMeter()
    validation_l1_rel_meter = LossMeter()

    # switch to evaluate mode
    model.eval()
    time_zero = time.time()
    batch_size_per_gpu = args.batch_size
    valid_data_size = len(trainer_config.val_loader)

    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(trainer_config.val_loader)):
            before_op_time = time.time()
            inputs = tocuda(inputs)

            mykargs = {
                'inputs': inputs,
                'is_train': False,
                'is_verbose': False,
                }

            outputs, losses = model(**mykargs)

            batch_l1_meter_sum = losses['batch_l1_meter_sum']
            batch_l1_inv_meter_sum = losses['batch_l1_inv_meter_sum']
            batch_l1_rel_meter_sum = losses['batch_l1_rel_meter_sum']
            batch_l1_meter_count = losses['batch_l1_meter_count']

            # record losses
            validation_l1_meter.update(loss=batch_l1_meter_sum, count=batch_l1_meter_count)
            validation_l1_inv_meter.update(loss=batch_l1_inv_meter_sum, count=batch_l1_meter_count)
            validation_l1_rel_meter.update(loss=batch_l1_rel_meter_sum, count=batch_l1_meter_count)
            loss_meter.update(loss= losses['loss'], count=1)

            batch_L1_loss = batch_l1_meter_sum / batch_l1_meter_count
            batch_L1_inv_loss = batch_l1_inv_meter_sum / batch_l1_meter_count
            batch_L1_rel_loss = batch_l1_rel_meter_sum / batch_l1_meter_count

            # print info
            duration = time.time() - before_op_time
            samples_per_sec = batch_size_per_gpu / duration
            if is_print_info_gpu0 and batch_idx % args.print_freq == 0:
                #print ("[???validation] l1_meter: sum/count = ", batch_l1_meter_sum, batch_l1_meter_count)
                valid_print_string = "@Vld-Batch: epoch {:>3} |examples/s: {:3.1f}" + " |s/frame: {:3.1f}" + \
                    " |loss: {:.2f} |L1: {:.2f} |L1-inv: {:.2f} |L1-rel: {:.2f} |tElapsed: {}"
                valid_to_print_list = [epoch_idx, samples_per_sec, duration/batch_size_per_gpu,
                                losses['loss'].mean(),
                                batch_L1_loss.mean(),
                                batch_L1_inv_loss.mean(),
                                batch_L1_rel_loss.mean(),
                                sec_to_hm_str(duration)
                                ]
                info_printer.set_description_str(valid_print_string.format(*valid_to_print_list))
                #print(valid_print_string.format(*valid_to_print_list))

        # save the average loss to log_tb,
        # and just save the depth result at last step for visualization
        L1_loss = validation_l1_meter.avg
        L1_inv_loss = validation_l1_inv_meter.avg
        L1_rel_loss = validation_l1_rel_meter.avg
        loss_avg = loss_meter.avg
        if 'riav_mvs' in args.network_class_name:
            validation_losses = {
                # same name to that in train log;
                'loss/last_itr/L1': L1_loss,
                'loss/last_itr/L1-inv': L1_inv_loss,
                'loss/last_itr/L1-rel': L1_rel_loss,
              }
        else:
            validation_losses = {
                'loss/L1': L1_loss,
                'loss/L1-inv': L1_inv_loss,
                'loss/L1-rel': L1_rel_loss,
                }


        if tb_writer and tb_writer['val'] is not None:
            log_tb(step, tb_writer['val'], inputs, outputs, args, validation_losses, is_log_imgs=True)

        duration = time.time() - time_zero
        samples_per_sec = valid_data_size / duration
        if is_print_info_gpu0:
            valid_print_string = "@Vld-All: epoch {:>3} |examples/s: {:3.1f}" + " |s/frame: {:3.1f}" + \
                " |loss: {:.3f} |L1: {:.3f} |L1-inv: {:.2f} |L1-rel: {:.3f} |tElapsed: {}"
            valid_to_print_list = [
                epoch_idx, samples_per_sec, 1.0/samples_per_sec,
                loss_avg.mean(), L1_loss.mean(), L1_inv_loss.mean(), L1_rel_loss.mean(),
                sec_to_hm_str(duration)]
            my_info = valid_print_string.format(*valid_to_print_list)
            logger.info(my_info)

def val_per_epoch_bl_itermvs(epoch_idx, step, model, trainer_config, tb_writer):
    """
    Validate the model on an epoch
    """
    args = trainer_config.opt

    # used to identify principal GPU to print or save;
    if is_main_process():
            is_print_info_gpu0 = True
    else:
        is_print_info_gpu0 = False

    info_printer = tqdm(total=0, position=1, bar_format='{desc}')

    # switch to evaluate mode
    model.eval()
    time_zero = time.time()
    batch_size_per_gpu = args.batch_size
    valid_data_size = len(trainer_config.val_loader)

    val_avg_meter =  DictAverageMeter()

    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(trainer_config.val_loader)):
            before_op_time = time.time()
            inputs = tocuda(inputs)

            mykargs = {
                'inputs': inputs,
                'is_train': False,
                'is_verbose': False,
                # to exclude regress and conf loss, to warm up the classification;
                'is_regress_loss': False,
                'do_tb_summary' : False,
                'do_tb_summary_image' : False,
                # only for validation, to accumulate the evluation metrics;
                'val_avg_meter':  val_avg_meter,
                }

            outputs, losses = model(**mykargs)

            # print info
            duration = time.time() - before_op_time
            samples_per_sec = batch_size_per_gpu / duration
            if is_print_info_gpu0 and batch_idx % args.print_freq == 0:
                #print ("[???validation] l1_meter: sum/count = ", batch_l1_meter_sum, batch_l1_meter_count)
                valid_print_string = "@Vld-Batch: epoch {:>3} |examples/s: {:3.1f}" + " |s/frame: {:3.1f}" + \
                    " |loss: {:.2f} |tElapsed: {}"
                valid_to_print_list = [epoch_idx, samples_per_sec, duration/batch_size_per_gpu,
                                losses['loss'].mean(),
                                sec_to_hm_str(duration)
                                ]
                info_printer.set_description_str(valid_print_string.format(*valid_to_print_list))
                #print(valid_print_string.format(*valid_to_print_list))

        # save the average loss to log_tb,
        validation_meter = val_avg_meter.mean() # dict
        #logger.info("avg_test_scalars:")
        #for k, v in validation_meter.items():
        #    # which is a dict;
        #    logger.info ("   {} : {}".format(k, v.item()))
        val_loss = validation_meter['loss']


        if tb_writer and tb_writer['val'] is not None:
            log_tb(step, tb_writer['val'], inputs, outputs, args, validation_meter, is_log_imgs=True)

        duration = time.time() - time_zero
        samples_per_sec = valid_data_size / duration
        if is_print_info_gpu0:
            valid_print_string = "@Vld-All: epoch {:>3} |examples/s: {:3.1f}" + " |s/frame: {:3.1f}" + \
                " |loss: {:.3f} |tElapsed: {}"
            valid_to_print_list = [
                epoch_idx, samples_per_sec, 1.0/samples_per_sec,
                val_loss.mean(),
                sec_to_hm_str(duration)]
            my_info = valid_print_string.format(*valid_to_print_list)
            logger.info(my_info)

def log_time(batch_idx, epoch, duration, loss,
    num_total_steps, num_steps_per_epoch,
    step, start_time,
    batch_size_per_gpu,
    epe=None,
    m_2_mm_scale=None # change loss or error from meter to milimter; 
    ):
    """
    Print a logging statement to the terminal
    """
    samples_per_sec = batch_size_per_gpu / duration
    time_sofar = time.time() - start_time
    training_time_left = (num_total_steps / step - 1.0) * time_sofar if step > 0 else 0
    print_string = "epoch {:>3} | batch {:>5}/{:>5} | examples/s: {:3.1f}" + \
        " | s/frame: {:3.2f}"

    to_print_list = [epoch, batch_idx, num_steps_per_epoch,
                     samples_per_sec, duration/batch_size_per_gpu,
                     ]
    if epe is not None:
        print_string += " | epe: {:2.3f}"
        to_print_list.append(epe)

    print_string += " | loss: {:.4f} | tElapsed: {} | tLeft: {}"

    to_print_list += [
        loss, sec_to_hm_str(time_sofar),
        sec_to_hm_str(training_time_left)
    ]

    return print_string.format(*to_print_list)

def save_log_to_pfm(step, log_path, inputs, outputs, args):
    """
    Write intermediates results to the pfm files
    """
    inputs = tensor2numpy(inputs)
    outputs = tensor2numpy(outputs)
    log_result_path = pjoin(log_path, "pfm_imgs")
    if not os.path.exists(log_result_path):
        os.makedirs(log_result_path)
        #print ("[xxx] mkdir {}".format(log_result_path))
    for j in range(min(2, args.batch_size)):  # write a maxmimum of two images
        #s = args.depth_map_scale_int  # log only max scale
        s = 0
        tmp_name = "step{:06d}_scl{}_bt{}".format(step, s, j)
        for frame_id in args.frame_ids:
            if s == 0:
                pfm.save(pjoin(log_result_path, 'color_' + tmp_name + '_frm{}'.format(frame_id) + '.pfm'),
                    inputs[("color", frame_id, s)][j].transpose((1,2,0))) #[H, W, 3]
                if frame_id != 0:
                    if ("color", frame_id, s) in outputs:
                        pfm.save(
                            pjoin(log_result_path, "color_pred_" + tmp_name + '_frm{}'.format(frame_id) + '.pfm'),
                            outputs[("color", frame_id, s)][j].transpose((1,2,0)))

        depth = outputs[("depth", 0, s)][j, 0] #[H, W]
        pfm.save(pjoin(log_result_path, 'depth_multi_' + tmp_name + '.pfm'), depth)

        disp = outputs[("disp", s)][j, 0] #[H, W]
        #print ('[???] 0', disp.shape)
        pfm.save(pjoin(log_result_path, 'disp_multi_' + tmp_name + '.pfm'), disp)

        if "depth_gt" in inputs:
            #print ("[???] depth_gt shape = ", inputs["depth_gt"].shape)
            depth_gt = inputs["depth_gt"][j, 0]
            pfm.save(pjoin(log_result_path, 'depth_gt_' + tmp_name + '.pfm'), depth_gt)

        if outputs.get("lowest_cost") is not None:
            lowest_cost_depth = 1 / outputs['lowest_cost'][j][0] #[H,W]
            pfm.save(pjoin(log_result_path, 'lowest_cost_depth_' + tmp_name + '.pfm'), lowest_cost_depth)
            if 'lowest_cost_idx' in outputs:
                lowest_cost_idx = outputs["lowest_cost_idx"][j] #[H/4, W/4]
                pfm.save(pjoin(log_result_path, 'lowest_cost_idx_' + tmp_name+ '.pfm'), lowest_cost_idx.astype(np.float32))


def log_tb(step, writer, inputs, outputs, args, losses, is_log_imgs=True):
    """
    Write an event to the tensorboard events file
    """
    for l, v in losses.items():
        #print ("[???] {}: type {}".format(l, type(v)))
        if isinstance(v, torch.Tensor):
            v= v.mean()
        writer.add_scalar("{}".format(l), v, step)

    # learning rate
    for k, lr in args.cur_lrs.items():
        writer.add_scalar("lr/{}".format(k), lr, step)
    # depth range
    if 'min_depth_bin' in outputs:
        writer.add_scalar("min_depth_tracker", outputs['min_depth_bin'].mean(), step)
    if 'max_depth_bin' in outputs:
        writer.add_scalar("max_depth_tracker", outputs['max_depth_bin'].mean(), step)

    # add histogram
    # save disparity maps
    for tmp_s in [0, 1]:
        if ('disp', tmp_s) in outputs:
            writer.add_histogram("hist_mvs_disp_{}".format(tmp_s), outputs[("disp", tmp_s)], step)
        # save depth maps
        if ('depth', 0, tmp_s) in outputs:
            writer.add_histogram("hist_mvs_depth_{}".format(tmp_s), outputs[("depth", 0, tmp_s)], step)

    if ('disp_refine', 0) in outputs:
        writer.add_histogram("hist_mvs_disp_refine_{}".format(0), outputs[('disp_refine', 0)], step)

    if ('depth_gt', 0) in inputs:
        writer.add_histogram("hist_depth_gt_{}".format(0), inputs[('depth_gt', 0)], step)
    else:
        writer.add_histogram("hist_depth_gt_{}".format(0), inputs['depth_gt'], step)
    
    if 'depth_bins' in outputs:
        writer.add_histogram("hist_depth_bins", outputs['depth_bins'], step)

    # refine_net_type == 'refinenet_pairnet_8th':
    for scale in [0, 1, 2]:
        if ('disp_refine', 0, scale) in outputs:
            writer.add_histogram("hist_mvs_disp_refine_scl/{}".format(scale),
                                outputs[('disp_refine', 0, scale)], step)

    for stage_idx in range(4):
        if f'depth_min/stage{stage_idx}' in outputs:
            depth_min = outputs[f'depth_min/stage{stage_idx}']
            writer.add_histogram(
                f"hist_depth_min/stage{stage_idx}", depth_min, step)

        if f'depth_max/stage{stage_idx}' in outputs:
            depth_max = outputs[f'depth_max/stage{stage_idx}']
            writer.add_histogram(
                f"hist_depth_max/stage{stage_idx}", depth_max, step)


    if is_log_imgs:
        for j in range(min(4, args.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            #s = args.depth_map_scale_int  # log only max scale
            
            for frame_id in args.frame_ids:
                if ("color", frame_id, s) not in inputs:
                    img_0 = inputs[("color", frame_id, 0)]
                    inputs[("color", frame_id, s)] = F.interpolate(
                            img_0, 
                            [img_0.shape[2]//(2**s), img_0.shape[3]//(2**s)], 
                            mode="bilinear", 
                            align_corners=True
                        )

            for frame_id in args.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, step)
                if frame_id != 0:
                    if ("color", frame_id, s) in outputs:
                        writer.add_image(
                            f"color_pred_{frame_id}_{s}/{j}",
                            outputs[("color", frame_id, s)][j].data, step)
                    if ("color_gtdepth", frame_id, s) in outputs:
                        writer.add_image(
                            "color_gtdepth_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color_gtdepth", frame_id, s)][j].data, step)


            disp = colormap(outputs[("disp", s)][j, 0])
            writer.add_image(
                "disp_multi_{}/{}".format(s, j), disp, step)

            if ('disp_refine', 0) in outputs:
                disp = colormap(outputs[('disp_refine', 0)][j, 0])
                writer.add_image(
                    "disp_multi_refine_{}/{}".format(s, j), disp, step)
            
            for scale in [0, 1, 2]:
                if ('disp/no_resi_pose', scale) in outputs: # 
                    disp = colormap(outputs[('disp/no_resi_pose', scale)][j, 0])
                    #print ("Tb saving disp/no_resi_pose_{}/{}".format(scale, j))
                    writer.add_image(
                        "disp_multi_no_resi_pose_{}/{}".format(scale, j), disp, step)

            # refine_net_type == 'refinenet_pairnet_8th':
            for scale in [0, 1, 2]:
                if ('disp_refine', 0, scale) in outputs:
                    disp = colormap(outputs[('disp_refine', 0, scale)][j, 0])
                    writer.add_image(
                        "disp_multi_refine_bs{}_scl/{}".format(j, scale), disp, step)


            if ('disp_iters', 0) in outputs:
                #for itr in range(0, args.raft_iters, 2):
                for itr in range(0, len(outputs[('disp_iters', 0)]), 4):
                    disp = outputs[('disp_iters', 0)][itr][j, 0]
                    disp = colormap(disp)
                    writer.add_image(
                        "disp_raft_bs{}_itr/{}".format(j, itr), disp, step)

                    for frame_id in args.frame_ids:
                        #outputs[("color_iters", frame_id, scale, itr)]
                        tmp_key = ('color_iters', frame_id, s, itr)
                        if tmp_key in outputs:
                            if j ==0: # just save one batch;
                                writer.add_image(
                                    "color_raft_bs{}_frm{}_itr/{}".format(j, frame_id, itr),
                                    outputs[tmp_key][j], step)


            elif ('depth_iters', 0) in outputs:
                #for itr in range(0, args.raft_iters, 2):
                for itr in range(0, len(outputs[('depth_iters', 0)]), 4):
                    disp = 1.0 / (1e-8+ outputs[('depth_iters', 0)][itr][j, 0])
                    disp = colormap(disp)
                    writer.add_image(
                        "disp_raft_bs{}_itr/{}".format(j, itr), disp, step)

            for stage_idx in range(4):
                if (f"depth/stage{stage_idx}/raft", 0, 0) in outputs:
                    disp = 1.0/(1e-8+outputs[(f"depth/stage{stage_idx}/raft", 0, 0)][j, 0])
                    disp = colormap(disp)
                    writer.add_image(
                        "disp_cascd_bs{}_stage/{}/raft".format(j, stage_idx), disp, step)

                if (f"depth/stage{stage_idx}/init", 0, 0) in outputs:
                    disp = 1.0/(1e-8+outputs[(f"depth/stage{stage_idx}/init", 0, 0)][j, 0])
                    disp = colormap(disp)
                    writer.add_image(
                        "disp_cascd_bs{}_stage/{}/init".format(j, stage_idx), disp, step)


            # add s=1
            for tmp_s in [1, 2, 3]:
                if ('disp', tmp_s) in outputs:
                    disp = colormap(outputs[("disp", tmp_s)][j, 0])
                    writer.add_image(
                        "disp_multi_{}/{}".format(tmp_s, j), disp, step)

            if 'depth_gt' in inputs or ('depth_gt', 0) in inputs:
                depth_gt = inputs['depth_gt'][j,0] if 'depth_gt' in inputs else inputs[('depth_gt', 0)][j,0]
                if "depth_mask" in inputs:
                    mask = inputs["depth_mask"][j,0] > 0.5
                else:
                    mask = (depth_gt > args.min_depth) & (depth_gt < args.max_depth)
                depth_gt[depth_gt < args.min_depth] = args.min_depth
                depth_gt[depth_gt > args.max_depth] = args.max_depth
                # here we do not normalize to get good color visualization;
                if args.dataset in ["dtu_yao"]:
                    tmp_normalize = True
                else:
                    tmp_normalize = False
                #disp_gt = colormap(1.0/(depth_gt), normalize= False) # Scannet
                #disp_gt = colormap(1.0/(depth_gt), normalize= True) # DTU
                disp_gt = colormap(1.0/(depth_gt), normalize= tmp_normalize)
                writer.add_image(
                    "disp_gt/{}".format(j), disp_gt, step)

                writer.add_image(
                    "depth_gt_mask/{}".format(j), mask.float(), step, dataformats='HW')



            flow_key_map = {-1: 'backwardflow_gt', 1: 'forwardflow_gt', 0: 'none'}
            for frame_id in args.frame_ids:
                # GT flows
                if frame_id in flow_key_map and flow_key_map[frame_id] in inputs:
                    flow = inputs[flow_key_map[frame_id]][j]
                    flow_rgb = flow_viz.flow_uv_to_colors(
                        u= flow.cpu().numpy()[0, :,:],
                        v= flow.cpu().numpy()[1, :,:]
                        )
                    #print ("[???] flow_rgb ", flow_rgb.shape)
                    writer.add_image(
                        "flow_gt_{}/{}".format(frame_id, j),
                        flow_rgb, step, dataformats='HWC')
                elif ('flow', frame_id) in inputs:
                    flow = inputs[('flow', frame_id)][j]
                    flow_rgb = flow_viz.flow_uv_to_colors(
                        u= flow.cpu().numpy()[0, :,:],
                        v= flow.cpu().numpy()[1, :,:]
                        )
                    #print ("[???] flow_rgb ", flow_rgb.shape)
                    writer.add_image(
                        "flow_gt_{}/{}".format(frame_id, j),
                        flow_rgb, step, dataformats='HWC')

                # prediction flows
                if ('flow', '%d'%(frame_id)) in outputs:
                    # flow in size [N, 2, H, W]
                    flow = outputs[('flow', '%d'%(frame_id))][j]
                    flow_rgb = flow_viz.flow_uv_to_colors(
                        u= flow.cpu().numpy()[0, :,:],
                        v= flow.cpu().numpy()[1, :,:]
                        )
                    writer.add_image(
                        "flow_lastItr_raft_frm{}_bs/{}".format(frame_id, j),
                        flow_rgb, step, dataformats='HWC')

                # prediction flows via GT depth;
                if ('flow_via_gt_depth', '%d'%(frame_id)) in outputs:
                    flow = outputs[('flow_via_gt_depth', '%d'%(frame_id))][j]
                    flow_rgb = flow_viz.flow_uv_to_colors(
                        u= flow.cpu().numpy()[0, :,:],
                        v= flow.cpu().numpy()[1, :,:]
                        )
                    writer.add_image(
                        "flow_via_8th_gt_depth_frm{}_bs/{}".format(frame_id, j),
                        flow_rgb, step, dataformats='HWC')

                # prediction flows
                tmp_key = ('flow_iters', '%d'%(frame_id))
                if j == 0 and (tmp_key in outputs): # just save one batch;
                    flow_preds = outputs[tmp_key]
                    for itr in range(0, len(flow_preds), 4):
                        flow = flow_preds[itr][j] #[2, H, W]
                        flow_rgb = flow_viz.flow_uv_to_colors(
                            u= flow.cpu().numpy()[0, :,:],
                            v= flow.cpu().numpy()[1, :,:]
                            )
                        writer.add_image(
                            "flow_raft_frm{}_bs0_itr/{}".format(frame_id, itr),
                            flow_rgb, step, dataformats='HWC')


            tmp_key = ('flow1d_iters', 0)
            if j == 0 and (tmp_key in outputs): # just save one batch;
                flow1d_preds = outputs[tmp_key]
                for itr in range(0, len(flow1d_preds), 4):
                    flow1d = flow1d_preds[itr][j, 0] #[ H, W]
                    flow1d_rgb = colormap(flow1d)
                    writer.add_image(
                        "flow1d_raft_bs{}_itr/{}".format(j, itr), flow1d_rgb, step)

            tmp_key = ('flow1d', 0)
            if tmp_key in outputs:
                flow1d = outputs[tmp_key][j, 0] #[H,W]
                flow1d_rgb = colormap(flow1d)
                writer.add_image(
                    "flow1d_lastItr_raft_bs/{}".format(j), flow1d_rgb, step)

            tmp_key = ('flow1d_gt', 0)
            if tmp_key in outputs:
                flow1d = outputs[tmp_key][j, 0] #[H,W]
                flow1d_rgb = colormap(flow1d)
                writer.add_image(
                    "flow1d_gt/{}".format(j), flow1d_rgb, step)

            if "softargmin_depth" in outputs:
                disp = 1.0/(1e-8+outputs['softargmin_depth'][j, 0])
                #print ("///// tb: softargmin_depth", disp.shape)
                disp = colormap(disp)
                writer.add_image(
                    "softargmin_cost/{}".format(j), disp, step)

            for stage_idx in range(4):
                tmp_key = f'softargmin_depth/stage{stage_idx}'
                if tmp_key in outputs:
                    disp = 1.0/(1e-8+outputs[tmp_key][j, 0])
                    #print (tmp_key, disp.shape)
                    disp = colormap(disp)
                    writer.add_image(
                        f"softargmin_cost_bs{j}/stage{stage_idx}",
                        disp, step)

            if outputs.get('lowest_cost/no_resi_pose') is not None:
                lowest_cost = outputs['lowest_cost/no_resi_pose'][j, 0] # alread in disparity;
                #print ("//// tb: lowest_cost", lowest_cost.shape)

                min_val = np.percentile(lowest_cost.cpu().numpy(), 10)
                max_val = np.percentile(lowest_cost.cpu().numpy(), 90)
                lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                lowest_cost = colormap(lowest_cost)

                writer.add_image(
                    "lowest_cost_no_resipose/{}".format(j), lowest_cost, step)

            if outputs.get("lowest_cost") is not None:
                lowest_cost = outputs["lowest_cost"][j, 0] # alread in disparity;
                #print ("//// tb: lowest_cost", lowest_cost.shape)

                min_val = np.percentile(lowest_cost.cpu().numpy(), 10)
                max_val = np.percentile(lowest_cost.cpu().numpy(), 90)
                lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                lowest_cost = colormap(lowest_cost)

                writer.add_image(
                    "lowest_cost/{}".format(j), lowest_cost, step)
            
            if "confidence" in outputs:
                conf = torch.sigmoid(outputs['confidence'][j, 0])
                writer.add_image(
                    "depth_conf/{}".format(j), conf, step, dataformats='HW')
            if "confidence_gt" in outputs:
                conf = torch.sigmoid(outputs['confidence_gt'][j, 0])
                writer.add_image(
                    "depth_confGT/{}".format(j), conf, step, dataformats='HW')



def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    for i, l in enumerate(lr):
        print ("\tlr/{} = {}".format(i, l))
    return lr

def adjust_learning_rate_per_param(optimizer, epoch_idx, args):
    scheduler_step_size = args.scheduler_step_size
    gamma = args.lr_gamma
    assert isinstance(scheduler_step_size, list), "require type(scheduler_step_size)=list"
    scheduler_step_size.sort() # sorted in Ascending order
    # adjust lr by epoch_idx
    def find_idx(epoch_idx, scheduler_step_size):
        scheduler_step_size.sort()
        for i, p in enumerate(scheduler_step_size):
            if epoch_idx < p:
                return i
        return i+1

    idx = find_idx(epoch_idx, scheduler_step_size)
    #lr_dict = defaultdict(float)
    lr_dict = {}
    
    if is_main_process():
        logger.info("[***:] current epo_idx = {:02d}".format(epoch_idx))
        is_print_info_gpu0 = True
    else:
        is_print_info_gpu0 = False
    
        
    for param_group in optimizer.param_groups:
        name = param_group['my_name']

        if 'pose' in name:
            lr_name = 'learning_rate_pose'
        else:
            lr_name = f'learning_rate_{name}'
        # get initial lr
        if hasattr(args, lr_name):
            init_lr = getattr(args, lr_name)
            if init_lr <= 0:
                init_lr = args.learning_rate
        else:
            init_lr = args.learning_rate

        lr = init_lr * (gamma ** idx)
        param_group['lr'] = lr
        lr_dict[name] = lr
        if is_print_info_gpu0:
            logger.info("\t==> layers {:12s} : epoch_idx = {:02d} : lr = {:.2e}".format(
                name, epoch_idx, lr))
    return lr_dict

"""
## per-parameter optimizer setup
# Args:
# model_names_to_train: e.g., = ['mono_encoder', 'mono_depth', 'pose_encoder',
#                                'pose', 'encoder', 'depth', 'feat_prob']
# > see: https://pytorch.org/docs/0.3.0/optim.html#per-parameter-options;
"""
def get_parameters_to_train(p_my_model, model_names_to_train, args, 
                epoch_idx=None, 
                freeze_layers=None
                ):

    is_verbose = False
    if is_main_process():
        is_verbose = True 
    
    full_model_names_to_train = [
            'pose_encoder', 'pose',
            'encoder',
            'depth',
        ]

    parameters_to_train = []
    if is_verbose:
        print ("[***] model_names_to_train = ", model_names_to_train)
    if epoch_idx is not None:
        scheduler_step_size = args.scheduler_step_size
        gamma = args.lr_gamma
        assert isinstance(scheduler_step_size, list), "require type(scheduler_step_size)=list"
        scheduler_step_size.sort() # sorted in Ascending order
        # adjust lr by epoch_idx
        def find_idx(epoch_idx, scheduler_step_size):
            #scheduler_step_size.sort()
            for i, p in enumerate(scheduler_step_size):
                if epoch_idx < p:
                    return i
            return i+1
        idx = find_idx(epoch_idx, scheduler_step_size)

    ## We would like to keep the order of modles, and this order will
    # serve as kind of index, e.g., accessing the lr of those params
    # and write them to tensorboard summary;
    sum_n = 0
    frozen_n = 0
    sub_count_dict = {}
    for name in full_model_names_to_train:
        if name in model_names_to_train:
            if is_verbose:
                print ("checking ", name)
            if freeze_layers and (name in freeze_layers):
                for p_name, param in getattr(p_my_model, name).named_parameters():
                    param.requires_grad = False
                    #if is_verbose:
                    #    print (f"freezing {p_name}")
                if is_verbose:
                    print ("froze module ", name)
            
            tmp_n, tmp_n2 =  count_parameters(getattr(p_my_model, name))
            frozen_n += (tmp_n2 - tmp_n)
            sum_n += tmp_n
            sub_count_dict[name] = tmp_n

            # set lr for pose and mono if needed;
            #if 'pose' in name:
            #    lr_name = 'learning_rate_pose'
            #elif 'mono' in name:
            #    lr_name = 'learning_rate_mono'
            # set lr to other modules using args.learning_rate
            #else:
            #    lr_name = 'learning_rate'

            if 'pose' in name:
                lr_name = 'learning_rate_pose'
            else:
                lr_name = f'learning_rate_{name}'
            # get initial lr
            if hasattr(args, lr_name):
                init_lr = getattr(args, lr_name)
                if init_lr <= 0:
                    init_lr = args.learning_rate
            else:
                init_lr = args.learning_rate

            lr = init_lr * (gamma ** idx) if epoch_idx is not None else init_lr
            if is_verbose:
                print ("\t==> layers {} : lr = {:.2e}".format(name, lr))
            parameters_to_train += [
                {
                    #'params': getattr(p_my_model, name).parameters(),
                    'params': torch.nn.ParameterList(
                        filter(lambda p: p.requires_grad, getattr(p_my_model, name).parameters() )
                        ),
                    'lr': lr,
                    'my_name': name,
                }]
            if is_verbose:
                print (f"  ==> We select submodel {name}, with params # = {tmp_n}(val)/{tmp_n2}(all)")
    return parameters_to_train


def get_ourmethod_parameters_to_train(p_my_model, model_names_to_train,
            args, epoch_idx=None, freeze_layers=None):
    is_verbose = False
    if is_main_process():
        is_verbose = True
        logger.info ("[***] model_names_to_train = ", model_names_to_train)
    # organizing the param() based on the key-words;
    sub_model_dict, param_all_n = select_our_fullmodel_sub_modules(
                                            p_my_model, freeze_layers, is_verbose)
    assert 'depth' in model_names_to_train, "depth module not found!!!"
    keys_submodules = [
        'pose_encoder', 'pose',
        'fnet',
        'cnet',
        'gru',
        'attention',
        'ada_bins',
        'f1_attention',
        'conf', # depth confidence head;
        ]

    if epoch_idx is not None:
        scheduler_step_size = args.scheduler_step_size
        gamma = args.lr_gamma
        assert isinstance(scheduler_step_size, list), "require type(scheduler_step_size)=list"
        scheduler_step_size.sort() # sorted in Ascending order
        # adjust lr by epoch_idx
        def find_idx(epoch_idx, scheduler_step_size):
            #scheduler_step_size.sort()
            for i, p in enumerate(scheduler_step_size):
                if epoch_idx < p:
                    return i
            return i+1
        idx = find_idx(epoch_idx, scheduler_step_size)

    parameters_to_train = []
    ## We would like to keep the order of modles, and this order will
    # serve as kind of index, e.g., accessing the lr of those params
    # and write them to tensorboard summary;
    for name in keys_submodules:
        #if name == 'fnet' or name == 'cnet':
        #    lr_name = 'learning_rate'
        #else:
        #    lr_name = f'learning_rate_{name}'
        if 'pose' in name:
            lr_name = 'learning_rate_pose'
        else:
            lr_name = f'learning_rate_{name}'
        # get initial lr
        if hasattr(args, lr_name):
            init_lr = getattr(args, lr_name)
            if init_lr <= 0:
                init_lr = args.learning_rate
        else:
            init_lr = args.learning_rate

        lr = init_lr * (gamma ** idx) if epoch_idx is not None else init_lr
        if is_verbose:
            print ("\t==> Re-organized layer {} : lr = {:.2e}".format(name, lr))
        parameters_to_train += [
            {
                #'params': sub_model_dict[name].parameters(),
                'params': torch.nn.ParameterList(
                    filter(lambda p: p.requires_grad, sub_model_dict[name].parameters() )
                    ),
                'lr': lr,
                'my_name': name,
            }
            ]
    grad_para_n = 0
    for pp in parameters_to_train:
        grad_para_n += count_parameters(pp['params'])[0]
    if is_verbose:
        print ("checking valid param (i.e. requires_grad==True): #= {}, vs invalid # = {}".format(
           grad_para_n, param_all_n - grad_para_n
        ))
    return parameters_to_train


def select_our_fullmodel_sub_modules(my_model, freeze_layers = [], is_verbose=False):
    parent_module = "depth."

    keys_to_select_submodules = {
        'pose_encoder': ['pose_encoder.encoder'],
        'pose': ['pose.squeeze', 'pose.pose_0', 'pose.pose_1', 'pose.pose_2'],
        'fnet': ['feature_extractor', 'feature_shrinker', 'feature_fusion', 'spf.'],
        'ada_bins': [
                 # adaptive cascaded: 3 stages
                 'adaptive_bins_layer', 'atten_conv_out.', 'offset.',
                 ],
        'cnet': ['cnet'],
        'gru': ['update_block.encoder', 'update_block.flow_head',
                'update_block.mask', 'prob_net',
                # 3-gru
                'update_block.gru08', 'update_block.gru16', 'update_block.gru32',
                'context_zqr_convs',
                # 1-gru
                'update_block.gru.',
               ],
        'attention': ['update_block.aggregator.', '.att.'],
        
        'f1_attention': ['f1_att.', 'f1_aggregator.'],
        'conf': ['confidence_head.'],
    }
    sub_model_dict = {}
    sub_count_dict = {}

    count = count_parameters(my_model)
    if is_verbose:
        print ("Parsing model.named_parameters():")
        print ("   ==> full model: parameters # = valid/all = {}/{}".format(
            count[0], count[1]))
    if 0 and is_verbose:
        for name, param in my_model.named_parameters():
            print ("{} : param.requires_grad = {}".format(name, param.requires_grad))
            # >> : e.g., depth.update_block.gru32.convz.weight : param.requires_grad = True

    sum_n = 0
    frozen_n = 0
    for key, sub_ones in keys_to_select_submodules.items():
        assert isinstance(sub_ones, list)
        tmp_params = torch.nn.ParameterList()
        for sub_name in sub_ones:
            if is_verbose:
                print ("checking ", sub_name) # e.g., 'feature_extractor'
            for p_name, param in my_model.named_parameters():
                if sub_name in p_name:
                    #if sub_name == 'update_block.aggregator.' or sub_name == '.att.':
                    #    print ("???", sub_name, p_name)
                    tmp_params.append(param)

        sub_model_dict[key] = tmp_params
        if freeze_layers and (key in freeze_layers):
            for p_name, param in sub_model_dict[key].named_parameters():
                param.requires_grad = False
            if is_verbose:
                print ("froze module ", key)

        tmp_n, tmp_n2 =  count_parameters(tmp_params)
        frozen_n += (tmp_n2 - tmp_n)
        sum_n += tmp_n
        sub_count_dict[key] = tmp_n
        if is_verbose:
            print (f"  ==> We select submodel {key}, with params # = {tmp_n}(val)/{tmp_n2}(all)")

    if is_verbose:
        print (sub_count_dict)
        #print (frozen_n)
    assert sum_n == count[1]-frozen_n, "{} != {}".format(sum_n, count[1]-frozen_n)
    return sub_model_dict, count[1]


def load_3gru_with_gma_from_no_gma_pth(my_model_gma, model_no_gma_dict):
    if isinstance(my_model_gma, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        prefix_key = 'module.'
    else:
        prefix_key = ''

    if 'state_dict' in model_no_gma_dict:
        pretrained_dict = model_no_gma_dict['state_dict']
    else:
        pretrained_dict = model_no_gma_dict

    my_model_gma_dict =  my_model_gma.state_dict()

    names = [   "depth.update_block.gru08.convz.weight",
                "depth.update_block.gru08.convr.weight",
                "depth.update_block.gru08.convq.weight",
            ]
    names = [prefix_key + n for n in names]

    skip_names = [
        "depth.update_block.aggregator.gamma",
        "depth.update_block.aggregator.gamma",
        "depth.update_block.aggregator.to_v.weight",
        "depth.att.to_qk.weight",
        ]
    skip_names = [prefix_key + n for n in skip_names]

    for k, v in my_model_gma_dict.items():
        message = "{} : {} ".format(k, v.shape)
        # weight
        if k in names:
            # torch.Size([128, 384, 3, 3]) ==> torch.Size([128, 512, 3, 3])
            # diff = 128
            v_0 = pretrained_dict[k]
            rand_tmp = torch.rand(v.size(0), v.size(1)-v_0.size(1), v.size(2), v.size(3)).to(v.device)
            new_val = torch.cat((v_0, rand_tmp), dim=1)
            # update
            my_model_gma_dict[k] = new_val
            message += " to {}".format(new_val.shape)

        elif k in skip_names:
            message += " skipped"
            pass
        else:
            message += " well loaded"
            my_model_gma_dict[k] = pretrained_dict[k]
        #print (message)

    my_model_gma.load_state_dict(my_model_gma_dict, strict=True)
    print ("Successfully loaded ckpt from model_no_gma !!!")


def load_from_eighth_scale_raft_pth(my_model, eighth_model):
    if 'state_dict' in eighth_model:
        pretrained_dict = eighth_model['state_dict']
    else:
        pretrained_dict = eighth_model
    new_dict = {}
    def is_skip(k):
        if ".downsample" in k:
            return True
        elif "module.depth.update_block.mask" in k:
            return True
        else:
            return False

    for k, v in pretrained_dict.items():
        if is_skip(k):
            continue
        new_dict[k] = v

    my_model.load_state_dict(new_dict, strict=False)
    print ("[~~~] loading eighth_scale raft done!")

