"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import sys
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2
import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from collections.abc import defaultdict
except ImportError:
    from collections import defaultdict

import tqdm
from datetime import datetime
import time
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image
import PIL.Image as pil
import glob
from path import Path
# profiler: measure the time and memory consumption of the model’s operators.
from torch.profiler import profile, record_function, ProfilerActivity


### In order to correctly call third_party/* ###
### modules, and do not destory the original import format ###
### inside the maskrcnn_benchmark python files ###
sys.path.append('third_parties/DeepVideoMVS')
sys.path.append('third_parties/ESTDepth')
sys.path.append('third_parties/IterMVS')
sys.path.append('third_parties/RAFT_Stereo')

""" This project related """
from src.options import MVSdepthOptions
from src.models import __models__
from config.config import Config
from src import datasets
from src.utils import pfmutil as pfm
from src.utils.utils import (
    readlines, write_to_file,
    count_parameters
)
from src.evaluate_util import (
    compute_errors_v2,
    compute_errors_batch,
    save_metrics_to_csv_file,
    pil_loader,
    kitti_colormap,
    manydepth_colormap,
    )
from src.utils import flow_viz
from src.datasets.dtu_yao import half_and_crop_depth

#cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)



splits_dir = "splits"



bad_x_dict = {
    "scannet": [0.5, 0.8, 1.0], # in meter
    "7scenes": [0.5, 0.8, 1.0], # in meter
    "rgbdscenesv2": [0.5, 0.8, 1.0], # in meter
    "tumrgbd": [0.5, 0.8, 1.0], # in meter
    "vkt2": [1.0, 2.0, 3.0], # in meter
    "kitti": [1.0, 2.0, 3.0], # in meter
    "dtu": [0.25, 0.5, 1.0], # in millimeters (mm)
    "eth3d": [0.5, 0.8, 1.0], # in meter
}

eval_depth_hw_dict = {
    #'eth3d': (1280, 1920), # high-res data;
    'eth3d_yao_eval': (-1, -1), # low-res data;
    'dtu_yao_eval': (1200, 1600),
}


def save_scannet_to_disk(result_dir_in, for_disk_depths, gt_depth_paths, rgb_img_paths=None):
    n = len(for_disk_depths)
    assert len(gt_depth_paths) == n, "Should be same len()"

    result_dir = os.path.join(result_dir_in, 'pfms_4eval')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for idx in range(0, n):
        depth_pred = for_disk_depths[idx].astype(np.float32)
        # assuming:
        # == */scene0000_00/frames/depth/*.png
        cur_path = gt_depth_paths[idx]
        scene, img_idx = cur_path.split("/")[-4], int(cur_path.split("/")[-1][:-len('.png')])
        to_save_path = os.path.join(result_dir, scene)
        if not os.path.exists(to_save_path):
            os.makedirs(to_save_path)
        pfm.save(os.path.join(to_save_path, "%06d.pfm"%(img_idx)), depth_pred)
    print ("Saved {} pfm depth maps to {} !!!".format(n, result_dir))

    if rgb_img_paths is not None:
        assert len(rgb_img_paths) == n, "Should be same len()"
        result_dir = Path(result_dir_in) /'rgb_dep_4bundleFusion'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            os.makedirs(result_dir / "depth")
            os.makedirs(result_dir / "rgb")
        for idx in range(0, n):
            # assuming:
            # == */scene0000_00/frames/depth/*.png
            cur_path = gt_depth_paths[idx]
            scene, img_idx = cur_path.split("/")[-4], int(cur_path.split("/")[-1][:-len('.png')])
            # save png depth
            depth_pred_png = np.uint16(for_disk_depths[idx].astype(np.float32)*3000)
            cv2.imwrite(result_dir / "depth"/ "%06d.png"%(img_idx), depth_pred_png)

            img_path = rgb_img_paths[idx]
            # in cases GT or pred depth might be down-sampled
            img = pil_loader(img_path).resize((depth_pred_png.shape[1], depth_pred_png.shape[0]), pil.NEAREST)
            img.save(result_dir / "rgb"/ "%06d.png"%(img_idx))

        print ("Saved {} png depth and png rgb to {} !!!".format(n, result_dir))


def save_to_video(video_name, for_video_depth, for_video_img, is_kitti_color=False, maxval=-1):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #scale = 1
    #scale = 2
    h, w = for_video_img[0].shape[0:2]
    board_wid = 10
    board1 = np.zeros((h, board_wid, 3)).astype(np.uint8)
    #hh, ww = 1054, 3358
    hh, ww = h, 2*w + board_wid
    #out = cv2.VideoWriter('results/fig-plot/qual-dep-with-err.avi',fourcc,
    out = cv2.VideoWriter(video_name + '.avi', fourcc,
                10.0,# fps
                (ww, hh) # frameSize
            )
    for idx in range(len(for_video_img)):
        img = for_video_img[idx]
        depth = for_video_depth[idx]
        if is_kitti_color:
            depth = kitti_colormap(1.0/depth, maxval)
        else:
            depth = manydepth_colormap(depth, maxval)
        collage = np.concatenate((img, board1, depth), 1)
        out.write(collage[:,:,::-1]) # for opencv GBR order;
    out.release()
    print ("[***] saved video {}".format(video_name + ".avi"))


def save_to_mp4_video(video_name, for_video_depth, for_video_img,
                    is_kitti_color=False,
                    is_no_color = False,
                    maxval=-1,
                    is_inverse_depth=False
                    ):

    #fourcc = cv2.VideoWriter_fourcc(*'MP4V') # OpenCV: FFMPEG: tag 0x5634504d/'MP4V' could not be supported;
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Then fallback to use tag 0x7634706d/'mp4v'
    h, w = for_video_img[0].shape[0:2]
    board_wid = 10
    board1 = np.zeros((h, board_wid, 3)).astype(np.uint8)
    hh, ww = h, 2*w + board_wid
    #out = cv2.VideoWriter('results/fig-plot/qual-dep-with-err.avi',fourcc,
    if is_inverse_depth:
        video_name += '_disp'
    if is_no_color:
        video_name += '_noclr'

    #fps = 20
    #fps = 30
    #fps = 10
    for fps in [10, 20, 30]:
        out = cv2.VideoWriter(video_name + '_fps{}'.format(fps) + '.mp4', fourcc,
                    fps,# fps
                    (ww, hh) # frameSize
            )
        for idx in range(len(for_video_img)):
            img = for_video_img[idx]
            depth = for_video_depth[idx]
            if is_inverse_depth:
                depth = 1.0 / depth

            if not is_no_color:
                if is_kitti_color:
                    depth = kitti_colormap(depth, maxval)
                else:
                    depth = manydepth_colormap(depth, maxval)
            else:
                depth = np.repeat(depth, axis=-1)

            collage = np.concatenate((img, board1, depth), 1)
            out.write(collage[:,:,::-1]) # for opencv GBR order;
        out.release()
    print ("[***] saved video {}".format(video_name + ".mp4"))


def print_metrics(mean_errors, bad_x_thred, mean_median_aligned_errors = None):
    print("\n  " + ("{:>9} | " * 12).format( "abs", "abs_inv", "abs_rel", "sq_rel", "rmse",
                                            "rmse_log", "a1", "a2", "a3",
                                            "bad-%.1f"%(bad_x_thred[0]),
                                            "bad-%.1f"%(bad_x_thred[1]),
                                            "bad-%.1f"%(bad_x_thred[2]),
                                            ))
    print( ("&{: 9.5f}  " * 12).format( * mean_errors.tolist()) + "\\\\" )
    if mean_median_aligned_errors is not None:
        print (" ==> mean_median_aligned_errors")
        print( ("&{: 9.5f}  " * 12).format( * mean_median_aligned_errors.tolist()) + "\\\\" )

    print( "\n-> Done!" )


def run_evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    num_gpu = torch.cuda.device_count()
    assert num_gpu == 1, "For error metric calculation and collection, we only use DP code with 1 GPU. " \
        "Multi-GPUs senario has not beed tested yet by us!"
    MIN_DEPTH = opt.eval_valid_depth_min
    MAX_DEPTH = opt.eval_valid_depth_max
    assert opt.eval_valid_depth_min >= 0.01 and opt.eval_valid_depth_max <= 100, "General: invalid range!"
    depth_scale_factor = 1.0
    if opt.dataset in ["dtu_yao", "dtu_yao_eval"]:
        MIN_DEPTH *= 1000
        MAX_DEPTH *= 1000
        depth_scale_factor = 1000.0 # change m to mm, since depth_gt is in mm;

    if opt.dataset in ["kitti", "vkt2"]:
        opt.disable_median_scaling = False


    GT_POINTS_THRED=100

    # find which model definition;
    if opt.network_class_name in ['riav_mvs', 'riav_mvs_cas']:
        assert opt.raft_mvs_type in [
            "raft_mvs", 'raft_mvs_gma',
            'raft_mvs_asyatt_f1_att',
            'raft_mvs_sysatt_f1_f2_att',
            'raft_mvs_casbins',
            'raft_mvs_adabins'
            ], f"Wrong raft_mvs_type={opt.raft_mvs_type} found!"
        class_name = opt.network_class_name
        if not class_name.endswith('_eval'):
            class_name = class_name + "_eval"
        print ("Will load our model {}".format(opt.network_class_name))
    
    elif opt.network_class_name in [
            "bl_pairnet", "bl_mvsnet",
            "bl_itermvs", "bl_estdepth"]:
        class_name = opt.network_class_name
        if opt.network_sub_class_name:
            class_name = opt.network_sub_class_name
        if not class_name.endswith('_eval'):
            class_name = class_name + "_eval"

    else:
        print ("Wrong class_name {}".format(opt.network_class_name))
        raise NotImplementedError

    mvsModel = __models__[class_name]
    print ("[!!!] loaded model {}.{} !!!".format(
        opt.network_class_name, class_name))
    

    if 'scannet' in opt.dataset:
        bad_x_thred = bad_x_dict['scannet']
    elif '7scenes' in opt.dataset:
        bad_x_thred = bad_x_dict['7scenes']
    elif 'tumrgbd' in opt.dataset:
        bad_x_thred = bad_x_dict['tumrgbd']
    elif 'rgbdscenesv2' in opt.dataset:
        bad_x_thred = bad_x_dict['rgbdscenesv2']
    elif 'vkt2' in opt.dataset:
        bad_x_thred = bad_x_dict['vkt2']
    elif 'kitti' in opt.dataset:
        bad_x_thred = bad_x_dict['kitti']
    elif 'dtu' in opt.dataset:
        bad_x_thred = bad_x_dict['dtu']
    elif 'eth3d' in opt.dataset:
        bad_x_thred = bad_x_dict['eth3d']
    else:
        raise NotImplementedError

    print (" will run testing ...")
    # setup models
    print ("[***] For testing, we just use regular torch.nn.DataParallel !!!")

    if opt.load_weights_path: # load '.tar' file;
        assert os.path.isfile(opt.load_weights_path), \
            "Cannot find a ckpt at {}".format(opt.load_weights_path)

        print("[***] -> Try to load weights from {}".format(opt.load_weights_path))
        model_ckpt_path = opt.load_weights_path
        model_dict = torch.load(model_ckpt_path)
        ckpt_toload = model_dict['state_dict'] if 'state_dict' in model_dict else model_dict['model']
        #import pdb
        #pdb.set_trace()
        if 0:
            tmp_key = 'module.depth.f1_aggregator'
            ckpt_toload[f'{tmp_key}.value_conv.weight'] = ckpt_toload[f'{tmp_key}.to_v.weight']
            ckpt_toload[f'{tmp_key}.scale_factor'] = ckpt_toload[f'{tmp_key}.gamma']
            ckpt_toload[f'{tmp_key}.projection_layer.weight'] = ckpt_toload[f'{tmp_key}.project.weight']
            del ckpt_toload[f'{tmp_key}.to_v.weight']
            del ckpt_toload[f'{tmp_key}.gamma']
            del ckpt_toload[f'{tmp_key}.project.weight']

        my_model = mvsModel(opt)
        my_model = torch.nn.DataParallel(my_model).cuda()

        #if "module.depth_tracker" not in model_dict['state_dict']:
        #    depth_tracker_np = np.array([model_dict.get('min_depth_bin'),
        #                                model_dict.get('max_depth_bin')]).astype(np.float32)
        #    model_dict['state_dict']["module.depth_tracker"] = torch.from_numpy(depth_tracker_np)

        if 0:
            tmp_key = 'module.depth_tracker'
            if tmp_key in ckpt_toload:
                del ckpt_toload[f'module.depth_tracker']
        my_model.load_state_dict(ckpt_toload, strict=True)
        print("[***] -> Done! Loaded weights from {}".format(opt.load_weights_path))
        # torch.save({
        #     'model' : my_model.state_dict(),
        #     }, 
        #     "checkpoints_nfs/saved/released/riavmvs_full_tmp.pth.tar"
        #     )
        # sys.exit()


    else:
        # reset opt.min_depth_bin, opt.max_depth_bin. from checkpoint file;
        # which will be used to build the model;
        my_model = mvsModel(opt)
        my_model = torch.nn.DataParallel(my_model).cuda()
        print("[~~~~~~~~~~~~~~] -> Do not load any ckpts, so as to use initialized sub-modules")

    # to eval mode;
    my_model.eval()

    ngpus_per_node = torch.cuda.device_count()
    print("[***] we are using {} gpus for DP".format(ngpus_per_node))

    frames_to_load = opt.frame_ids.copy()
    print('[***] Loading frames: {}'.format(frames_to_load))
    matching_frames = frames_to_load[1:]
    print('[***] matching frames: {}'.format(matching_frames))

    num_scales = len(opt.scales)


    my_load_depth = True
    # Setup dataloaders
    if opt.dataset == 'cityscapes':
        raise NotImplementedError

    elif opt.dataset == 'vkt2':
        raise NotImplementedError

    elif opt.dataset == 'kitti':
        if opt.eval_split_txt_name:
            filename_txt = os.path.join(splits_dir, opt.split, opt.eval_split_txt_name)
        else:
            filename_txt = os.path.join(splits_dir, opt.split, "test_files.txt")
        tmp_kwargs = {
                'load_image_path': True,
                'load_depth_path': True,
                }
        dataset = datasets.KITTIRAWDataset(
            data_path = opt.data_path,
            filename_txt = filename_txt,
            height = opt.height,
            width = opt.width,
            frame_idxs = opt.frame_ids,
            num_scales = 1,
            is_train = False,
            img_ext = '.jpg',
            **tmp_kwargs
            )


    # for scannet testset evaluation: n3 means 1 ref + 2 src frames
    elif opt.dataset in ["scannet_n3_eval",
                        "scannet_n3_val",
                        "scannet_n3_eval_sml",
                        "scannet_n2_eval",
                        "scannet_n5_eval_sml",
                        "scannet_n5_eval",
                        ]:
        tmp_kwargs = {
            'load_depth_path': True,
            'load_image_path': True,
            #'six_digit': opt.six_digit,
            }
        dataset = datasets.MVS_Scannet_eval(
                data_path = opt.data_path,
                filename_txt_list = opt.filename_txt_list,
                height = opt.height,
                width = opt.width,
                nviews = opt.mvsdata_nviews,
                depth_min= opt.min_depth,
                depth_max= opt.max_depth,
                load_depth = my_load_depth,
                **tmp_kwargs
                )

    # for 7scenes testset evaluation: n3 means 1 ref + 2 src frames
    elif opt.dataset in ["7scenes_n3_eval",
                        "7scenes_n3_eval_sml",
                        "7scenes_n2_eval",
                        "7scenes_n2_eval_sml",
                        "7scenes_n5_eval",
                        "7scenes_n5_eval_sml",
                        ]:
        tmp_kwargs = {
            'load_depth_path': True,
            'load_image_path': True
            }
        #print ("my_load_depth = ", my_load_depth)
        #sys.exit()
        dataset = datasets.MVS_7scenes_eval(
                data_path = opt.data_path,
                filename_txt_list = opt.filename_txt_list,
                split = 'test',
                height = opt.height,
                width = opt.width,
                nviews = opt.mvsdata_nviews,
                depth_min= opt.min_depth,
                depth_max= opt.max_depth,
                load_depth = my_load_depth,
                **tmp_kwargs
                )

    # for tum_rgbd testset evaluation: n3 means 1 ref + 2 src frames
    elif opt.dataset in ["tumrgbd_n3_eval",
                        "tumrgbd_n5_eval",
                        ]:
        tmp_kwargs = {
            'load_depth_path': True,
            'load_image_path': True
            }
        #print ("my_load_depth = ", my_load_depth)
        #sys.exit()
        dataset = datasets.MVS_tumrgbd_eval(
                data_path = opt.data_path,
                filename_txt_list = opt.filename_txt_list,
                split = 'test',
                height = opt.height,
                width = opt.width,
                nviews = opt.mvsdata_nviews,
                depth_min= opt.min_depth,
                depth_max= opt.max_depth,
                load_depth = my_load_depth,
                **tmp_kwargs
                )


    # for rgbdscenesv2 testset evaluation: n3 means 1 ref + 2 src frames
    elif opt.dataset in ["rgbdscenesv2_n3_eval",
                        "rgbdscenesv2_n5_eval",
                        ]:
        tmp_kwargs = {
            'load_depth_path': True,
            'load_image_path': True
            }
        #print ("my_load_depth = ", my_load_depth)
        #sys.exit()
        dataset = datasets.MVS_rgbdscenesv2_eval(
                data_path = opt.data_path,
                filename_txt_list = opt.filename_txt_list,
                split = 'test',
                height = opt.height,
                width = opt.width,
                nviews = opt.mvsdata_nviews,
                depth_min= opt.min_depth,
                depth_max= opt.max_depth,
                load_depth = my_load_depth,
                **tmp_kwargs
                )


    elif opt.dataset == 'dtu_yao': # for validation;
        if opt.eval_split_txt_name:
            filenames = os.path.join(splits_dir, opt.split, opt.eval_split_txt_name)
        else:
            filenames = os.path.join(splits_dir, opt.split, "val.txt")
        tmp_kwargs = {
                'data_mode': 'val',
                'load_depth_path': True,
                'load_image_path': True,
                }
        dataset = datasets.MVS_DTU_Yao(opt.data_path, filenames,
                                    opt.height, opt.width,
                                    opt.mvsdata_nviews,
                                    num_scales,
                                    is_train=False,
                                    robust_train=False,
                                    #load_depth=True,
                                    load_depth = my_load_depth,
                                    depth_min= opt.min_depth,
                                    depth_max= opt.max_depth,
                                    **tmp_kwargs
                                )
    elif opt.dataset == 'dtu_yao_eval':
        if opt.eval_split_txt_name:
            filenames = os.path.join(splits_dir, opt.split, opt.eval_split_txt_name)
        else:
            filenames = os.path.join(splits_dir, opt.split, "test.txt")
        tmp_kwargs = {
                'data_mode': 'test',
                'load_depth_path': my_load_depth,
                'load_image_path': True,
                }
        dataset = datasets.MVS_DTU_Yao_eval(opt.data_path, filenames,
                                    opt.height, opt.width,
                                    opt.mvsdata_nviews,
                                    num_scales,
                                    is_train=False,
                                    robust_train=False,
                                    #load_depth=True,
                                    load_depth = my_load_depth,
                                    depth_min= opt.min_depth,
                                    depth_max= opt.max_depth,
                                    **tmp_kwargs
                                    )

    elif opt.dataset == 'eth3d_yao_eval':
        filenames = os.path.join(splits_dir, opt.split, opt.eval_split_txt_name)
        if 'val_lowres.txt' == opt.eval_split_txt_name:
            data_mode = 'val'
            my_load_depth = True
        elif 'test_lowres.txt' == opt.eval_split_txt_name:
            data_mode = 'test'
            my_load_depth = False
        else:
            raise NotImplementedError
        tmp_kwargs = {
                'data_mode': data_mode,
                'resolution': 'low-res',
                'load_depth_path': my_load_depth,
                'load_image_path': True,
                }
        dataset = datasets.MVS_ETH3D_Yao_eval(
                            opt.data_path, filenames,
                            opt.height, opt.width,
                            opt.mvsdata_nviews,
                            num_scales,
                            is_train=False,
                            robust_train=False,
                            load_depth = my_load_depth,
                            depth_min= opt.min_depth,
                            depth_max= opt.max_depth,
                            **tmp_kwargs
                            )
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    print(f"-> Computing predictions with size {opt.width}x{opt.height}")
    print(f'\tdepth = {opt.min_depth} to {opt.max_depth}; ' + \
        f'Evaluation valid depth = {opt.eval_valid_depth_min} to {opt.eval_valid_depth_max}')


    gt_depth_path = []
    imgs_path = []
    if not opt.disable_median_scaling:
        print("   Monocular (i.e., not stereo) evaluation - "
            "report two results: w/ and w/o median scaling")
    else:
        print("   Multi-view stereo evaluation - "
            "report result: w/o median scaling")
    errors = []
    median_aligned_errors = []
    meds = []

    SAVE_DEPTH_STEP = max(10, len(dataloader) // 20)
    print ("[////] SAVE_DEPTH_STEP=", SAVE_DEPTH_STEP)
    #SAVE_DEPTH_STEP = 1
    # save all depth prediction to disk;
    if opt.save_all_results:
        SAVE_DEPTH_STEP = 1

    PRINT_FREQ = max(100, len(dataloader) // 4)

    # for baseline estdepth
    pre_costs = None
    pre_cam_poses = None
    pre_scene = None
    cur_scene = None

    # do inference
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm.tqdm(dataloader)):
            is_verbose= batch_idx%PRINT_FREQ == 0
            #if is_verbose:
            #    print ("processing batch_idx {}/{}".format(batch_idx, len(dataloader)))

            invalid_pose = False
            for fi in frames_to_load:
                if torch.isnan(data[("pose", fi)]).any() or torch.isinf(data[("pose", fi)]).any():
                    invalid_pose = True
            if invalid_pose:
                continue

            # due to batch dim, data['depth_gt_path'] = list of str
            # so we use += instead of append;
            if my_load_depth:
                gt_depth_path += data[('depth_gt_path', 0)]
                gt_depth = data[("depth_gt", 0)]*depth_scale_factor
            imgs_path += data[('image_path', 0)]


            global_idxs_str = data["global_id_str"] # list of string, len() = batch_size;
            global_idxs = data["global_id"] # list of float, len() = batch_size;
            if torch.cuda.is_available():
                if my_load_depth:
                    gt_depth = gt_depth.cuda()
                    valid_mask = (gt_depth >= MIN_DEPTH) & (gt_depth <= MAX_DEPTH)
                    valid_count = valid_mask.float().sum()
                else:
                    valid_count = -1
                global_idxs = global_idxs.cuda() # list of float, len() = batch_size;
                #print ("[???] input_color shape ", input_color.shape)

            #----- profiler ----
            if 0:
                #Using profiler to analyze execution time
                with profile(activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("model_inference"):
                        res = my_model(data, frames_to_load, is_verbose)
                print(prof.key_averages().table(
                    sort_by="cuda_time_total",
                    #row_limit=10
                    ))
                #Using profiler to analyze memory consumption
                with profile(activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA],
                    profile_memory=True, record_shapes=True) as prof:
                        res = my_model(data, frames_to_load, is_verbose)
                print(prof.key_averages().table(
                    sort_by="self_cuda_memory_usage",
                    #row_limit=10
                    ))
                prof.export_chrome_trace("results/trace.json")
                sys.exit()

            #---------------------
            # run model inference
            #---------------------
            start_time = time.time()
            if 'bl_estdepth' == opt.network_class_name:
                cur_scene = data['scene']
                res = my_model(data, frames_to_load, is_verbose,
                                pre_costs = pre_costs,
                                pre_cam_poses = pre_cam_poses,
                )
                # update pre_costs and pre_cam_poses if in the same scene data;
                if pre_scene == cur_scene:
                    pre_costs = res['pre_costs']
                    pre_cam_poses = res['pre_cam_poses']
                else:
                    #print (f"new scene!! pre_scene={pre_scene}, cur_scene={cur_scene}")
                    pre_costs = None
                    pre_cam_poses = None

                # update pre_scene
                pre_scene = cur_scene




            else:
                res = my_model(data, frames_to_load, is_verbose)

            time_sofar = time.time() - start_time
            pred_depth = res['depth']*depth_scale_factor
            batch_size = pred_depth.size(0)
            
            
            if 0: # for runtime and model parameters
                if batch_idx % 10 == 0:
                    print ("times per frame = ", time_sofar / batch_size )
                    print ("frame per time = ",  batch_size / time_sofar )
                    count_parameters(my_model)
                    tmp_count = count_parameters(my_model)
                    print("[!!!!!xxxx] modle {}: Parameter Count = valid/all = {}/{}".format(
                        class_name,
                        tmp_count[0],
                        tmp_count[1]
                        ))
                    if batch_idx == 200:
                        sys.exit()
            if "confidence" in res:
                confidence = res["confidence"]
            else:
                confidence = None
            # resize to gt_depth dimension
            if my_load_depth:
                height_gt, width_gt = gt_depth.size()[2:4]
            else:
                height_gt, width_gt = eval_depth_hw_dict[opt.dataset]

            pred_depth_raw = pred_depth # before resize to GT depth;
            if height_gt > 0 and width_gt > 0:
                pred_depth = torch.nn.functional.interpolate(pred_depth,
                                        size=(height_gt, width_gt),
                                        mode='nearest'
                                        #mode='bilinear'
                                    )
            if confidence is not None:
                confidence_raw = confidence
                if height_gt > 0 and width_gt > 0:
                    confidence = torch.nn.functional.interpolate(confidence,
                                        size=(height_gt, width_gt),
                                        mode='nearest'
                                    )

            #print ("////????", valid_count)
            if my_load_depth and (valid_count > GT_POINTS_THRED):
                err = compute_errors_batch(
                        img_global_id = global_idxs, # used to find image path;
                        groundtruth = gt_depth, #[N,1,H,W]
                        prediction = pred_depth, #[N,1,H,W]
                        min_depth = MIN_DEPTH,
                        max_depth = MAX_DEPTH,
                        bad_x_thred = bad_x_thred,
                        is_median_scaling = False
                    )
                #print ("err = ", err)

                errors.append(err)

                if not opt.disable_median_scaling:
                    err2, med =  \
                            compute_errors_batch(
                                img_global_id = global_idxs, # used to find image path;
                                groundtruth = gt_depth, #[N,1,H,W]
                                prediction = pred_depth, #[N,1,H,W]
                                min_depth = opt.min_depth,
                                max_depth = opt.max_depth,
                                bad_x_thred = bad_x_thred,
                                is_median_scaling = True
                                )
                    median_aligned_errors.append(err2)
                    meds.append(med)

            # change to numpy to save to disk
            is_save_files = batch_idx % SAVE_DEPTH_STEP == 0
            if is_save_files:
                #pred_depth_np = pred_depth.squeeze(dim=1).cpu().numpy() #[N,1,H,W] --> [N, H, W]
                if opt.save_all_results:
                    # save high-res
                    pred_depth_np = pred_depth.squeeze(dim=1).cpu().numpy() #[N,1,H,W] --> [N, H, W]
                    if confidence is not None:
                        confidence_np = confidence.squeeze(dim=1).cpu().numpy()
                else:
                    # save low-res
                    pred_depth_np = pred_depth_raw.squeeze(dim=1).cpu().numpy() #[N,1,H,W] --> [N, H, W]
                    if confidence is not None:
                        confidence_np = confidence_raw.squeeze(dim=1).cpu().numpy()
                if my_load_depth:
                    gt_depth_np = gt_depth.squeeze(dim=1).cpu().numpy() #[N,H,W]
                res_fldr = opt.result_dir
                if not os.path.exists(res_fldr):
                    os.makedirs(res_fldr)
                for j in range(batch_size):
                    img_name = global_idxs_str[j]
                    d_path_pfm = '%s/d_%s.pfm'%(res_fldr, img_name)
                    pfm.save(d_path_pfm, pred_depth_np[j])
                    #print ("[***] Saved {}".format(d_path_pfm))
                    if confidence is not None:
                        conf_path_pfm = '%s/conf_%s.pfm'%(res_fldr, img_name)
                        pfm.save(conf_path_pfm, confidence_np[j])


                    #if 1:
                    if not opt.save_all_results:
                        # GT
                        if my_load_depth:
                            d_path_gt_pfm = '%s/d_gt_%s.pfm'%(res_fldr, img_name)
                            pfm.save(d_path_gt_pfm, gt_depth_np[j])

                        # RGB
                        save_img_path_png = '%s/img_%s.png'%(res_fldr, img_name)
                        # in some cases GT or pred depth might be down-sampled
                        # (H, W)
                        img = pil_loader(data[('image_path', 0)][j]).resize((pred_depth_np.shape[-1], pred_depth_np.shape[-2]), pil.NEAREST)
                        img.save(save_img_path_png)

                        """ copied from manydepth/test_simple.py """
                        # Saving colormapped depth image and cost volume argmin
                        to_plot_list = [('d_clr_%s.png'%img_name, pred_depth_np[j])]
                        if my_load_depth:
                            to_plot_list = [
                                ('d_gt_clr_%s.png'%img_name, gt_depth_np[j]),
                                ('d_clr_%s.png'%img_name, pred_depth_np[j]),
                            ]
                        for idx, (plot_name, toplot) in enumerate(to_plot_list):
                            if 0:
                                if idx != 0:
                                    _vmax=np.percentile(toplot, 95)
                                else: # GT depth, a large region has no GT values;
                                    _vmax= toplot.max()

                            if 1:
                                # use one fix max, to make color space consistence among
                                # different frames;
                                #_vmax= opt.max_depth
                                #_vmax= to_plot_list[0][1].max()
                                #_vmax=np.percentile(to_plot_list[1][1], 95)
                                #_vmax= 10.0
                                _vmax= 5.0
                            im = manydepth_colormap(toplot, maxval=_vmax*depth_scale_factor)
                            name_dest_im = os.path.join(res_fldr, plot_name)
                            im.save(name_dest_im)
                            #print("-> Saved output image to {}".format(name_dest_im))


    print("    --->  image paths")
    output_path = os.path.join(
        opt.result_dir, "{}_imgs_path.txt".format(opt.split))
    write_to_file(output_path, imgs_path)

    print('[***] Finished predicting!')
    if my_load_depth:
        errors = torch.cat(errors, dim=0)
        assert errors.size(0) == len(dataset)
        if not opt.disable_median_scaling:
            median_aligned_errors = torch.cat(median_aligned_errors, dim=0)
            meds = torch.cat(meds, dim=0)
            assert median_aligned_errors.size(0) == meds.size(0) == len(dataset)
        #print(f'[***] {errors.shape}, {median_aligned_errors.shape}, {meds.shape}!')


        #------- save to *.npy file -----
        errors = errors.cpu().numpy()
        output_path = os.path.join(
            opt.result_dir, "{}_errors.npy".format(opt.split))
        np.save(output_path, errors)

        if not opt.disable_median_scaling:
            median_aligned_errors = median_aligned_errors.cpu().numpy()
            print("[***] Saving median scaled errors")
            output_path = os.path.join(
                opt.result_dir, "{}_medscal_errors.npy".format(opt.split))
            np.save(output_path, median_aligned_errors)
            print("    --->  Saving ratios and medians")
            meds = meds.cpu().numpy()
            output_path = os.path.join(
                opt.result_dir, "{}_ratio_med.npy".format(opt.split))
            np.save(output_path, meds)


        nan_num = 0
        for j in range(errors.shape[0]):
            if any(np.isnan(errors[j])):
                nan_num += 1
                print ("nan found at idx {} /{}, = {}".format(j, errors.shape[0], errors[j]))

        no_ratio_info = "We consider MIN_MAX_D={:0.3f}/{:0.3f} meters | No scaling ratio | valid frames: {:d}/{:d} | ratio=1.0 ".format(
            MIN_DEPTH, MAX_DEPTH, len(errors)-nan_num, len(errors))
        print (no_ratio_info)

        if not opt.disable_median_scaling:
            tmp_num = len(meds)
            nan_num = 0
            ratios = meds[:, 0]
            for j in range(ratios.shape[0]):
                if np.isnan(ratios[j]):
                    nan_num += 1
            # ignoring NaNs.
            medd = np.nanmedian(ratios)
            stdd = np.nanstd(ratios/medd)
            median_ratio_info = "We consider MIN_MAX_D={:0.3f}/{:0.3f} meters | Scaling ratios | valid frames: {:d}/{:d} | med: {:0.3f} | std: {:0.3f}".format(
                MIN_DEPTH, MAX_DEPTH, tmp_num - nan_num, tmp_num, medd, stdd)
            print (median_ratio_info)


        mean_errors = np.nanmean(errors[:,:12], axis=0) # skip: the img_id at dim=12;
        if not opt.disable_median_scaling:
            mean_median_aligned_errors = np.nanmean(median_aligned_errors[:,:12], axis=0)
        else:
            mean_median_aligned_errors = None

        print_metrics(mean_errors, bad_x_thred, mean_median_aligned_errors)

        """ save as csv file, Excel file format """
        #timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if "scannet" in opt.dataset and opt.scannet_sub_test_type is not None:
            tag = opt.scannet_sub_test_type
            csv_file = os.path.join(opt.result_dir, "{}-{}-err.csv".format(opt.split, opt.scannet_sub_test_type))
        else:
            tag = ''
            csv_file = os.path.join(opt.result_dir, "{}-err.csv".format(opt.split))

        # assume the result dir is in this format:
        # "./results/mvs-pytorch-vkt2-val-D192-epo20/depth-epo-001"
        # we want to extract the last few dirs: i.e., "mvs-pytorch-vkt2-val-D192-epo20/depth-epo-001"
        tmp_pos = opt.result_dir.find("/results")
        tmp_dir = opt.result_dir[tmp_pos+1+len("/results"):]
        print ("[***] tmp_dir = ", tmp_dir)

        your_note = no_ratio_info
        if not opt.disable_median_scaling:
            your_note += ("," + median_ratio_info)

        save_metrics_to_csv_file(mean_errors, csv_file, opt.model_name,
                                dir_you_specifiy = tmp_dir,
                                bad_x_thred_list = bad_x_thred,
                                mean_median_aligned_errors = mean_median_aligned_errors,
                                your_note= your_note
                                )

        dst_csv_file = os.path.join(opt.csv_dir,
                    f'depth-err-eval-gpu{opt.machine_name}-{opt.eval_gpu_id}.csv'
                    )
        os.system(f'cat {csv_file} >> {dst_csv_file}')
        print (f"cat {csv_file} to {dst_csv_file}")

        to_save = opt.__dict__.copy()
        with open(os.path.join(opt.result_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

def run_evaluate_saved_results(opt, name_map_txt_file, pred_depth_dir):
    assert 'scannet' in opt.dataset, "Only work on ScanNet evaluation"
    MIN_DEPTH = opt.eval_valid_depth_min
    MAX_DEPTH = opt.eval_valid_depth_max
    assert opt.eval_valid_depth_min >= 0.1 and opt.eval_valid_depth_max <= 20, \
        "ScanNet: invalid evluation depth range!"
    GT_POINTS_THRED=100
    errors = []
    bad_x_dict = {
        "scannet": [0.5, 0.8, 1.0], # in meter
        "vkt2": [1.0, 2.0, 3.0], # in meter
        "dtu": [2.0, 10.0, 20.0] # in millimeters (mm)
    }

    bad_x_thred=bad_x_dict['scannet']
    file_names = readlines(name_map_txt_file)

    #file_names = file_names[:50]
    epe_per_scene = defaultdict(float)
    abs_rel_per_scene = defaultdict(float)
    valid_frame_num_scene = defaultdict(float)

    print (len(file_names), file_names[0])

    for idx, img_path in enumerate(tqdm.tqdm(file_names)):
        pos_tmp = img_path.find('scene')
        scene = img_path[pos_tmp:pos_tmp+len('scene0707_00')]
        pos_tmp2 = img_path.find('color/')
        img_name = img_path[pos_tmp2+len('color/'):]
        #print (f"find scene/img {scene}/{img_name} from img_path {img_path}")
        #if scene != 'scene0765_00':
        #    continue

        gt_filename = img_path.replace("frames/color", "frames/depth")
        gt_depth = cv2.imread(gt_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        gt_depth = gt_depth / 1000.0 # change mm to meters;
        gt_height, gt_width = gt_depth.shape[:2]
        mask = np.logical_and(gt_depth >= MIN_DEPTH, gt_depth <= MAX_DEPTH)

        pred_filename = os.path.join(pred_depth_dir, f"d_{idx:08d}.pfm")
        pred_dep = pfm.readPFM(pred_filename)
        pred_dep = cv2.resize(pred_dep, (gt_width, gt_height), interpolation=cv2.INTER_NEAREST)

        pred_depth = pred_dep[mask]
        gt_depth = gt_depth[mask]
        if mask.astype(np.float32).sum() > GT_POINTS_THRED:
            err = compute_errors_v2(gt_depth, pred_depth, \
                min_depth = MIN_DEPTH, max_depth = MAX_DEPTH, bad_x_thred=bad_x_thred)
            errors.append(err)

            if not any(np.isnan(err)):
                epe_per_scene[scene] += err[0]
                abs_rel_per_scene[scene] += err[2]
                valid_frame_num_scene[scene] += 1
                print ("idx = ", idx, gt_filename, pred_filename, f"epe={err[0]}")
    # get averaged values;
    errors = np.array(errors)

    nan_num = 0
    for j in range(errors.shape[0]):
        if any(np.isnan(errors[j])):
            nan_num += 1
            print ("nan found at idx {} /{}, = {}".format(j, errors.shape[0], errors[j]))

    no_ratio_info = "We consider MIN_MAX_D={:0.3f}/{:0.3f} meters | No scaling ratio | valid frames: {:d}/{:d} | ratio=1.0 ".format(
        MIN_DEPTH, MAX_DEPTH, len(errors)-nan_num, len(errors))
    print (no_ratio_info)

    mean_errors = np.nanmean(errors[:,:12], axis=0) # skip: the img_id at dim=12;
    print_metrics(mean_errors, bad_x_thred)

    """ save as csv file, Excel file format """
    #timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if "scannet" in opt.dataset and opt.scannet_sub_test_type is not None:
        tag = opt.scannet_sub_test_type
        csv_file = os.path.join(pred_depth_dir, "{}-{}-np-err.csv".format(opt.split, opt.scannet_sub_test_type))
    else:
        tag = ''
        csv_file = os.path.join(pred_depth_dir, "{}-np-err.csv".format(opt.split))

    # assume the result dir is in this format:
    # "./results/mvs-pytorch-vkt2-val-D192-epo20/depth-epo-001"
    # we want to extract the last few dirs: i.e., "mvs-pytorch-vkt2-val-D192-epo20/depth-epo-001"
    tmp_pos = opt.result_dir.find("/results")
    tmp_dir = opt.result_dir[tmp_pos+1+len("/results"):]

    your_note = no_ratio_info
    save_metrics_to_csv_file(mean_errors, csv_file, opt.model_name,
                            dir_you_specifiy = tmp_dir,
                            bad_x_thred_list = bad_x_thred,
                            mean_median_aligned_errors = None,
                            your_note= your_note
                            )
    # save error info per scene
    csv_file2 = os.path.join(pred_depth_dir, "{}-scene-err.csv".format(opt.split))
    with open( csv_file2, 'w') as fwrite:
        timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        messg = timeStamp + ",method={},resultDir,{}\n".format(
            opt.model_name, tmp_dir)
        fwrite.write(messg)
        fwrite.write("scene,abs,abs_rel,num_frame\n")
        for k, v in epe_per_scene.items():
            num_frame = valid_frame_num_scene[k]
            epe = epe_per_scene[k]/num_frame
            abs_rel = abs_rel_per_scene[k]/num_frame
            fwrite.write(f"{k},{epe},{abs_rel},{num_frame}\n")
        print (f"saved {csv_file2}")



    dst_csv_file = os.path.join(opt.csv_dir,
                f'depth-err-eval-gpu{opt.machine_name}-{opt.eval_gpu_id}.csv'
                )
    os.system(f'cat {csv_file} >> {dst_csv_file}')
    print (f"cat {csv_file} to {dst_csv_file}")

