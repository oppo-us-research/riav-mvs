"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import random
from datetime import datetime
import numpy as np
import PIL.Image as pil

import os
import cv2
from path import Path
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from kornia import adjust_brightness, adjust_gamma, adjust_contrast

from .dataset_util import crawl
from src.utils.utils import readlines, write_to_file


class PreprocessImage:
    def __init__(self, K, old_width, old_height, 
        new_width, new_height, 
        distortion_crop=0, perform_crop=True):
        
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

        self.new_width = new_width
        self.new_height = new_height
        self.perform_crop = perform_crop

        original_height = np.copy(old_height)
        original_width = np.copy(old_width)

        if self.perform_crop:
            old_height -= 2 * distortion_crop
            old_width -= 2 * distortion_crop

            old_aspect_ratio = float(old_width) / float(old_height)
            new_aspect_ratio = float(new_width) / float(new_height)

            if old_aspect_ratio > new_aspect_ratio:
                # we should crop horizontally to decrease image width
                target_width = old_height * new_aspect_ratio
                self.crop_x = int(np.floor((old_width - target_width) / 2.0)) + distortion_crop
                self.crop_y = distortion_crop
            else:
                # we should crop vertically to decrease image height
                target_height = old_width / new_aspect_ratio
                self.crop_x = distortion_crop
                self.crop_y = int(np.floor((old_height - target_height) / 2.0)) + distortion_crop

            self.cx -= self.crop_x
            self.cy -= self.crop_y
            intermediate_height = original_height - 2 * self.crop_y
            intermediate_width = original_width - 2 * self.crop_x

            factor_x = float(new_width) / float(intermediate_width)
            factor_y = float(new_height) / float(intermediate_height)

            self.fx *= factor_x
            self.fy *= factor_y
            self.cx *= factor_x
            self.cy *= factor_y
        else:
            self.crop_x = 0
            self.crop_y = 0
            factor_x = float(new_width) / float(original_width)
            factor_y = float(new_height) / float(original_height)

            self.fx *= factor_x
            self.fy *= factor_y
            self.cx *= factor_x
            self.cy *= factor_y

    def apply_depth(self, depth):
        raw_height, raw_width = depth.shape
        #print("raw depth shape = ", depth.shape)
        cropped_depth = depth[self.crop_y:raw_height - self.crop_y, self.crop_x:raw_width - self.crop_x]
        resized_cropped_depth = cv2.resize(cropped_depth, (self.new_width, self.new_height), interpolation=cv2.INTER_NEAREST)
        return resized_cropped_depth

    def apply_rgb(self, image, scale_rgb, mean_rgb, std_rgb, normalize_colors=True):
        raw_height, raw_width, _ = image.shape
        #print("raw image shape = ", image.shape)
        cropped_image = image[self.crop_y:raw_height - self.crop_y, self.crop_x:raw_width - self.crop_x, :]
        cropped_image = cv2.resize(cropped_image, (self.new_width, self.new_height), interpolation=cv2.INTER_LINEAR)

        if normalize_colors:
            cropped_image = cropped_image / scale_rgb
            cropped_image[:, :, 0] = (cropped_image[:, :, 0] - mean_rgb[0]) / std_rgb[0]
            cropped_image[:, :, 1] = (cropped_image[:, :, 1] - mean_rgb[1]) / std_rgb[1]
            cropped_image[:, :, 2] = (cropped_image[:, :, 2] - mean_rgb[2]) / std_rgb[2]
        return cropped_image

    def get_updated_intrinsics(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]]
                        )


class MVSDataset(Dataset):
    def __init__(self, 
                data_path,
                filename_txt,
                split,
                height,
                width,
                nviews, # ref img + source imgs
                #robust_train, # scale to adjust depth;
                depth_min, 
                depth_max,
                **kwargs
                ): 
        super(MVSDataset, self).__init__()
        
        self.data_path = data_path
        self.filename_txt = filename_txt
        self.height = height
        self.width = width
        self.depth_min = depth_min
        self.depth_max = depth_max
        #self.robust_train = robust_train
        #self.num_scales = [1,2,3] #1/2,1/4,1/8
        #self.interp = InterpolationMode.BILINEAR 
        self.nviews = nviews
        assert self.nviews in [2,3,5], "We preper either 2, 3 or 5 frames"

        self.split = split
        assert self.split in ["train", "val", 
                    "train_small", "val_small"
                    #"test_small", "test_small"
                    ] # for test, use scannet_eval.py dataloader;
        self.is_train = (split in ['train', "train_small"])
        print (f"[**] Scannet-NPZ: is_train={self.is_train}, split={self.split}")
        
        self.load_depth_path = kwargs.get('load_depth_path', False)
        self.load_image_path = kwargs.get('load_image_path', False)
        #self.read_flow =  kwargs.get('read_flow', False)
        self.splitfile_dir = kwargs.get('splitfile_dir', None)
        
        self.train_crawl_step = kwargs.get('train_crawl_step', 3)
        self.min_pose_distance = kwargs.get('min_pose_distance', 0.125)
        self.max_pose_distance = kwargs.get('max_pose_distance', 0.325)
        self.num_workers = kwargs.get('num_workers', 8)
        
        #self.resize = {}
        #for i in self.num_scales:
        #    s = 2 ** i
        #    self.resize[i] = transforms.Resize((self.height // s, self.width // s),
        #                                       interpolation=self.interp)
        
        # other args 
        self.geometric_scale_augmentation = kwargs.get('geometric_scale_augmentation', True)
        self.distortion_crop = kwargs.get('distortion_crop', 10)
        if not self.is_train:
            self.geometric_scale_augmentation = False
            print ("[***] For val, disable geometric_scale_augmentation")
        else:
            print ("[***] For train, enable geometric_scale_augmentation")
            seed = kwargs.get('seed', 1234)
            random.seed(seed)
        
        self.train_predict_two_way = kwargs.get('train_predict_two_way', True)
         
        # Default image and depth files extensions
        self.depth_ext = ".npz"
        self.img_ext = ".npz"
 
        ##  default dimensions
        self.img_full_shape = (640, 480) # width x height
        self.dep_full_shape = (640, 480) # width x height
        
 
        self.metas = self.build_metas()
        
        # first folder
        zero_scene = self.metas[0]['scene']
        depH, depW = self.get_depth_dimension(zero_scene)
        assert self.dep_full_shape == (depW, depH), "Wrong depth size"
        


    def __len__(self):
        return len(self.metas)
    
    def get_depth_dimension(self, folder):
        frame_index = 0
        depth_path  = self.get_depth_path(folder, frame_index)
        depth = np.load(depth_path)['depth']
        depH, depW = depth.shape
        return depH, depW

    def get_depth_path(self, folder, frame_index):
        depth_path = Path(self.data_path) / folder / "{:06d}{}".format(frame_index, self.depth_ext)
        return depth_path

    
    def build_metas(self):
        metas = []
        if self.filename_txt:
            file_list = readlines(self.filename_txt)
            print ("Read {} samples from {}".format(len(file_list), self.filename_txt))
            for i in range(len(file_list)):
                line = file_list[i].split()
                folder = line[0]
                frame_indexs = [int(i) for i in line[1:]]
                assert any([i >= 0 for i in frame_indexs]), "Requres >=0 frame index"
                # use the middle one as ref_idx if len() = 3;
                if len(frame_indexs) == 3:
                    #print ("frame_indexs was = ", frame_indexs)
                    frame_indexs = [frame_indexs[i] for i in [1,0,2]]
                    #print ("frame_indexs now = ", frame_indexs)
                elif len(frame_indexs) == 5:
                    #print ("frame_indexs was = ", frame_indexs)
                    frame_indexs = [frame_indexs[i] for i in [2,0,1,3,4]]
                    #print ("frame_indexs now = ", frame_indexs)
                else:
                    raise NotImplementedError
                metas.append({'scene': folder, "indices": frame_indexs})
                if self.is_train and self.train_predict_two_way and len(frame_indexs) == 2:
                    frame_indexs.reverse()
                    metas.append({'scene': folder, "indices": frame_indexs})
        else:
            assert self.splitfile_dir, "Loading scenes: require dir to read train/val split"
            tmp_split_fn = Path(self.splitfile_dir) / "{}_files.txt".format(self.split)
            print ("Try to read scene list form ", tmp_split_fn)
            assert os.path.isfile(tmp_split_fn), "cannot find {}".format(tmp_split_fn)
            scenes = readlines(tmp_split_fn)
            print ("scenes list, #= ", len(scenes), ", e.g, ", scenes[0], " ... ", scenes[-1])
            metas = crawl(dataset_path= self.data_path,
                        scenes = scenes,
                        subsequence_length = self.nviews,
                        train_crawl_step = self.train_crawl_step,
                        min_pose_distance = self.min_pose_distance, 
                        max_pose_distance = self.max_pose_distance, 
                        num_workers = self.num_workers
                    )
            timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.save_metas_to_file( metas, './results/scanpz_{}_mea{}_{}.txt'.format(self.split, self.nviews-1, timeStamp))
        return metas
    
    def save_metas_to_file(self, metas, txt_fn):
        for_txt_list = []
        for index in range(len(metas)):
            sample = metas[index]
            scene = sample['scene']
            view_indices = sample['indices']
            #print ("scene={}, view_indices={}".format(scene, view_indices))
            view_num = len(view_indices)
            line = "{} ".format(scene) + ("{} "*(view_num-1)).format(*view_indices[:-1]) + "{}".format(view_indices[-1])
            for_txt_list.append(line)    
        write_to_file(txt_fn, for_txt_list)

    def __getitem__(self, index):
        #-------------------------------------------
        # to follow the API in manydepth
        #-------------------------------------------
        """
        Returns a single training item from the dataset as a dictionary.
        The dictionary contains torch tensors with keys that are 
        either strings or tuples, which indicate the type and configuration of the data:

        Keys:
            - ("color", <frame_id>, <scale>): Raw color images.
            - ("color_aug", <frame_id>, <scale>): Augmented color images.
            - ("K", scale) : Camera intrinsics.
            - ("inv_K", scale): Inverse camera intrinsics.
            - "depth_gt": Ground truth depth maps.

        Parameters:
            - <frame_id>: An integer representing the temporal step relative 
                          to the current index (e.g., 0, -1, 1).
            - <scale>: An integer indicating the image scale relative to 
                           the full-size image:
                - "-1": Native resolution as loaded from disk.
                -  "i=0, 1, 2 and 3": Resized to (self.width//2^i, self.height//2^i).
        """
        
        inputs = {}

        #do_color_aug = self.is_train and random.random() > 0.5
        
        sample = self.metas[index]
        
        scene = sample['scene']
        view_indices = sample['indices']
        scene_path = Path(self.data_path) / scene

        K = np.loadtxt(scene_path / 'K.txt', dtype=np.float32)
        #print ("Loaded K_33=\n", K)

        scene_poses = np.reshape(np.loadtxt(scene_path / 'poses.txt', dtype=np.float32), 
                                 newshape=(-1, 4, 4))
        scene_npzs = sorted(scene_path.files('*.npz'))

        
        #if self.is_train and random.random() > 0.5:
        #    view_indices.reverse()

        raw_poses = []
        raw_images = []
        raw_depths = []
        for i, view_idx in enumerate(view_indices):
            data = np.load(scene_npzs[view_idx])
            raw_images.append(data['image'])
            # use idx i (not view_idx)
            #inputs[("color", i, -1)] = data['image'] # -1: as loaded from disk;
            raw_depths.append(data['depth'])
            raw_poses.append(scene_poses[view_idx])
        
        #print ("[???] raw image size = ", raw_images[0].shape)
        if (self.img_full_shape[0] <= self.width) or (self.img_full_shape[1] <= self.height):
            my_perform_crop = False
        else:
            my_perform_crop = True
        #print ("[***] my_perform_crop = ", my_perform_crop)
        preprocessor = PreprocessImage(
                            K=K,
                            old_width=self.img_full_shape[0],
                            old_height=self.img_full_shape[1],
                            new_width= self.width,
                            new_height= self.height,
                            distortion_crop = self.distortion_crop,
                            perform_crop= my_perform_crop
                            )


        rgb_sum = 0
        min_depth_in_sequence = self.depth_min
        max_depth_in_sequence = self.depth_max
        intermediate_depths = []
        intermediate_images = []
        
        for i in range(len(view_indices)):
            depth = (raw_depths[i]).astype(np.float32) / 1000.0 # mm to meters;
            depth_nan = depth == np.nan
            depth_inf = depth == np.inf
            depth_nan_or_inf = np.logical_or(depth_inf, depth_nan)
            depth[depth_nan_or_inf] = 0
            depth = preprocessor.apply_depth(depth)
            intermediate_depths.append(depth)

            valid_mask = depth > 0
            valid_depth_values = depth[valid_mask]

            if len(valid_depth_values) > 0:
                current_min_depth = np.min(valid_depth_values)
                current_max_depth = np.max(valid_depth_values)
                min_depth_in_sequence = min(min_depth_in_sequence, current_min_depth)
                max_depth_in_sequence = max(max_depth_in_sequence, current_max_depth)

            image = raw_images[i]
            image = preprocessor.apply_rgb(image=image,
                                           scale_rgb=1.0,
                                           mean_rgb=None,
                                           std_rgb=None,
                                           normalize_colors=False
                                        )

            rgb_sum += np.sum(image)
            intermediate_images.append(image)
        rgb_average = rgb_sum / (len(raw_images) * self.height * self.width * 3)
        #print ("[???] rgb_average = ", rgb_average)

        # GEOMETRIC AUGMENTATION
        geometric_scale_factor = 1.0
        if self.is_train and self.geometric_scale_augmentation:
            possible_low_scale_value = self.depth_min / min_depth_in_sequence
            possible_high_scale_value = self.depth_max / max_depth_in_sequence
            #print ("[???] possible_low_scale_value= {}, possible_high_scale_value={}".format(
            #    possible_low_scale_value, possible_high_scale_value))
            if random.random() > 0.5:
                low = max(possible_low_scale_value, 0.666)
                high = min(possible_high_scale_value, 1.5)
            else:
                low = max(possible_low_scale_value, 0.8)
                high = min(possible_high_scale_value, 1.25)
            
            
            geometric_scale_factor = random.uniform(low, high)

            #print ("[???] low={}, high={}".format(low, high))
            #print ("[???] geometric_scale_factor=", geometric_scale_factor)

            # COLOR AUGMENTATION
            color_transforms = []
            brightness = random.uniform(-0.03, 0.03)
            contrast = random.uniform(0.8, 1.2)
            gamma = random.uniform(0.8, 1.2)
            color_transforms.append((adjust_gamma, gamma))
            color_transforms.append((adjust_contrast, contrast))
            color_transforms.append((adjust_brightness, brightness))
            random.shuffle(color_transforms)

        K = preprocessor.get_updated_intrinsics()
        #print ("Updated K due to image resize, K =\n", K)
        K_44 = np.eye(4).astype(np.float32) # 4x4
        K_44[:3,:3] = K
        
        for i, f_id in enumerate(view_indices):
            #print ("i, f_id = ", i, f_id)
            image = intermediate_images[i]
            depth = intermediate_depths[i]*geometric_scale_factor
            

            image = np.transpose(image, (2, 0, 1))

            image = torch.from_numpy(image.astype(np.float32))
            ## No imagenet normalization, 
            # here we just return image in [0, 1] by dividing 255.0;
            image = image / 255.0
            
            # save color (without augmentation) in [0, 1] range
            inputs[("color", i, 0)] = image # use idx i (not f_id)
            
            if self.is_train and (55.0 < rgb_average < 200.0):
                for (color_transform_function, color_transform_value) in color_transforms:
                    image = color_transform_function(image, color_transform_value)

            inputs[("color_aug", i, 0)] = image

            pose = raw_poses[i].astype(np.float32)
            # adjust translate t in pose matrix, since we have augmented the depth;
            pose[0:3, 3] = pose[0:3, 3] * geometric_scale_factor
            
            #NOTE: this loaded ExtM is actually (4x4 matrix, camera to world);
            # So we need to do inv to get the real extrinsic matric E;
            mat_E_inv = pose.astype(np.float32)
            mat_E = np.linalg.inv(mat_E_inv)
            inputs[("pose", i)] = torch.from_numpy(mat_E)
            inputs[("pose_inv", i)] = torch.from_numpy(mat_E_inv)
            #print ("f_id = ", f_id, ", pose =\n", mat_E, "\npose_inv =\n", mat_E_inv)
        
            for s in [0, 1, 2, 3]:
                # adjusting intrinsics to match each scale in the pyramid
                tmp_K = K_44.copy()
                tmp_K[0, :] /= (2 ** s)
                tmp_K[1, :] /= (2 ** s)
                # multiply intrinsics and extrinsics to get projection matrix
                proj_mat = np.matmul(tmp_K, mat_E)
                proj_mat_inv = np.linalg.inv(proj_mat)
                inputs[("proj_mat", i, s)] = torch.from_numpy(proj_mat)
                inputs[("proj_mat_inv", i, s)] = torch.from_numpy(proj_mat_inv)
            
            
            # save depth for each view, might do depth prediction for each view;
            depth = depth[None].astype(np.float32) # add F dim, [1, H, W]
            inputs["depth_gt", i] = torch.from_numpy(depth)
        
        inputs[("K", 0)] = torch.from_numpy(K_44)
        inputs[("inv_K", 0)] = torch.from_numpy(np.linalg.pinv(K_44))
        # We need inputs[("K", 2)] since we usually
        # do MVS feature warping in 1/4 scale;
        # We need inputs[("K", 3)] since raft
        # estimates flow in 1/8 scale;
        for scale in [1, 2, 3, 4]:
            tmp_K = K_44.copy()
            tmp_K[0:2, :] /= (2 ** scale)
            tmp_inv_K = np.linalg.pinv(tmp_K)
            inputs[("K", scale)] = torch.from_numpy(tmp_K)
            inputs[("inv_K", scale)] = torch.from_numpy(tmp_inv_K)
            #print ("(K, %d) = "%scale, inputs[("K", scale)] ) 
            #print ("(inv_K, %d) = "%scale, inputs[("inv_K", scale)] ) 
        
        
        # for ScanNet, depth_min/max and depth_tracker are the same;
        inputs["depth_min"] = torch.Tensor([self.depth_min])
        inputs["depth_max"] = torch.Tensor([self.depth_max]) 
        inputs["min_depth_tracker"] = torch.Tensor([self.depth_min])
        inputs["max_depth_tracker"] = torch.Tensor([self.depth_max])

        return inputs 


""" 
How to run this file:
- cd ~/PROJ_ROOT/
- python -m src.datasets.scannet_npz
"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.utils import pfmutil as pfm
    import sys
    import time
    from .dataset_util import warp_based_on_depth
    from src.utils.utils import kitti_colormap, monodepth_colormap
    from datetime import datetime
    
    def check_valid_pose(inputs):
        is_valid_pose = False
        for key, ipt in inputs.items():
            if isinstance(key, tuple) and 'pose' in key[0]:
                if torch.isnan(ipt).any():
                    return False
        return True
    
    ## sanity_check
    def sanity_check_train(my_dataset, data_loader_bs):
        data_loader = torch.utils.data.DataLoader(
            my_dataset, 
            data_loader_bs,
            shuffle=False, # set True for the iter() defined later;
            num_workers= 32,
            pin_memory=True,
            drop_last=False)

        N = len(data_loader)
        for batch_idx, inputs in enumerate(data_loader):
            before_op_time = time.time() 
            if not check_valid_pose(inputs):
                print ("Found invalid pose")
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        print(key, value.shape)
                    else:
                        print(key, value)
                return 
            if batch_idx % 400 == 0:
                print ("processing batch idx %d out of total %d" %(batch_idx, N))
        print ("Congrates! Sanity check passed!!!")
    
    #data_path = "/home/ccj/datasets/Scannet/other_exported_train/deepvideomvs"
    #data_path = "/mnt/Backup2/changjiang/data/Scannet/other_exported_train/deepvideomvs"
    data_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/other_exported_train/deepvideomvs"

    #split_txt_path= './splits/scannetv2'
    """ 3 frames (1 ref + 2 source) """
    nviews = 3 # ref img + source imgs
    #nviews = 2 # ref img + source imgs
    #split = 'train_small' 
    split = 'train' 
    #split = 'val'
    split_txt_path= 'splits/scannetv2/scannet_mea2_npz/train_files.txt'
    #split_txt_path= './splits/scannetv2/scannet_mea1_npz/val_files.txt'
    height = 256
    #width = 512
    width = 256
    #height = 256
    #width = 320
    #height = 480
    #width = 640
    _DEPTH_MIN = 0.25
    _DEPTH_MAX = 20.0
    kwargs = {
        #'geometric_scale_augmentation': False,
        'geometric_scale_augmentation': True,
        'distortion_crop': 5,
        'load_depth_path': True,
        'load_image_path': False,
        'splitfile_dir': './splits/scannet_benchmark',
        #'seed': 1234,
        'seed': datetime.now(),
    }
    dataset = MVSDataset( 
                data_path = data_path,
                filename_txt = split_txt_path,
                split = split,
                height = height,
                width = width,
                nviews = nviews,
                #robust_train = True, # scale to adjust depth;
                depth_min= _DEPTH_MIN, 
                depth_max= _DEPTH_MAX,
                load_depth = True,
                **kwargs
                )
        
    
    print("~~~ Number of samples:", len(dataset))
    
    if 1:
        for i, inputs in enumerate(dataset):
            if i % 1000 == 0:
                print (f"processing {i} / {len(dataset)}")
            depth_gt_0 = inputs[("depth_gt", 0)] # [N,1,H,W]
            mask_0 = (depth_gt_0 >= _DEPTH_MIN) & (depth_gt_0 <= _DEPTH_MAX)
            mask_0 = mask_0.float()
            if mask_0.mean() <= 0:
                print ("///???/// depth_gt max = ", depth_gt_0.max(), ", min = ", depth_gt_0.min())
                sys.exit()
        sys.exit()

    if 0:
        sanity_check_train(dataset, data_loader_bs = 48)
        sys.exit()
    
    inputs = dataset[10]
    frames_to_load = list(range(0, nviews)) 
    #frames_to_load.reverse()
    print ("frames_to_load = ", frames_to_load)

    if 0:
        """ numpy version """
        f, ax = plt.subplots(4, nviews)
        
        scale = 1
        #scale = 0
        ref_img = inputs[('color_aug', 0, 0)].numpy().transpose([1, 2, 0])
        ref_img = cv2.resize(ref_img, (width//2**scale, height//2**scale), interpolation=cv2.INTER_LINEAR)
        K = inputs[("K", scale)].numpy() # 4 x 4
        print ("K ", K.shape)
        ref_Extrins = inputs[("pose", 0)].numpy() # 4 x 4
        print ("E ", ref_Extrins.shape)
        ref_proj_mat = np.matmul(K, ref_Extrins) # 4 x 4

        for i, frame_id in enumerate(frames_to_load):
            img = inputs[('color_aug', frame_id, 0)].numpy().transpose([1, 2, 0])
            img = cv2.resize(img, (width//2**scale, height//2**scale), interpolation=cv2.INTER_LINEAR)
            dep =  inputs["depth_gt", frame_id].squeeze(0).numpy()
            dep = cv2.resize(dep, (width//2**scale, height//2**scale), interpolation=cv2.INTER_NEAREST)
            mask = (_DEPTH_MIN < dep) & (dep < _DEPTH_MAX)
            mask = mask.astype(np.float32)
            
            ax[0, i].imshow(img)
            if frame_id == 0:
                n = 'Ref frame %s'% frame_id
            else:
                n = 'Src frame %s'% frame_id
            ax[0, i].set_title('%s'%n)
            ax[1, i].imshow(dep, cmap='gray')
            ax[1, i].set_title('depth (gray colormap) for %s'%n)
            #ax[2, i].imshow( kitti_colormap(1.0/dep, maxval=1)) # 1/depth to mimic the kitti color for disparity map;
            #ax[2, i].set_title('depth (kitti colormap) for %s'%n)
            ax[2, i].imshow( monodepth_colormap(1.0/(dep+1e-6), normalize=False, torch_transpose=False)) # 1/depth to mimic the kitti color for disparity map;
            ax[2, i].set_title('1/depth (monodepth colormap) for %s'%n)
            
            cur_E = inputs[("pose", frame_id)].numpy() # 4 x 4
            warped_gt_depth = warp_based_on_depth(
                src_img = img,
                ref_img = ref_img,
                ref_proj_mat = ref_proj_mat,
                src_proj_mat = np.matmul(K, cur_E),
                depth = dep,
                mask= mask
                )
            ax[3, i].imshow( warped_gt_depth)
            ax[3, i].set_title('warped for %s to ref'%n)
        plt.show()
    #sys.exit()

