"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2

import torch
from torchvision import transforms
from torch.utils.data import Dataset

""" load our own moduels """
from src.utils.utils import readlines
from src.utils.comm import print0
from .dataset_util import pil_loader


class MVSDataset(Dataset):
    def __init__(self, 
                data_path,
                filename_txt_list,
                height,
                width,
                nviews, # ref img + source imgs
                depth_min, 
                depth_max,
                load_depth,
                **kwargs
                ): 
        super(MVSDataset, self).__init__()
        
        self.data_path = data_path
        self.filename_txt = filename_txt_list
        assert self.filename_txt, "Cannot not be empty or None"
        self.height = height
        self.width = width
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.nviews = nviews
        self.load_depth = load_depth
 
        self.data_mode = 'test'
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        
        # Default image and depth files extensions
        self.depth_ext = '.png'
        self.img_ext = '.png' # data format is prepared by CCJ;
        #self.six_digit = True # {:06d}.format(f_idx)
        #self.is_train = False
        self.six_digit = kwargs.get('six_digit', True) # {:06d}.format(f_idx)
        if not self.six_digit:
            # for 'validation' only;
            self.img_ext = '.jpg'
        
        print0 (f"[***] self.img_ext = {self.img_ext}")
        self.load_depth_path = kwargs.get('load_depth_path', False)
        self.load_image_path = kwargs.get('load_image_path', False)
        
        ##  default dimensions
        self.img_shape = (640, 480) # width x height
        self.dep_shape = (640, 480) # width x height
        
        self.metas = self.build_metas()
        # first folder
        zero_scene = self.metas[0]['scene']
        depH, depW = self.get_depth_dimension(zero_scene)
        assert self.dep_shape == (depW, depH), "Wrong depth size"
        
        imgH, imgW = self.get_img_dimension(zero_scene)
        assert self.img_shape == (imgW, imgH), f"Wrong depth size, {self.img_shape} != {(imgW, imgH)}"
        
        print0 ("[***] Scannet_eval: has {} samples".format(len(self.metas)))

    def __len__(self):
        return len(self.metas)

    def build_metas(self):
        metas = []
        if isinstance(self.filename_txt, list):
            fn_txts = self.filename_txt
        elif isinstance(self.filename_txt, str):
            fn_txts = [self.filename_txt]
        else:
            raise NotImplementedError

        data_idx = 0 # global index among the metas;
        # use this data_idx to save the predicted depth 
        # in paralle, without worrying about overwrite; 
        for fn_txt in fn_txts:
            file_list = readlines(fn_txt)
            print ("Read {} samples from {}".format(len(file_list), fn_txt))
            for i in range(len(file_list)):
                # E.g., 'scene0806_00 1160 -10 10'
                line = file_list[i].split() 
                #print ("line = ", line)
                folder = line[0]
                frame_indexs = [int(i) for i in line[1:]]
                ref_idx = frame_indexs[0]
                src_idxs = frame_indexs[1:]
                if any([i < 0 for i in src_idxs]):
                    src_idxs = [ i + ref_idx for i in src_idxs]
                
                frame_indexs = [ref_idx] + src_idxs
                #print ("?? frame_indexs = ", frame_indexs)
                #NOTE: temporary 
                def check_path_ok(idx):
                    img_path = self.get_image_path(folder, idx)
                    return os.path.exists(img_path)
                if len(frame_indexs) == 3 and self.nviews==5:
                    if check_path_ok(ref_idx + 10) and check_path_ok(ref_idx+20):
                        frame_indexs += [ref_idx + 10, ref_idx + 20]
                    else:
                        # no found 5 frames
                        continue

                metas.append({
                    'scene': folder, 
                    "indices": frame_indexs,
                    "global_id": data_idx,
                    "global_id_str": f"{data_idx:08d}",
                    }
                    )
                data_idx += 1
                #print ({'scene': folder, "indices": frame_indexs})
                #sys.exit()
        print ("metas[0]:\n", metas[0])
        return metas
    
    def get_img_dimension(self, folder):
        frame_index = 0
        image_path = self.get_image_path(folder, frame_index)
        color = self.loader(image_path)
        imgW, imgH = color.size # read by PIL.Image;
        return imgH, imgW

    def get_depth_dimension(self, folder):
        frame_index = 0
        depth_path  = self.get_depth_path(folder, frame_index)
        #print ("[???] depth name = ", depth_path)
        depth = self.read_scannet_png_depth(depth_path)
        depH, depW = depth.shape # read by cv2;
        return depH, depW

    def get_depth_path(self, folder, frame_index, side=None):
        if self.six_digit:
            f_str = "{:06d}{}".format(frame_index, self.depth_ext)
        else:
            f_str = "{:d}{}".format(frame_index, self.depth_ext)
        #f_str = "{:06d}{}".format(frame_index, self.depth_ext)
        depth_path = os.path.join(
            self.data_path,
            folder,
            'frames/depth', 
            f_str)
        return depth_path

    def read_scannet_png_depth(self, depth_png_filename):
        #NOTE: The depth map in milimeters can be directly loaded
        depth = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = depth / 1000.0 # change mm to meters;
        return depth
    
    def get_depth(self, folder, frame_index):
        depth_png_filename = self.get_depth_path(folder, frame_index)
        depth_gt = self.read_scannet_png_depth(depth_png_filename)
        return depth_gt
    
    def get_image_path(self, folder, frame_index):
        if self.six_digit:
            f_str = "{:06d}{}".format(frame_index, self.img_ext)
        else:
            f_str = "{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "{}/frames/color".format(folder),
            f_str)
        return image_path

    def get_color(self, folder, frame_index):
        #print ("[???] img = ",self.get_image_path(folder, frame_index))
        color_raw = self.loader(self.get_image_path(folder, frame_index))
        # since raw_image has size (1296, 968), raw_depth in size (640, 480)
        # We have to resize raw_image to (640, 480) before croping;
        color = color_raw.resize((self.width, self.height), Image.BILINEAR)
        #print ("img raw size = ", color_raw.size, "img size = ", color.size, "img_path= ", self.get_image_path(folder, frame_index))
        return color

    def check_image_path(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])
        image_path = self.get_image_path(scene_name, frame_index)
        return os.path.isfile(image_path)

    def load_intrinsics(self,
        folder # e.g., == 'scene0000_00'
        ):
        fpath_txt = os.path.join(
            self.data_path,
            "{}/frames/intrinsic/intrinsic_depth.txt".format(folder))
        
        K = np.loadtxt(fpath_txt, dtype=np.float32).reshape((4,4)) # 4x4
        #print ("loaded K = ", K) 
        depW, depH = self.dep_shape
        # NOTE: Make sure your depth intrinsics matrix is *normalized* by depth dimension;
        K[0,:] /= depW # 1/W
        K[1,:] /= depH # 1/H
        #print ("normalized K = ", K)
        return K.copy().astype(np.float32)

    # Extrinsic Matrix
    def get_pose(self, folder, frame_index):
        f_str = "{:06d}".format(frame_index)
        fpath_txt = os.path.join(
            self.data_path,
            "{}/frames/pose/{}.txt".format(folder, f_str)
            )
        #print ("[???] reading pose from {}".format(fpath_txt))
        #NOTE: this loaded pose is actually (4x4 matrix, camera to world);
        # So we need to do inv to get the real extrinsic matric E;
        pose_cam2world = np.reshape(np.loadtxt(fpath_txt, dtype=np.float32), \
                    newshape=(4, 4))
        pose = np.linalg.inv(pose_cam2world)
        inv_pose = pose_cam2world
        #if np.isnan(pose.any()) or np.isnan(inv_pose.any()):
        #    print ("nan found at {}".format(fpath_txt))
        return pose, inv_pose
    

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
        sample = self.metas[index]
        
        scene = sample['scene']
        view_indices = sample['indices']
        inputs["global_id"] = torch.Tensor([sample['global_id']])
        inputs["global_id_str"] = sample['global_id_str']
        inputs['scene'] = sample['scene']
        assert len(view_indices) == self.nviews
        
        for i, view_idx in enumerate(view_indices):
            inputs[("color", i, 0)] = self.to_tensor(self.get_color(scene, view_idx))
            pose, pose_inv = self.get_pose(scene, view_idx)
            inputs[("pose", i)] = torch.from_numpy(pose)
            inputs[("pose_inv", i)] = torch.from_numpy(pose_inv)
            if self.load_depth:
                # save depth for each view, might do depth prediction for each view;
                depth_gt = self.get_depth(scene, view_idx)[None].astype(np.float32)
                inputs[("depth_gt", i)] = torch.from_numpy(depth_gt)

            if self.load_depth_path:
                inputs[("depth_gt_path", i)] = self.get_depth_path(scene, view_idx)
            if self.load_image_path:
                inputs[("image_path", i)] = self.get_image_path(scene, view_idx)
            
            #newly added;
            K_44 = self.load_intrinsics(scene)
            for s in [0, 1, 2, 3]:
                tmp_K = K_44.copy()
                # adjusting intrinsics to match each scale in the pyramid
                tmp_K[0, :] *= self.width // (2 ** s)
                tmp_K[1, :] *= self.height // (2 ** s)
                # multiply intrinsics and extrinsics to get projection matrix
                proj_mat = np.matmul(tmp_K, pose.copy())
                proj_mat_inv = np.linalg.inv(proj_mat)
                inputs[("proj_mat", i, s)] = torch.from_numpy(proj_mat)
                inputs[("proj_mat_inv", i, s)] = torch.from_numpy(proj_mat_inv)
        
        # normalized K: 4 x 4
        # adjusting intrinsics to match each scale in the pyramid
        for scale in [0, 1, 2, 3, 4]:
            K = self.load_intrinsics(scene)
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
        
        # for ScanNet, depth_min/max and depth_tracker are the same;
        inputs["depth_min"] = torch.Tensor([self.depth_min])
        inputs["depth_max"] = torch.Tensor([self.depth_max]) 
        inputs["min_depth_tracker"] = torch.Tensor([self.depth_min])
        inputs["max_depth_tracker"] = torch.Tensor([self.depth_max])

        return inputs

""" 
How to run this file:
- cd ~/PROJ_ROOT/
- python -m src.datasets.scannet_eval
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
    def sanity_check_train(my_dataset, data_loader_bs = 16):
        data_loader = torch.utils.data.DataLoader(
            my_dataset, data_loader_bs,
            shuffle=False, # set True for the iter() defined later;
            num_workers= 16,
            pin_memory=True,
            drop_last=False)

        N = len(data_loader)
        for batch_idx, inputs in enumerate(data_loader):
            before_op_time = time.time() 
            if not check_valid_pose(inputs):
                print ("Found invalid pose, at ", inputs[("image_path", 0)])
                #for key, value in inputs.items():
                #    if isinstance(value, torch.Tensor):
                #        print(key, value.shape)
                #    else:
                #        print(key, value)
                #return 
            if batch_idx % 200 == 0:
                print ("processing batch idx %d out of total %d" %(batch_idx, N))
        print ("Congrates! Sanity check passed!!!")
    

    #data_path = "/home/ccj/datasets/Scannet/scans_test"
    #txt_file_dir = '/home/ccj/code/proj-raft-mvs/splits/scannetv2_small/test'
    data_path = "/nfs/flash/STG/project/SemanticDenseMapping/Scannet/scans_test"
    data_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_test"
    txt_file_dir = '/nfs/STG/SemanticDenseMapping/changjiang/proj-raft-mvs/splits/scannetv2/test/test_iters_estdepth'
    filename_txt_list = [
        'test_files_iter_00.txt',
        'test_files_iter_01.txt',
        'test_files_iter_02.txt',
        'test_files_iter_03_not_used.txt'
        ]
    filename_txt_list = [ os.path.join(txt_file_dir, i) for i in filename_txt_list]
    
    data_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_val"
    filename_txt_list = [ '/home/us000182/code/proj-riav-mvs/splits/scannetv2/test/val_deepvideo_keyframe-nmeas2/val_files.txt']

    """ 3 frames (1 ref + 2 source) """
    nviews = 3 # ref img + source imgs
    #nviews = 2 # ref img + source imgs
    split = 'test' 
    #split = 'test_small'
    # wxh=320x256, default size
    height = 256
    width = 256
    #height = 480
    #width = 640
    _DEPTH_MIN = 0.25
    _DEPTH_MAX = 20.0
    kwargs = {}
    kwargs['load_depth_path'] = False
    kwargs['load_image_path'] = True
    dataset = MVSDataset( 
                data_path = data_path,
                filename_txt_list = filename_txt_list,
                height = height,
                width = width,
                nviews = nviews,
                depth_min= _DEPTH_MIN, 
                depth_max= _DEPTH_MAX,
                load_depth = True,
                **kwargs
                ) 
    print("~~~ Number of samples:", len(dataset))
    
    if 1:
        sanity_check_train(dataset, data_loader_bs = 1)
        sys.exit()
    
    inputs = dataset[10]
    
    #for key, value in inputs.items():
    #    print(key, type(value))
    #sys.exit()

    frames_to_load = list(range(0, nviews)) 
    #frames_to_load.reverse()
    print ("frames_to_load = ", frames_to_load)

    if 1:
        """ numpy version """
        f, ax = plt.subplots(4, nviews)
        
        scale = 1
        scale = 0
        ref_img = inputs[('color', 0, 0)].numpy().transpose([1, 2, 0])
        ref_img = cv2.resize(ref_img, (width//2**scale, height//2**scale), interpolation=cv2.INTER_LINEAR)
        K = inputs[("K", scale)].numpy() # 4 x 4
        print ("K ", K.shape)
        ref_Extrins = inputs[("pose", 0)].numpy() # 4 x 4
        print ("E ", ref_Extrins.shape)
        ref_proj_mat = np.matmul(K, ref_Extrins) # 4 x 4

        for i, frame_id in enumerate(frames_to_load):
            img = inputs[('color', frame_id, 0)].numpy().transpose([1, 2, 0])
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

    