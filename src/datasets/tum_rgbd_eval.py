"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import cv2
import numpy as np
from PIL import Image
from path import Path
import re

import torch
from torchvision import transforms
from torch.utils.data import Dataset

""" load our own moduels """
from .dataset_util import pil_loader
from src.utils.utils import readlines, write_to_file


"""
TUM-RGBD File Formats: 
See https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats;
- The color images are stored as 640x480 8-bit RGB images in PNG format.
- The depth maps are stored as 640x480 16-bit monochrome images in PNG format.
- The color and depth images are already pre-registered using the OpenNI 
  driver from PrimeSense, i.e., the pixels in the color and depth images 
  correspond already 1:1.
- The depth images are scaled by a factor of 5000, i.e., a pixel value of 
  5000 in the depth image corresponds to a distance of 1 meter from the 
  camera, 10000 to 2 meter distance, etc. A pixel value of 0 means 
  missing value/no data.
"""


class MVSDataset(Dataset):
    def __init__(self, 
                data_path,
                filename_txt_list,
                split,
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
        #assert self.filename_txt, "Cannot not be empty or None"
        self.height = height
        self.width = width
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.nviews = nviews
        self.load_depth = load_depth
        
        self.split = split
        assert self.split in ['test', 'test_small']
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        
        # Default image and depth files extensions
        self.depth_ext = '.png'
        self.img_ext = '.png' # data format is prepared by CCJ;

        self.load_depth_path = kwargs.get('load_depth_path', False)
        self.load_image_path = kwargs.get('load_image_path', False)
        self.splitfile_dir = kwargs.get('splitfile_dir', None)
        self.frame_sampling = kwargs.get('frame_sampling', 'SimpleFrame')
        
        ##  default dimensions
        self.img_shape = (640, 480) # width x height
        self.dep_shape = (640, 480) # width x height
        self.test_seqs_list = [
            "rgbd_dataset_freiburg1_desk",
            "rgbd_dataset_freiburg1_plant",
            "rgbd_dataset_freiburg1_room",
            "rgbd_dataset_freiburg1_teddy",
            "rgbd_dataset_freiburg2_desk",
            "rgbd_dataset_freiburg2_dishes",
            "rgbd_dataset_freiburg2_large_no_loop",
            "rgbd_dataset_freiburg3_cabinet",
            "rgbd_dataset_freiburg3_long_office_household",
            "rgbd_dataset_freiburg3_nostructure_notexture_far",
            "rgbd_dataset_freiburg3_nostructure_texture_far",
            "rgbd_dataset_freiburg3_structure_notexture_far",
            "rgbd_dataset_freiburg3_structure_texture_far",
            ]
        
        #self.train_crawl_step = kwargs.get('train_crawl_step', 3)
        #self.min_pose_distance = kwargs.get('min_pose_distance', 0.125)
        #self.max_pose_distance = kwargs.get('max_pose_distance', 0.325)
        #self.num_workers = kwargs.get('num_workers', 8)
        
        self.metas = self.build_metas()
        print ("[***] Scannet_eval: has {} samples".format(len(self.metas)))
        
        # first folder
        #print ("????????", self.metas[0])
        zero_scene = self.metas[0]['scene']
        depH, depW = self.get_depth_dimension(zero_scene)
        assert self.dep_shape == (depW, depH), "Wrong depth size"
        imgH, imgW = self.get_img_dimension(zero_scene)
        assert self.img_shape == (imgW, imgH), "Wrong depth size"

    def __len__(self):
        return len(self.metas)

    def build_metas(self):
        metas = []
        if self.filename_txt:
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
                    # E.g., 'chess/seq-03,50,60,70'
                    line = file_list[i].split(",") 
                    #print ("??? reading ", line)
                    folder = line[0]
                    # get median as the ref frame idx;
                    view_idxs = [int(i) for i in line[1:]]
                    tmp_n = len(view_idxs)
                    assert tmp_n == self.nviews and tmp_n%2 == 1, "Needs odd number"
                    view_idxs = sorted(view_idxs)
                    ref_idx = view_idxs[tmp_n//2]
                    view_idxs.remove(ref_idx) # in-place;
                    src_idxs = view_idxs
                    #print (view_idxs, ref_idx, src_idxs)
                    frame_indexs = [ref_idx] + src_idxs
                    assert len(frame_indexs) == tmp_n
                    metas.append({
                        'scene': folder, 
                        "indices": frame_indexs,
                        "global_id": data_idx,
                        "global_id_str": f"{data_idx:08d}",
                        }
                        )
                    data_idx += 1
                    #print ({'scene': folder, "indices": frame_indexs})
         
        else: # simple sampling
            print ("Will simply sampling image pairs ...")
            scenes = self.test_seqs_list
            print ("scenes list, #= ", len(scenes), ", e.g, ", scenes[0], " ... ", scenes[-1])
            seq_inter= 1 
            frame_interval = 20
            #frame_interval = 10
            #frame_interval = 5
            seq_inter_new = seq_inter * frame_interval
            seqs = self.prepare_seqs(
                scene_names = scenes, 
                database = self.data_path, 
                seq_length = self.nviews, 
                interval = frame_interval, 
                seq_inter = seq_inter_new
                )
            lines = []
            for samples in seqs:
                line = "{}".format(samples[0]['scene_name'])
                for sample in samples:
                    line += ",{}".format(sample['frame_index'])
                lines.append(line)
            timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            eval_dataset = 'tumrgbd'
            seq_len = self.nviews
            write_to_file(
                f"./results/{eval_dataset}-n{seq_len}-simple{frame_interval}-seqinter{seq_inter_new}-{timeStamp}.txt",
                lines)
            metas = []
            data_idx = 0
            for i in range(len(lines)):
                # E.g., 'rgbd_dataset_freiburg1_desk,0,10,20'
                line = lines[i].split(",") 
                #print ("??? reading ", line)
                folder = line[0]
                # get median as the ref frame idx;
                view_idxs = [int(i) for i in line[1:]]
                tmp_n = len(view_idxs)
                assert tmp_n == self.nviews and tmp_n%2 == 1, "Needs odd number"
                view_idxs = sorted(view_idxs)
                ref_idx = view_idxs[tmp_n//2]
                view_idxs.remove(ref_idx) # in-place;
                src_idxs = view_idxs
                #print (view_idxs, ref_idx, src_idxs)
                frame_indexs = [ref_idx] + src_idxs
                assert len(frame_indexs) == tmp_n
                metas.append({
                    'scene': folder, 
                    "indices": frame_indexs,
                    "global_id": data_idx,
                    "global_id_str": f"{data_idx:08d}",
                    })
                data_idx += 1
        return metas

    # adopted from git-codes-study/estDepth-master/data/general_eval.py;
    def prepare_seqs(self, scene_names, database, seq_length, interval, seq_inter, eval_all = False):
        seqs = []
        def check_pose(cam_pose):
            flag = np.all(np.isfinite(cam_pose))
            return flag
        
        for scene_name in scene_names:
            img_fldr = Path(database)/scene_name
            img_names = sorted((img_fldr / "images").files("*{}".format(self.img_ext)))
            num = len(img_names)
            print (f"[***] processing img_fldr = {img_fldr}, found {num} imgs")
            
            scene_poses = np.reshape(np.loadtxt(img_fldr/'poses.txt', dtype=np.float32), newshape=(-1, 4, 4))
            if eval_all:
                start_indexs = interval
            else:
                start_indexs = 1
            
            for start_i in range(start_indexs):
                #print ("??? start_i = ", start_i )
                for i in range(start_i, num - seq_length * interval, seq_inter):
                    flag = True
                    samples = []   
                    for s_ in range(seq_length):
                        s = s_ * interval
                        img_name = img_names[i + s]
                        index = int(re.findall(r'\d+', os.path.basename(img_name))[0])
                        #print ("index = ", index)

                        img_path = '%s/images/%06d%s' % (img_fldr, index, self.img_ext)
                        dmap_path = '%s/depth/%06d%s' % (img_fldr, index, self.depth_ext)
                        pose = scene_poses[index]

                        flag = flag & check_pose(pose)

                        sample = {
                                'img_path': img_path,
                                'dmap_path': dmap_path,
                                'frame_index': index,
                                'scene_name': scene_name,
                                }
                        samples.append(sample)

                    if flag:
                        seqs.append(samples)
            #sys.exit()
        print ("[???] seqs len = ", len(seqs))
        return seqs
    
    def get_img_dimension(self, folder):
        frame_index = 0
        image_path = self.get_image_path(folder, frame_index)
        color = self.loader(image_path)
        imgW, imgH = color.size # read by PIL.Image;
        return imgH, imgW

    def get_depth_dimension(self, folder):
        frame_index = 0
        depth_path  = self.get_depth_path(folder, frame_index)
        depth = self.read_png_depth(depth_path)
        depH, depW = depth.shape # read by cv2;
        return depH, depW

    def get_depth_path(self, folder, frame_index, side=None):
        f_str = "depth/{:06d}{}".format(frame_index, self.depth_ext)
        depth_path = os.path.join(self.data_path, folder, f_str)
        return depth_path

    def read_png_depth(self, depth_png_filename):
        #NOTE: The depth map in milimeters can be directly loaded
        #print ("??? depth_png_filename = ", depth_png_filename )
        dmap = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        # the original raw depth has been converted to mm (by divided by 5) 
        # when exported to our format depth;
        depth = dmap / 1000.0 # change mm to meters;
        # NOTE: do changing from mm to meter; otherwise the 
        # self.depth_min which is in meter,
        # will kill all the depth values;
        dmask = (dmap >= self.depth_min) & (dmap <= self.depth_max) & (np.isfinite(dmap))
        dmap[~dmask] = 0
        return depth
    
    def get_depth(self, folder, frame_index):
        depth_png_filename = self.get_depth_path(folder, frame_index)
        depth_gt = self.read_png_depth(depth_png_filename)
        return depth_gt
    
    def get_image_path(self, folder, frame_index):
        f_str = "images/{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_color(self, folder, frame_index):
        #print ("[???] img = ",self.get_image_path(folder, frame_index))
        color_raw = self.loader(self.get_image_path(folder, frame_index))
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
        K_fname = os.path.join(self.data_path, folder, 'K.txt')
        K_33 = np.reshape(np.loadtxt(K_fname, dtype=np.float32), newshape=(3, 3))
        #print ("loaded K_33 = \n", K_33) 
        img_w, img_h = self.img_shape
        K = np.eye(4, dtype=np.float32)
        K[:3,:3] = K_33
        K[0,:] /= img_w # normalized by 1/W
        K[1,:] /= img_h # normalized by 1/H
        #print ("normalized K = \n", K)
        return K.copy().astype(np.float32)

    # Extrinsic Matrix
    def get_pose(self, scene_poses, frame_index):
        # Pose: camera-to-world, 4×4 matrix in homogeneous coordinates,
        # So we need to do inv to get the real extrinsic matric E;
        pose_cam2world = scene_poses[frame_index]
        pose = np.linalg.inv(pose_cam2world)
        inv_pose = pose_cam2world
        #if np.isnan(pose.any()) or np.isnan(inv_pose.any()):
        #    print ("nan found at {}".format(fpath_txt))
        return pose, inv_pose
    

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        sample = self.metas[index]
        
        scene = sample['scene']
        view_indices = sample['indices']
        if 'global_id' in sample:
            inputs["global_id"] = torch.Tensor([sample['global_id']])
        if 'global_id_str' in sample:
            inputs["global_id_str"] = sample['global_id_str']
        assert len(view_indices) == self.nviews
         
        pose_fnm = os.path.join(self.data_path, scene, 'poses.txt')
        scene_poses = np.reshape(np.loadtxt(pose_fnm, dtype=np.float32), newshape=(-1, 4, 4))
        
        for i, view_idx in enumerate(view_indices):
            inputs[("color", i, 0)] = self.to_tensor(self.get_color(scene, view_idx))
            pose, pose_inv = self.get_pose(scene_poses, view_idx)
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
        for scale in [0, 1, 2, 3]:
            K = self.load_intrinsics(scene)
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
        
        # for 7 Scenes, depth_min/max and depth_tracker are the same;
        inputs["depth_min"] = torch.Tensor([self.depth_min])
        inputs["depth_max"] = torch.Tensor([self.depth_max]) 
        inputs["min_depth_tracker"] = torch.Tensor([self.depth_min])
        inputs["max_depth_tracker"] = torch.Tensor([self.depth_max])
        
        return inputs

""" 
How to run this file:
- cd ~/PROJ_ROOT/
- python -m src.datasets.tum_rgbd_eval
"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    import time
    from .dataset_util import warp_based_on_depth
    from src.utils import pfmutil as pfm
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
                print ("Found invalid pose")
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        print(key, value.shape)
                    else:
                        print(key, value)
                return 
            if batch_idx % 200 == 0:
                print ("processing batch idx %d out of total %d" %(batch_idx, N))
        print ("Congrates! Sanity check passed!!!")
    
    data_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/tum_rgbd/exported_test"
    #data_path = "/home/ccj/datasets/tum_rgbd/exported_test"
    filename_txt_list = [
        #'splits/tumrgbd/tumrgbd-n3-simple10-seqinter10.txt',
        #'splits/tumrgbd/tumrgbd-n5-simple10-seqinter10.txt',
    ]
    
    """ 3 frames (1 ref + 2 source) """
    nviews = 3 # ref img + source imgs
    nviews = 5 # ref img + source imgs
    #split = 'test' 
    split = 'test_small'
    # wxh=320x256, default size
    height = 256
    width = 320
    _DEPTH_MIN = 0.25
    _DEPTH_MAX = 20.0
    kwargs = {}
    kwargs['load_depth_path'] = False
    kwargs['load_image_path'] = False
    
    dataset = MVSDataset( 
                data_path = data_path,
                filename_txt_list = filename_txt_list,
                split = split,
                height = height,
                width = width,
                nviews = nviews,
                depth_min= _DEPTH_MIN, 
                depth_max= _DEPTH_MAX,
                load_depth = True,
                **kwargs
                ) 
    print("~~~ Number of samples:", len(dataset))
    
    if 0:
        sanity_check_train(dataset, data_loader_bs = 4)
        sys.exit()
    
    inputs = dataset[30]
    
    for key, value in inputs.items():
        print(key, type(value))
    #sys.exit()

    frames_to_load = list(range(0, nviews)) 
    #frames_to_load.reverse()
    print ("frames_to_load = ", frames_to_load)

    if 0:
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
