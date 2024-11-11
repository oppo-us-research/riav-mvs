"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from DeepVideoMVS (https://github.com/ardaduz/deep-video-mvs)
# MIT license.
# ------------------------------------------------------------------------------------

from path import Path
import os
import math
import numpy as np
from datetime import datetime
import cv2
from PIL import Image
from path import Path
import torch
from torchvision import transforms

""" load our own moduels """
from src.utils import pfmutil as pfm
from src.datasets.mvs_dataset import MVSDatasetBase
from src.datasets.dataset_util import crawl
from src.utils.utils import readlines, write_to_file


def eth3d_low_res_simple_sampling(
    dataset_path,
    scenes,
    crawl_step,
    subsequence_length
    ):
    metas = []
    for scene in scenes:
        img_path = Path(dataset_path) / scene / "images"
        img_filenames = sorted(img_path.files("*.jpg"))
        sequence_length = len(img_filenames)
        for i in range(sequence_length):
            start = i - subsequence_length//2*crawl_step
            end = i + subsequence_length//2*crawl_step
            if 0 <= start and end < sequence_length:
                indices = [i]
                for j in range(1,subsequence_length//2+1):
                    indices.append(i-j*crawl_step)
                for j in range(1,subsequence_length//2+1):
                    indices.append(i+j*crawl_step)
                sample = {
                    'scene': scene,
                    'indices': indices 
                }
                metas.append(sample)
    return metas
            

class PreprocessImage:
    def __init__(self, K, 
        old_width, 
        old_height, 
        new_width, 
        new_height, 
        perform_crop = False,
        #seed = 0
        ):
        
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        #self.rng = np.random.RandomState(seed)
        
        self.new_width = new_width
        self.new_height = new_height

        self.perform_crop = perform_crop
        
        
        if self.perform_crop:
            # try to down-sampling the img by 2 or 4;
            scales_try = [4,3,2,1]
            scale_do = 4
            for s in scales_try:
                scale_do = s
                if old_width > s*new_width and old_height> s*new_height:
                    break
            dx = old_width - scale_do*new_width
            dy = old_height - scale_do*new_height
            self.scale_do = scale_do
            #print (f"[???] scale_do = {self.scale_do}")
            #self.crop_x = self.rng.randint(min(0, dx), max(0, dx) + 1)
            #self.crop_y = self.rng.randint(min(0, dy), max(0, dy) + 1)
            self.crop_x = int(math.ceil((old_width - new_width) / 2))
            self.crop_y = int(math.ceil((old_height - new_height) / 2))
            # adjust old size for factor_x and factor_y
            old_height = old_height - self.crop_y
            old_width = old_width - self.crop_x

        else:
            self.crop_x = 0
            self.crop_y = 0
       
        factor_x = float(new_width) / float(old_width)
        factor_y = float(new_height) / float(old_height)
        
        self.cx -= self.crop_x
        self.cy -= self.crop_y

        self.fx *= factor_x
        self.fy *= factor_y
        self.cx *= factor_x
        self.cy *= factor_y

    def apply_depth(self, depth):
        #raw_height, raw_width = depth.shape
        #print("raw depth shape = ", depth.shape)
        # starting and end index
        if self.perform_crop:
            idx_y0 = self.crop_y
            idx_y1 = self.crop_y + self.scale_do*self.new_height 
            idx_x0 = self.crop_x
            idx_x1 = self.crop_x + self.scale_do*self.new_width
            cropped_depth = depth[idx_y0:idx_y1, idx_x0: idx_x1]
            resized_depth = cv2.resize(
                cropped_depth, (self.new_width, self.new_height), 
                interpolation=cv2.INTER_NEAREST)
        else:
            resized_depth = cv2.resize(
                depth, (self.new_width, self.new_height), 
                interpolation=cv2.INTER_NEAREST)
        
        return resized_depth

    def apply_rgb(self, pil_image):
        #raw_width, raw_height = pil_image.size
        #print("raw image shape = ", pil_image.size)
        if self.perform_crop:
            # Setting the points for cropped pillow image
            top = self.crop_y
            bottom = self.crop_y + self.scale_do*self.new_height 
            left = self.crop_x
            right = self.crop_x + self.scale_do*self.new_width
            cropped_img = pil_image.crop((left, top, right, bottom))
            resized_img = cropped_img.resize(
                            (self.new_width, self.new_height), Image.ANTIALIAS)
        else:
            resized_img = pil_image.resize(
                            (self.new_width, self.new_height), Image.ANTIALIAS)
        return resized_img

    def get_updated_intrinsics(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]]
                        )

class MVSDataset(MVSDatasetBase):
    def __init__(self, *args, **kwargs):
        super(MVSDataset, self).__init__(*args, **kwargs)
        self.num_scales = 4
        
        #assert (self.width, self.height) == (1920, 1280), \
        #    "eth3d requires img size (1920, 1280)"   
        #assert (self.width, self.height) == (1920//2, 1280//2), \
        #    "eth3d requires img size (960, 640)"   
        
        self.data_mode = kwargs.get('data_mode', 'test')
        self.seed = kwargs.get('seed', 0)
        self.perform_crop = kwargs.get('perform_crop', False)
        self.resolution = kwargs.get('resolution', ['high-res','low-res']) # image resolution;
        assert self.data_mode in ["test", 'val', 'train']
        print ("[***] eth3d: self.data_mode=", self.data_mode)

        assert self.height%32==0 and self.width%32==0, \
                'img_w and img_h must both be multiples of 32!'
        
        self.eval_width =  kwargs.get('eval_width', 1920)
        self.eval_height =  kwargs.get('eval_height', 1280)
        print (f"[***] ETH3D: self.data_mode={self.data_mode}, resolution={self.resolution}\n" + \
               f"      input img size={self.width}x{self.height}\n" + \
               f"      gt depth for evaluation size = {self.eval_width}x{self.eval_height}")
        
        
        eth3d_scans_dict = {}
        eth3d_scans_dict['high-res/train'] = [
            # 13 scenes
            'courtyard', 'delivery_area', 'electro', 'facade',
            'kicker', 'meadow', 'office', 'pipes', 'playground',
            'relief', 'relief_2', 'terrace', 'terrains'
            ]
         
        eth3d_scans_dict['high-res/test'] = [
            # 12 scenes
            'botanical_garden', 'boulders', 'bridge', 'door',
            'exhibition_hall', 'lecture_room', 'living_room', 'lounge',
            'observatory', 'old_computer', 'statue', 'terrace_2'
            ]

        eth3d_scans_dict['low-res/train'] = [
            'delivery_area', 'electro', 'forest', 'playground', 'terrains' 
            ]
        
        eth3d_scans_dict['low-res/test'] = [
            'lakeside', 'sand_box', 'storage_room', 'storage_room_2', 'tunnel'
            ]
        self.eth3d_scans_dict = eth3d_scans_dict
        
        self.mm_to_meter_factor = 1.0 # eth3d already in meter depth; 
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=self.interp)
        
        # crawling low-res image sequences;
        self.train_crawl_step = kwargs.get('train_crawl_step', 1)
        self.min_pose_distance = kwargs.get('min_pose_distance', 0.125)
        self.max_pose_distance = kwargs.get('max_pose_distance', 0.325)
        self.num_workers = kwargs.get('num_workers', 4)
        # low-res
        self.low_res_filetxt= kwargs.get('low_res_filetxt', None)
        
        self.metas = self.build_metas()
        #assert self.load_depth == False
        #assert self.load_depth_path == False

    
    # low-res data, which is video data (i.e., frame sequence)
    def build_metas_low_res(self, 
                scans, data_idx_init=0, 
                low_res_filetxt=None, 
                #is_pose_dist_heuristic=False
                is_pose_dist_heuristic=True
                ):
        # dummy light condition
        dummy_light_idx = 0
        timeStamp = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        if not low_res_filetxt:
            data_idx = data_idx_init # global index among the metas;
            
            # low-res, have image sequences as ScanNet, 
            # we do sampling every N frames;
            print ("low-res scans list, #= ", len(scans), 
                    ", e.g, ", scans[0], " ... ", scans[-1])
            if is_pose_dist_heuristic:
                crawl_metas = crawl(dataset_path = self.data_path,
                            scenes = scans,
                            subsequence_length = self.nviews+1,
                            train_crawl_step = self.train_crawl_step,
                            min_pose_distance = self.min_pose_distance, 
                            max_pose_distance = self.max_pose_distance, 
                            num_workers = self.num_workers
                        )
                self.save_metas_to_file(crawl_metas, './results/eth3d_lowres_keyframe_n{:02d}_{}.txt'.format(self.nviews, timeStamp))
            else: 
                # simple sampling every self.nviews frames;
                crawl_metas = eth3d_low_res_simple_sampling(
                        dataset_path = self.data_path,
                        scenes = scans,
                        crawl_step = 4,
                        subsequence_length = self.nviews
                        )
                self.save_metas_to_file(crawl_metas, './results/eth3d_lowres_simple_n{:02d}_{}.txt'.format(self.nviews, timeStamp))
            
            metas = []
            # change this data format (actually crawl is for ScanNet NPZ) 
            # to that format in high-res;
            for index in range(len(crawl_metas)):
                sample = crawl_metas[index]
                scan = sample['scene']
                view_indices = sample['indices']
                ref_view = view_indices[0]
                src_views = view_indices[1:]
                if len(src_views) != 0:
                    metas += [(scan, dummy_light_idx, ref_view, src_views, data_idx)]
                    data_idx += 1
            print(f"Low-res built metas: {self.data_mode}, smaples # = {len(metas)}")
        
        else:
            metas = []
            file_list = readlines(low_res_filetxt)
            print ("Low-res eth3d read {} samples from {}".format(len(file_list), low_res_filetxt))
            data_idx = data_idx_init # global index among the metas;
            for i in range(len(file_list)):
                line = file_list[i].split()
                scan = line[0]
                ref_view = int(line[1])
                src_views = [int(i) for i in line[2:]]
                if len(src_views) != 0:
                    metas += [(scan, dummy_light_idx, ref_view, src_views, data_idx)]
                    data_idx += 1

        #print (metas)
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
    
    def build_metas_high_res(self, scans, data_idx_init= 0):
        metas = []
        if scans:
            print ("high-res scans list, #= ", len(scans), 
                    ", e.g, ", scans[0], " ... ", scans[-1])
        # dummy light condition
        dummy_light_idx = 0
        data_idx = data_idx_init # global index among the metas;
        for scan in scans:
            # for high-res, have pair.txt
            with open(os.path.join(self.data_path, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) != 0:
                        metas += [(scan, dummy_light_idx, ref_view, src_views, data_idx)]
                        data_idx += 1
                    
        print(f"High-res metas: {self.data_mode}, smaples # = {len(metas)}")
        #print (metas)
        return metas
    
    def build_metas(self):
        data_idx_init= 0
        with open(self.filenames) as f:
            scans = [line.rstrip() for line in f.readlines() if not line.startswith('#')]
        high_res_scans = [line for line in scans if 'high-res' in line]
        low_res_scans = [line for line in scans if 'low-res' in line]
        metas = []
        if high_res_scans:
            metas += self.build_metas_high_res(high_res_scans, data_idx_init)
            data_idx_init += len(metas)
        if low_res_scans:
            metas += self.build_metas_low_res(
                low_res_scans,
                data_idx_init,
                low_res_filetxt= self.low_res_filetxt 
                )
        return metas

    def read_cam_file(self, scan, view_idx):
        cam_filename = Path(self.data_path)/f'{scan}/cams_1/{view_idx:08d}_cam.txt'
        with open(cam_filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3)) # 3 x 3
        
        if len(lines) > 11: # for high-res
            depth_min = float(lines[11].split()[0])
            if depth_min < 0:
                depth_min = 1
            depth_max = float(lines[11].split()[-1])
        else: # for low-res
            depth_min = 0.1
            depth_max = 80.0
            #depth_min = -1
            #depth_max = -1
        
        original_h, original_w = self.get_color_size(scan, view_idx)
    
        self.preprocessor = PreprocessImage(
            K = intrinsics, 
            old_width = original_w, 
            old_height= original_h, 
            new_width = self.width, 
            new_height = self.height, 
            perform_crop = self.perform_crop,
            seed = self.seed
            )
       
        intrinsics_new = self.preprocessor.get_updated_intrinsics() # 3 x 3
        # NOTE: Make sure your depth intrinsics matrix 
        # is *normalized* by depth dimension;
        K = np.eye(4, dtype=np.float32)
        K[:2,:3] = intrinsics_new[:2,:3]
        K[0,:] /= self.width # normalized by 1/W
        K[1,:] /= self.height # normalized by 1/H
        #print ("normalized K = \n", K) 
        return K, extrinsics, depth_min, depth_max

    def get_color(self, scan, view_idx, dummy_light_idx=0):
        img_filename = self.get_image_path(scan, view_idx)
        color = self.loader(img_filename)
        #original_w, original_h = color.size
        #color = color.resize((self.width, self.height), Image.ANTIALIAS)
        color = self.preprocessor.apply_rgb(color)
        return color
     
    def get_color_size(self, scan, view_idx):
        img_filename = self.get_image_path(scan, view_idx)
        color = self.loader(img_filename)
        original_w, original_h = color.size
        return original_h, original_w
    
    def get_image_path(self, scan, view_idx, dummy_light_idx=0):
        # no light_idx for testset;
        img_filename = Path(self.data_path)/f'{scan}/images/{view_idx:08d}.jpg'
        return img_filename
    

    def get_depth_path(self, scan, view_idx):
        depth_filename = Path(self.data_path)/f'{scan}/depth/{view_idx:08d}.pfm'
        dummy_mask_filename = ''
        return depth_filename, dummy_mask_filename

    def get_depth(self, scan, view_idx, scale):
        # read pfm depth file
        depth_filename, _ = self.get_depth_path(scan, view_idx)
        depth_hr = pfm.readPFM(depth_filename)*scale # high resolution
        depth = self.preprocessor.apply_depth(depth_hr)
        mask = (depth > 0.5).astype(np.float32)
        return depth, mask
    
    def __len__(self):
        return len(self.metas)

"""
How to run this file:
- cd ~/manydepth-study/
- python -m src.datasets.eth3d
"""
if __name__ == "__main__":
    import time
    import sys
    import matplotlib.pyplot as plt
    from src.utils.utils import readlines, kitti_colormap
    from src.utils import pfmutil as pfm
    import time
    import sys
    from src.datasets.dataset_util import warp_based_on_depth

    # Data loading code
    data_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/eth3d_multi_view/eth3d_undistorted_ours"
    #data_path = "/home/ccj/datasets/eth3d_multi-view/eth3d_undistorted_ours"
    data_mode = 'train'
    #train_filenames = './splits/eth3d/{}_sml.txt'.format(data_mode)
    train_filenames = './splits/eth3d/train_lowres.txt'
    train_filenames = './splits/eth3d/train.txt'

    
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
            my_dataset, data_loader_bs,
            shuffle=False, # set True for the iter() defined later;
            num_workers= 4,
            pin_memory=True,
            drop_last=False)

        N = len(data_loader)
        max_depth = -1
        min_depth = 2000
        for batch_idx, inputs in enumerate(data_loader):
            before_op_time = time.time()
            if max_depth < inputs["depth_max"]:
                max_depth =  inputs["depth_max"]
            if min_depth > inputs["depth_min"]:
                min_depth =  inputs["depth_min"]
            
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
            
            print ("image_path={}, depth min={}, max={}".format(inputs[("image_path", 0)], inputs["depth_min"], inputs["depth_max"]))

        print ("Congrates! Sanity check passed!!!")
        print (min_depth, max_depth)

    height = 512*2
    width = 640*2

    frames_to_load = [0, 1, 2, 3, 4, 5, 6]
    nviews = len(frames_to_load)
    #num_scales = 4
    num_scales = 1

    kwargs = {
        'data_mode': 'train',
        'resolution': 'high-res',
        'load_depth_path': True,
        'load_image_path': True,
        #'perform_crop': True,
        'perform_crop': False,
        'low_res_filetxt': 'splits/eth3d/low-res/eth3d_lowres_keyframe_n08_2022-04-24_05_49_06.txt',
    }
    dataset = MVSDataset(
        data_path,
        train_filenames,
        height,
        width,
        nviews,
        num_scales,
        is_train= False,
        robust_train = False,
        load_depth = True,
        depth_min= 0.1,
        depth_max=80.0,
        **kwargs
        )
    
    if 0:
        sanity_check_train(dataset, data_loader_bs = 1)
        sys.exit()
    
    value_scale = 1.0
    #_DEPTH_MIN = value_scale* 425 # mm to m
    #_DEPTH_MAX = value_scale* 935 # mm to m

    print("Number of samples:", len(dataset))

    batch_idx = 33
    inputs = dataset[batch_idx]
    #for key, value in inputs.items():
    #    print(key, type(value))


    if 0:
        """ numpy version """
        ref_idx = 0
        f, ax = plt.subplots(4, nviews)
        #scale = 2
        scale = 0
        ref_img = inputs[('color', 0, scale)].numpy().transpose([1, 2, 0])
        print ("ref image_path = ", inputs[("image_path", ref_idx)])
        dep =  inputs[("depth_gt", ref_idx)].squeeze(0).numpy()
        mask = inputs[("depth_mask", ref_idx)].squeeze(0).numpy()
        #dep = value_scale*inputs[(f"dep_gt_level_{scale}", ref_idx)].squeeze(0).numpy()
        #mask = inputs[(f"dep_mask_level_{scale}", ref_idx)].squeeze(0).numpy()
        dep_min = value_scale*inputs['depth_min']
        dep_max = value_scale*inputs['depth_max']
        print (f"depth_min={dep_min}, max={dep_max}")

        print ("???? dep = ", dep)
        print ("???? mask = ", mask)

        K = inputs[("K", ref_idx, scale)].numpy() # 4 x 4
        print ("K ", K.shape)
        ref_Extrins = inputs[("pose", 0)].numpy() # 4 x 4
        print ("E ", ref_Extrins.shape)
        ref_proj_mat = np.matmul(K, ref_Extrins) # 4 x 4
        for i, frame_id in enumerate(frames_to_load):
            img = inputs[('color_aug', frame_id, scale)].numpy().transpose([1, 2, 0])
            ax[0, i].imshow(img)
            if frame_id == 0:
                n = 'scale 1/%d: Ref frm %s'% (2**scale, frame_id)
                n0 = n
            else:
                n = 'scale 1/%d: Src frm %s'% (2**scale, frame_id)
            ax[0, i].set_title('%s'%n)
            ax[1, i].imshow(dep, cmap='gray')
            ax[1, i].set_title('depth for %s'%n0)
            ax[2, i].imshow( kitti_colormap(1.0/(dep+1e-6))) # 1/depth to mimic the kitti color for disparity map;
            ax[2, i].set_title('depth (kitti clr) for %s'%n0)
            cur_E = inputs[("pose", frame_id)].numpy() # 4 x 4
            warped_gt_depth = warp_based_on_depth(
                src_img = img,
                ref_img = ref_img,
                ref_proj_mat = ref_proj_mat,
                src_proj_mat = np.matmul(K, cur_E),
                depth = dep/value_scale,
                #mask= mask
                )
            ax[3, i].imshow( warped_gt_depth)
            ax[3, i].set_title('warped for %s to ref'%n)
        #plt.savefig("./results/tmp/plt_bt%d.png"%(batch_idx))
        plt.show()
    #sys.exit()
