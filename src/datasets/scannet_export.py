"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

# ------------------------------------------------------------------------------------
# Modified from ScanNet (https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py)
# See open license at https://github.com/ScanNet/ScanNet/blob/master/LICENSE.
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from DeepVideoMVS (https://github.com/ardaduz/deep-video-mvs)
# MIT license.
# ------------------------------------------------------------------------------------

import os
import random
import numpy as np
from multiprocessing import Pool
import copy
import os
import struct
import zlib
from itertools import groupby
import cv2
import imageio
import numpy as np
import torch


COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}

def process_color_image(color, depth, K_color, K_depth):
    old_height, old_width = np.shape(color)[0:2]
    new_height, new_width = np.shape(depth)

    x = np.linspace(0, new_width - 1, num=new_width)
    y = np.linspace(0, new_height - 1, num=new_height)
    ones = np.ones(shape=(new_height, new_width))
    x_grid, y_grid = np.meshgrid(x, y)
    warp_grid = np.stack((x_grid, y_grid, ones), axis=-1)
    warp_grid = torch.from_numpy(warp_grid).float()
    warp_grid = warp_grid.view(-1, 3).t().unsqueeze(0)

    H = K_color.dot(np.linalg.inv(K_depth))
    H = torch.from_numpy(H).float().unsqueeze(0)

    width_normalizer = old_width / 2.0
    height_normalizer = old_height / 2.0
    
    ## Torch.bmm: performs a batch matrix-matrix product of matrices,
    # and the input matrices must be 3-D tensors each containing 
    # the same number of matrices;
    warping = H.bmm(warp_grid).transpose(dim0=1, dim1=2)
    warping = warping[:, :, 0:2] / (warping[:, :, 2].unsqueeze(-1) + 1e-8)
    warping = warping.view(1, new_height, new_width, 2)
    warping[:, :, :, 0] = (warping[:, :, :, 0] - width_normalizer) / width_normalizer
    warping[:, :, :, 1] = (warping[:, :, :, 1] - height_normalizer) / height_normalizer

    image = torch.from_numpy(np.transpose(color, axes=(2, 0, 1))).float().unsqueeze(0)

    warped_image = torch.nn.functional.grid_sample(input=image,
                                                   grid=warping,
                                                   mode='nearest',
                                                   padding_mode='zeros',
                                                   align_corners=True)

    warped_image = warped_image.squeeze(0).numpy().astype(np.uint8)
    warped_image = np.transpose(warped_image, axes=(1, 2, 0))
    return warped_image


class RGBDFrame():
    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack('f' * 16, file_handle.read(16 * 4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        if 1:
            self.color_data = b''.join(struct.unpack('c' * self.color_size_bytes, file_handle.read(self.color_size_bytes)))
            self.depth_data = b''.join(struct.unpack('c' * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))
        if 0:
            self.color_data = ''.join(
                str(struct.unpack('c' * self.color_size_bytes, file_handle.read(self.color_size_bytes)))
                )
            self.depth_data = ''.join(str(
                struct.unpack('c' * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))
                )

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


def find_longest_reliable_subsequence(is_ok):
    longest_interval_length = 0
    longest_interval = None

    index = 0
    for k, g in groupby(is_ok, None):
        length = len(list(g))
        if k:
            start_index = copy.deepcopy(index)
            end_index = start_index + length
            if length > longest_interval_length:
                longest_interval_length = copy.deepcopy(length)
                longest_interval = [start_index, end_index]
        index += length
    return longest_interval


class SensorData:
    def __init__(self, filename):
        self.version = 4
        print ("\tloading {}".format(filename))
        ## requires python2
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0] # CCJ's note: 'I': unsigned int
            assert self.version == version, "got {}".format(version)
            strlen = struct.unpack('Q', f.read(8))[0] # Q: unsigned long long
            print ("??? strlen = ", strlen)
            ##add a `b` prefix to the empty string to make it a byte object;
            self.sensor_name = b''.join(struct.unpack('c' * strlen, f.read(strlen)))
            #self.sensor_name = ''.join(str(struct.unpack('c' * strlen, f.read(strlen))))
            ## c: char, bytes of length 1
            #print ("??? self.sensor_name = ", self.sensor_name)
            self.intrinsic_color = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            self.num_frames = struct.unpack('Q', f.read(8))[0]

            self.frames = []
            for i in range(self.num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)
        

    def export_train(self, output_path, frame_skip):
        counter = 0

        poses = []
        for index in range(0, len(self.frames), frame_skip):
            pose = self.frames[index].camera_to_world
            if np.any(np.isnan(pose)) or np.any(np.isinf(pose)) or np.any(np.isneginf(pose)):
                print("Pose NaN, Inf or -Inf encountered!, Skipping...")
                continue
            poses.append(np.ravel(pose).tolist())

            depth = self.frames[index].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth, dtype=np.uint16).reshape(self.depth_height, self.depth_width)

            color = self.frames[index].decompress_color(self.color_compression_type)
            color = process_color_image(color=color,
                                        depth=depth,
                                        K_color=self.intrinsic_color[0:3, 0:3],
                                        K_depth=self.intrinsic_depth[0:3, 0:3])

            output_file = os.path.join(output_path, str(counter).zfill(6))
            np.savez_compressed(output_file, image=color, depth=depth)
            counter += 1
        np.savetxt(fname=os.path.join(output_path, "poses.txt"), X=np.array(poses), fmt='%.8e')
        np.savetxt(fname=os.path.join(output_path, "K.txt"), X=self.intrinsic_depth[0:3, 0:3])

    
    # updated by CCJ:
    def export_test(self, output_path, frame_skip):
        frame_skip = 1
        base_path = os.path.join(output_path, 'frames')
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        print ('exporting', self.num_frames // frame_skip, ' frames to', output_path)
        image_folder = os.path.join(base_path, 'color')
        depth_folder = os.path.join(base_path, 'depth')
        K_folder = os.path.join(base_path, 'intrinsic')
        pose_folder = os.path.join(base_path, 'pose')

        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
        if not os.path.exists(depth_folder):
            os.mkdir(depth_folder)
        if not os.path.exists(K_folder):
            os.mkdir(K_folder)
        if not os.path.exists(pose_folder):
            os.mkdir(pose_folder)
        
        np.savetxt(fname=os.path.join(K_folder, "intrinsic_depth.txt"), X=self.intrinsic_depth)
         
        for f in range(0, self.num_frames, frame_skip):
            pose = self.frames[f].camera_to_world
            np.savetxt(fname=os.path.join(pose_folder, "{:06d}.txt".format(f)), X=pose, fmt='%.8e')
            
            depth = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth, dtype=np.uint16).reshape(self.depth_height, self.depth_width)

            color = self.frames[f].decompress_color(self.color_compression_type)
            color = process_color_image(color=color,
                                        depth=depth,
                                        K_color=self.intrinsic_color[0:3, 0:3],
                                        K_depth=self.intrinsic_depth[0:3, 0:3])
        
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(image_folder, str(f).zfill(6) + '.png'), color, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            cv2.imwrite(os.path.join(depth_folder, str(f).zfill(6) + '.png'), depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])

def export_samples(scene_path):
    scene_name = scene_path.split("/")[-1]
    scene_output_path = os.path.join(output_path, scene_name)
    if not os.path.exists(scene_output_path):
        # load the data
        #print ('loading sensor data for %s' % scene_path)
        print ('%s' % scene_name)
        sd = SensorData(os.path.join(scene_path, scene_name + ".sens"))
        os.mkdir(scene_output_path)
        if is_train:
            sd.export_train(scene_output_path, frame_skip=frame_skip)
        else:
            sd.export_test(scene_output_path, frame_skip=frame_skip)
    else:
        #print ('existing scene %s, skipping...' % scene_path)
        pass


def sanity_check_test():
    exported_scenes = sorted(os.listdir(output_path))

    for exported_scene in exported_scenes:

        n_images = len(os.listdir(os.path.join(output_path, exported_scene, "images")))
        n_depths = len(os.listdir(os.path.join(output_path, exported_scene, "depth")))
        n_poses = len(np.loadtxt(os.path.join(output_path, exported_scene, "poses.txt")))

        if n_images != n_poses or n_images != n_depths or n_depths != n_poses:
            print (exported_scene, "is problematic")


def sanity_check_train():
    exported_scenes = sorted(os.listdir(output_path))

    for exported_scene in exported_scenes:
        if ".txt" not in exported_scene:
            n_poses = len(np.loadtxt(os.path.join(output_path, exported_scene, "poses.txt")))
            K = np.loadtxt(os.path.join(output_path, exported_scene, "K.txt"))
            n_files = len(os.listdir(os.path.join(output_path, exported_scene)))

            if n_files - 2 != n_poses:
                print (exported_scene, "is problematic")

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    lines = [l for l in lines if not l.startswith('#')]
    return lines

is_train = False
is_valid = True
#is_train = True
frame_skip = 4 if is_train else 1

is_sanity_check = False
#is_sanity_check = True
if is_train:
    input_path = "/nfs/SHARE/dataset/scannet_data/scans"
    #input_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans"
    #output_path = "/mnt/Backup2/changjiang/data/Scannet/other_exported_train/deepvideomvs"
    output_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/other_exported_train/deepvideomvs"
else:
    # test
    input_path = "/mnt/Data/jiachenliu/scannet_data/scans_test"
    input_path = "/nfs/SHARE/dataset/scannet_data/scans_test"
    #output_path = "/mnt/Backup2/changjiang/data/Scannet/scans_test"
    output_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_test"
    
    # validation
    if is_valid:
        val_txt = '/nfs/STG/SemanticDenseMapping/changjiang/proj-raft-mvs/splits/scannet_benchmark/val_files.txt'
        input_path = "/nfs/SHARE/dataset/scannet_data/scans"
        #input_path = "/nfs/STG/SemanticDenseMapping/jiachenliu/scannet_data/scans"
        output_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_val"

if __name__ == '__main__':

    if is_sanity_check:
        if is_train:
            sanity_check_train()
        else:
            sanity_check_test()
        exit(0)

    sequence_names = sorted(os.listdir(input_path))
    if is_valid:
        sequence_names = readlines(val_txt) 
    print ("sequence_names: # {}, including: {}, ... {}".format(
        len(sequence_names), sequence_names[0], sequence_names[-1]))

    if is_train:
        scene_names_dict = dict()
        for sequence_name in sequence_names:
            scene_name, idx = sequence_name.split('_')

            if scene_name in scene_names_dict:
                scene_names_dict[scene_name].append(idx)
            else:
                scene_names_dict[scene_name] = [idx]

        scene_names = list(scene_names_dict.keys()) # added list() for Python3 by CCJ;

        random.seed(123)
        random.shuffle(scene_names)
        n_scenes = len(scene_names)
        n_training = int(n_scenes * 0.9)
        training_scenes = scene_names[:n_training]
        validation_scenes = scene_names[n_training:]

        training_sequences = []
        for training_scene in training_scenes:
            idxs = scene_names_dict[training_scene]
            for idx in idxs:
                training_sequences.append(training_scene + "_" + idx)

        validation_sequences = []
        for validation_scene in validation_scenes:
            idxs = scene_names_dict[validation_scene]
            for idx in idxs:
                validation_sequences.append(validation_scene + "_" + idx)

        np.savetxt(os.path.join(output_path, "train.txt"), np.array(training_sequences), fmt='%s')
        np.savetxt(os.path.join(output_path, "validation.txt"), np.array(validation_sequences), fmt='%s')

    sequence_names.sort()
    print ("sequence_names: # {}, including {} ... {}".format(len(sequence_names), sequence_names[0], sequence_names[-1]))
    sequence_paths = []
    for index, sequence_name in enumerate(sequence_names):
        sequence_paths.append(os.path.join(input_path, sequence_name))


    #sequence_paths = sequence_paths[:4]
    if 1:
        pool = Pool(6)
        pool.map(export_samples, sequence_paths)
        pool.join()
        pool.close()
    if 0:
        for s in sequence_paths:
            export_samples(s)