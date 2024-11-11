#from collections import defaultdict
try:
    from collections.abc import defaultdict
except ImportError:
    from collections import defaultdict
import os
import sys

from path import Path

import numpy as np
import random

def check_pose(cam_pose):
    flag = np.all(np.isfinite(cam_pose))
    return flag

"""
# run this file:
# assume project_root="~/code/proj-raft-mvs"
# cd ~/code/proj-raft-mvs/
# python3 -m splits.gen_scannet_merging
"""

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    lines = [l for l in lines if not l.startswith('#')]
    return lines

def write_to_file(txt_fn, files_list):
    with open(txt_fn, 'w') as fw:
        for l in files_list:
            fw.write(l + "\n")
    print("Done! Saved {} names to {}".format(len(files_list), txt_fn))

if __name__ == "__main__":

    split = 'scannetv2'

    comments = "#scene_name frame_idx previous_frame_idx next_frame_idx"
     
    """ load DeepVideoMVS paper test_split """
    if 0:
        idx = 0
        total_omit_num = 0
        total_done_num = 0
        saved_txt_list = []
        #for scene in range(707, 708): # 100 scenes in ScanNet test set;
        for scene in range(707, 807): # 100 scenes in ScanNet test set;
            dvmvs_test_list = "keyframe+scannet+scene{:04d}_00+nmeas+1".format(scene)
            #frame_list = readlines("./splits/scannetv2/deepvideomvs_test_list/frame-selection-indices/{}".format(dvmvs_test_list))
            frame_list = readlines("/home/us000182/code/manydepth-study/splits/scannetv2/deepvideomvs_test_list/frame-selection-indices/{}".format(dvmvs_test_list))
            #print ("frame_list = len ", len(frame_list), "including: ", frame_list[0:10])
            num_to_do = len(frame_list)
            for i in frame_list:
                #print (i.split(" ")[0])
                if i == "TRACKING LOST":
                    num_to_do -= 1
                    continue
                #E.g.,: 000061.png 000043.png
                ref_idx, src_idx = i.split(" ")[0:2]
                ref_idx, src_idx = int(ref_idx[:-len(".png")]), int(src_idx[:-len(".png")])
                saved_txt_list.append(f"scene{scene:04d}_00 {ref_idx} {src_idx-ref_idx}")


        # saved to few larger chunk txt files
        n = len(saved_txt_list)
        print ("[***] found {} frames".format(n)) # found 19443 frames;
        # saved to 3 or 4 files;
        #step_len = 10000
        step_len = len(saved_txt_list) // 3
        tmp_sum = 0
        
        split_fpath = os.path.join("splits/" + split + "/test_deepvideo_keyframe-nmeas1")
        if not os.path.exists(split_fpath):
            os.makedirs(split_fpath)
        mode = 'test' 
        for idx, i in enumerate(range(0, len(saved_txt_list), step_len)):
            tmp_list = saved_txt_list[i: i+step_len]
            tmp_sum += len(tmp_list)
            fpath = os.path.join(split_fpath, "{}_files_iter_{:02d}.txt")
            write_to_file(fpath.format(mode, idx), tmp_list)
        # check the last iteration
        if (i+1)*step_len < len(saved_txt_list):
            tmp_list = saved_txt_list[(i+1)*step_len:]
            tmp_sum += len(tmp_list)
            fpath = os.path.join(split_fpath, "{}_files_iter_{:02d}.txt")
            write_to_file(fpath.format(mode, idx+1), tmp_list)
        assert tmp_sum == len(saved_txt_list), "Cannot finish all the frames, finished {}, but should do {}".format(tmp_sum, len(saved_txt_list))
        print ("saved {} small txts, with total #frames = {}".format(idx+1, tmp_sum))
        #sys.exit() 


    """ load and merge DeepVideoMVS paper test_split """
    if 0:
        idx = 0
        total_omit_num = 0
        total_done_num = 0
        split_indices_type = 'keyframe-nmeas2'
        #split_indices_type = 'simple10-nmeas2'
        
        saved_txt_list = []
        keyframe_index_files = sorted((Path('./manydepth/baselines/deep_video_mvs/sample-data/') / split_indices_type / "indices").files())
        #for scene in range(707, 708): # 100 scenes in ScanNet test set;
        for i in range(len(keyframe_index_files)): # 100 scenes in ScanNet test set;
        #for i in range(1): # 100 scenes in ScanNet test set;
            keyframing_type, dataset_name, scene_name, _, n_measurement_frames = keyframe_index_files[i].split("/")[-1].split("+")
            print("Predicting for scene:", dataset_name + "-" + scene_name, " - ", i, "/", len(keyframe_index_files))
            frame_list = readlines(keyframe_index_files[i])
            print ("frame_list = len ", len(frame_list), "including: ", frame_list[0:10])
            num_to_do = len(frame_list)
            # E.g., 000028.png 000000.png
            scene_frame_map = defaultdict(set)
            num_done = 0
            for i in frame_list:
                #print (i.split(" ")[0])
                if i == "TRACKING LOST":
                    num_to_do -= 1
                    continue
                ref_idx, src_idxs = i.split(" ")[0], i.split(" ")[1:]
                ref_idx = int(ref_idx[:-len(".png")])
                src_idxs = [int(src_idx[:-len(".png")]) for src_idx in src_idxs]
                #print ('ref_idx = ', ref_idx)
                #sys.exit()
                if ref_idx not in scene_frame_map[scene_name]:
                    scene_frame_map[scene_name].add(ref_idx)
                
                #print (scene_frame_map[scene_name])
                # E.g., scene_name frame_idx previous_frame_idx next_frame_idx
                #       scene0355_00 321 -6 6
                if len(src_idxs) == 2: # skip just 1 src frame;
                    cur_line = "{} {} {} {}".format(scene_name, ref_idx, src_idxs[0]-ref_idx, src_idxs[1]-ref_idx)
                    saved_txt_list.append(cur_line)
                    num_done += 1

            total_omit_num += num_to_do - num_done
            total_done_num += num_done
            print ("scene {} : omiting #frames={}".format(scene_name, num_to_do - num_done))
            idx += 1

        #saved_txt_list = [comments] + saved_txt_list
        print ("Parsing {}: Done #frames={}".format(split_indices_type ,total_done_num))
        print ("Parsing {}: omiting #frames={}".format(split_indices_type, total_omit_num))
        
        # test set is quite large, we save them to several txt files,
        # such that they can be processed within limited memory;
        step_len = len(saved_txt_list) // 3
        tmp_sum = 0
        
        split_fpath = os.path.join("splits/" + split + "/test_deepvideo_{}".format(split_indices_type) )
        if not os.path.exists(split_fpath):
            os.makedirs(split_fpath)
        
        for idx, i in enumerate(range(0, len(saved_txt_list), step_len)):
            tmp_list = saved_txt_list[i: i+step_len]
            tmp_sum += len(tmp_list)
            fpath = os.path.join(split_fpath, "{}_files_iter_{:02d}.txt")
            write_to_file(fpath.format(mode, idx), tmp_list)
        # check the last iteration
        if (i+1)*step_len < len(saved_txt_list):
            tmp_list = saved_txt_list[(i+1)*step_len:]
            tmp_sum += len(tmp_list)
            fpath = os.path.join(split_fpath, "{}_files_iter_{:02d}.txt")
            write_to_file(fpath.format(mode, idx+1), tmp_list)
        assert tmp_sum == len(saved_txt_list), "Cannot finish all the frames, finished {}, but should do {}".format(tmp_sum, len(saved_txt_list))
        print ("saved {} small txts, with total #frames = {}".format(idx+1, tmp_sum))
        print ("omiting #frames={}".format(total_omit_num))


    
    """ load ESTM paper test_split """
    if 1:
        import glob
        INTERVAL = 10
        #INTERVAL = 20
        START_IDX=0
        idx = 0
        total_omit_num = 0
        total_done_num = 0
        interval = INTERVAL
        start_idx = START_IDX
        img_fldr = "/mnt/Backup2/changjiang/data/Scannet/scans_test"
        img_fldr = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_test"
        mode = 'test'

        saved_txt_list = []
        num_frames = 5
        #num_frames = 3
        
        sum_samples = 0
        #for scene_idx in range(707, 708): # 100 scenes in ScanNet test set;
        #for scene_idx in range(711, 712): # 100 scenes in ScanNet test set;
        for idx,scene_idx in enumerate(range(707, 807)): # 100 scenes in ScanNet test set;
            scene = "scene{:04d}_00".format(scene_idx)
            img_names = glob.glob(os.path.join(img_fldr, scene, 'frames/color/*'))
            img_idx_names = [ int(i.split("/")[-1][:-len(".png")]) for i in img_names ]
            img_idx_names = sorted(img_idx_names) #sorted
            num = len(img_names)
            if idx % 10 == 0:
                print(f"processing {idx+1}/100", scene, "cur # samples = ", sum_samples)
            if scene_idx == 711:
                print ("num = ", num)
                #sys.exit()

            #frames_to_load = [0, -1*interval, 1*interval]
            if num_frames == 3:
                frames_to_load = [0, -1*interval, 1*interval]
            elif num_frames == 5:
                frames_to_load = [0, -2*interval, -1*interval, interval, 2*interval]
            else:
                raise NotImplementedError

            valid_pose_dict = defaultdict(int)
            for i in range(start_idx, num, interval):
                is_valid = False
                idxs_to_load = [i + tmp for tmp in frames_to_load]
                for f_idx in frames_to_load:
                    #print (i, i+f_idx)
                    if i+f_idx < 0 or i+f_idx>=num:
                        is_valid = False
                        #print ("???here")
                        break
                    elif valid_pose_dict[i+f_idx] == -1:
                        is_valid = False
                        #print ("???here2")
                        break
                    else:
                        pose_path = os.path.join(img_fldr, scene, 'frames/pose/%06d.txt'% (i+f_idx))
                        if not os.path.exists(pose_path):
                            #print (f"??? Not existing {pose_path}")
                            is_valid = False
                            break
                        else:
                            is_valid = check_pose(np.loadtxt(pose_path))
                            if not is_valid:
                                print ("??? not valid pose")
                        if not is_valid:
                            total_omit_num += 1
                        valid_pose_dict[i+f_idx] = 1 if is_valid else -1                    
                
                
                if is_valid:
                    line = "{} {} ".format(scene, i) + ("{} "*(len(frames_to_load)-2)).format(*frames_to_load[1:-1]) \
                        + "{}".format(frames_to_load[-1])

                    #print ("line = ", line)
                    #sys.exit()
                    saved_txt_list.append(line)
                    sum_samples += 1


        # test set is quite large, we save them to several txt files,
        # such that they can be processed within limited memory;
        step_len = 10000
        step_len = 4000
        step_len = len(saved_txt_list) // 3
        tmp_sum = 0
        split_fpath = os.path.join("splits/" + split + "/test_iters_estdepth_step%d"%(interval))
        split_fpath = os.path.join("splits/" + split + f"/test/test_estdepth_simple{interval}-nmeas{num_frames-1}")
        if not os.path.exists(split_fpath):
            os.makedirs(split_fpath)
        
        for idx, i in enumerate(range(0, len(saved_txt_list), step_len)):
            tmp_list = saved_txt_list[i: i+step_len]
            tmp_sum += len(tmp_list)
            fpath = os.path.join(split_fpath, "{}_files_iter_{:02d}.txt")
            write_to_file(fpath.format(mode, idx), tmp_list)
        # check the last iteration
        if (i+1)*step_len < len(saved_txt_list):
            tmp_list = saved_txt_list[(i+1)*step_len:]
            tmp_sum += len(tmp_list)
            fpath = os.path.join(split_fpath, "{}_files_iter_{:02d}.txt")
            write_to_file(fpath.format(mode, idx+1), tmp_list)
        assert tmp_sum == len(saved_txt_list), "Cannot finish all the frames, finished {}, but should do {}".format(tmp_sum, len(saved_txt_list))
        print ("saved {} small txts, with total #frames = {}".format(idx+1, tmp_sum))
        print ("omiting #frames={}".format(total_omit_num)) 