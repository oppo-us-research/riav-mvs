"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import shutil
from functools import partial
from multiprocessing.pool import Pool

import cv2
import numpy as np
from path import Path
from scipy.spatial.transform import Rotation


"""
Dataset File Formats:
> See: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats;

"""
def get_closest_index(target_timestamp, other_timestamps):
    differences = np.abs(other_timestamps - target_timestamp)
    return np.argmin(differences)


def process_scene(input_directory, output_folder):
    # used in DeepVideoMVS;
    #K = np.array([[525.0, 0.0, 320.0],
    #              [0.0, 525.0, 240.0],
    #              [0.0, 0.0, 1.0]])
    
    # Updated according to the tum-rgbd document;
    # > See: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    # > CALIBRATION OF THE COLOR CAMERA
    # > "Note: We recommend to use the ROS default parameter set 
    #   (i.e., without undistortion), as undistortion of the pre-registered 
    #   depth images is not trivial."
    K = np.array([[525.0, 0.0, 319.5],
                  [0.0, 525.0, 239.5],
                  [0.0, 0.0, 1.0]])

    print("processing", input_directory)

    image_filenames = sorted((input_directory / "rgb").files("*.png"))
    image_timestamps = np.loadtxt(input_directory / "rgb.txt", usecols=0)

    depth_filenames = sorted((input_directory / "depth").files("*.png"))
    depth_timestamps = np.loadtxt(input_directory / "depth.txt", usecols=0)
    
    # Ground-truth trajectories: The format of each line is 'timestamp tx ty tz qx qy qz qw'
    # - timestamp (float) gives the number of seconds since the Unix epoch.
    # - tx ty tz (3 floats) give the position of the optical center 
    #   of the color camera with respect to the world origin as 
    #   defined by the motion capture system.
    # - qx qy qz qw (4 floats) give the orientation of the optical 
    #   center of the color camera in form of a unit quaternion 
    #   with respect to the world origin as defined by the motion capture system.
    # - The file may contain comments that have to start with "#".

    poses_with_quat = np.loadtxt(input_directory / "groundtruth.txt")
    pose_timestamps = poses_with_quat[:, 0]
    pose_locations = poses_with_quat[:, 1:4]
    pose_quaternions = poses_with_quat[:, 4:]

    sequence = input_directory.split("/")[-1]
    current_output_dir = output_folder / sequence
    if os.path.isdir(current_output_dir):
        if os.path.exists("{}/poses.txt".format(current_output_dir)) and os.path.exists("{}/K.txt".format(current_output_dir)):
            return sequence
        else:
            shutil.rmtree(current_output_dir)
    os.mkdir(current_output_dir)
    os.mkdir(os.path.join(current_output_dir, "images"))
    os.mkdir(os.path.join(current_output_dir, "depth"))

    output_poses = []
    for i in range(len(depth_filenames)):
        depth_timestamp = depth_timestamps[i]

        pose_index = get_closest_index(depth_timestamp, pose_timestamps)
        image_index = get_closest_index(depth_timestamp, image_timestamps)

        depth_filename = depth_filenames[i]
        image_filename = image_filenames[image_index]
        pose_location = pose_locations[pose_index]
        pose_quaternion = pose_quaternions[pose_index]
        rot = Rotation.from_quat(pose_quaternion).as_matrix()
        pose = np.eye(4)
        pose[0:3, 0:3] = rot
        pose[0:3, 3] = pose_location

        image = cv2.imread(image_filename, -1)
        depth = (cv2.imread(depth_filename, -1).astype(float) / 5).astype(np.uint16)
        output_poses.append(pose.ravel().tolist())

        cv2.imwrite("{}/images/{}.png".format(current_output_dir, str(i).zfill(6)), 
                            image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite("{}/depth/{}.png".format(current_output_dir, str(i).zfill(6)), 
                            depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    output_poses = np.array(output_poses)
    np.savetxt("{}/poses.txt".format(current_output_dir), output_poses)
    np.savetxt("{}/K.txt".format(current_output_dir), K)

    return sequence


def main():
    input_folder = Path("/nfs/STG/SemanticDenseMapping/panji/TUM-RGBD")
    output_folder = Path("/nfs/STG/SemanticDenseMapping/changjiang/data/tum_rgbd/exported_test")

    input_directories = [
        input_folder / "rgbd_dataset_freiburg1_desk",
        input_folder / "rgbd_dataset_freiburg1_plant",
        input_folder / "rgbd_dataset_freiburg1_room",
        input_folder / "rgbd_dataset_freiburg1_teddy",
        input_folder / "rgbd_dataset_freiburg2_desk",
        input_folder / "rgbd_dataset_freiburg2_dishes",
        input_folder / "rgbd_dataset_freiburg2_large_no_loop",
        input_folder / "rgbd_dataset_freiburg3_cabinet",
        input_folder / "rgbd_dataset_freiburg3_long_office_household",
        input_folder / "rgbd_dataset_freiburg3_nostructure_notexture_far",
        input_folder / "rgbd_dataset_freiburg3_nostructure_texture_far",
        input_folder / "rgbd_dataset_freiburg3_structure_notexture_far",
        input_folder / "rgbd_dataset_freiburg3_structure_texture_far"]

    pool = Pool(6)
    for finished_scene in pool.imap_unordered(partial(process_scene, output_folder=output_folder), input_directories):
        print("finished", finished_scene)


if __name__ == '__main__':
    main()
