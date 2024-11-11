"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import time
import numpy as np
import cv2

from torch.utils.data import DataLoader


""" load modules from third_parties/DeepVideoMVS """
from third_parties.DeepVideoMVS.dvmvs.dataset_loader import MVSDataset

""" 
How to run this file:
- cd ~/PROJ_ROOT/
- python -m src.datasets.get_scannet_split
"""
def main():
    import sys
    subsequence_length = 3
    subsequence_length = 2
    
    train_seed = int(round(time.time()))

    data_path = "" # your data path;
    split_txt_path = "" # your split txt file path;

    dataset = MVSDataset(
        root= data_path,
        seed= train_seed,
        #split="TRAINING",
        split = "VALIDATION",
        subsequence_length=subsequence_length,
        scale_rgb=255.0,
        mean_rgb=[0.0, 0.0, 0.0],
        std_rgb=[1.0, 1.0, 1.0],
        geometric_scale_augmentation=False,
        split_txt_path= split_txt_path
        )
    
    print("Number of samples:", len(dataset))
    samples = dataset.samples
    num_samples = len(dataset)

    if 1: 
        txt_fn = "/mnt/Backup/changjiang/manydepth-study/results/dvmvs_val_files_mea{}.txt".format(subsequence_length-1)
        print ("txt_fn = ", txt_fn)
        i0, i1 = 0, num_samples
        with open(txt_fn, 'w') as fw:
            for sample_index in range(i0, i1):
                #print ("[**] sample idx = {}/{}".format(sample_index, num_samples))
                sample = samples[sample_index]
                scene = sample['scene']
                indices = sample['indices']
                subseq_len = len(indices)
                # scene0392_01 1005 -7 10
                l = "{} ".format(scene) + ("{} "*(subseq_len-1)).format(*indices[:-1]) + "{}\n".format(indices[-1])
                #print ("l = ", l)
                fw.write(l)
        sys.exit()
    
    if 0: 
        txt_fn = "/mnt/Backup/changjiang/manydepth-study/results/dvmvs_train_files_mea{}_00.txt".format(subsequence_length-1)
        print ("txt_fn = ", txt_fn)
        i0, i1 = 0, num_samples // 3
        with open(txt_fn, 'w') as fw:
            for sample_index in range(i0, i1):
                #print ("[**] sample idx = {}/{}".format(sample_index, num_samples))
                sample = samples[sample_index]
                scene = sample['scene']
                indices = sample['indices']
                subseq_len = len(indices)
                # scene0392_01 1005 -7 10
                l = "{} ".format(scene) + ("{} "*(subseq_len-1)).format(*indices[:-1]) + "{}\n".format(indices[-1])
                #print ("l = ", l)
                fw.write(l)
        
        txt_fn = "/mnt/Backup/changjiang/manydepth-study/results/dvmvs_train_files_mea{}_01.txt".format(subsequence_length-1)
        print ("txt_fn = ", txt_fn)
        i0, i1 = num_samples // 3, 2*(num_samples//3)
        with open(txt_fn, 'w') as fw:
            for sample_index in range(i0, i1):
                #print ("[**] sample idx = {}/{}".format(sample_index, num_samples))
                sample = samples[sample_index]
                scene = sample['scene']
                indices = sample['indices']
                subseq_len = len(indices)
                # scene0392_01 1005 -7 10
                l = "{} ".format(scene) + ("{} "*(subseq_len-1)).format(*indices[:-1]) + "{}\n".format(indices[-1])
                #print ("l = ", l)
                fw.write(l)
        
        txt_fn = "/mnt/Backup/changjiang/manydepth-study/results/dvmvs_train_files_mea{}_02.txt".format(subsequence_length-1)
        print ("txt_fn = ", txt_fn)
        i0, i1 = 2*(num_samples // 3), num_samples
        with open(txt_fn, 'w') as fw:
            for sample_index in range(i0, i1):
                #print ("[**] sample idx = {}/{}".format(sample_index, num_samples))
                sample = samples[sample_index]
                scene = sample['scene']
                indices = sample['indices']
                subseq_len = len(indices)
                # scene0392_01 1005 -7 10
                l = "{} ".format(scene) + ("{} "*(subseq_len-1)).format(*indices[:-1]) + "{}\n".format(indices[-1])
                #print ("l = ", l)
                fw.write(l)
        sys.exit()

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
    
    if subsequence_length == 3:
        for i, (images, depths, poses, K) in enumerate(loader):
            j = 1
            current_image = images[j]
            current_depth = depths[j].unsqueeze(1)

            previous_image = images[j - 1]
            previous_depth = depths[j - 1].unsqueeze(1)

            print("ref detph max={}, min={}".format(np.max(current_depth.squeeze(1).numpy()[0]), 
                                np.min(current_depth.squeeze(1).numpy()[0])))
            current_image = (np.transpose(current_image.numpy()[0], (1, 2, 0)) * 255).astype(np.uint8)
            current_depth = (current_depth.squeeze(1).numpy()[0] * 5000).astype(np.uint16)
            
            future_image = images[j + 1]
            future_depth = depths[j + 1].unsqueeze(1)
            
            measurement_image = (np.transpose(previous_image.numpy()[0], (1, 2, 0)) * 255).astype(np.uint8)
            measurement_depth = (previous_depth.squeeze(1).numpy()[0] * 5000).astype(np.uint16)

            measurement_image2 = (np.transpose(future_image.numpy()[0], (1, 2, 0)) * 255).astype(np.uint8)
            measurement_depth2 = (future_depth.squeeze(1).numpy()[0] * 5000).astype(np.uint16)
            if 1:
                cv2.imshow("Reference Image", cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
                cv2.imshow("Reference Depth", current_depth)
                
                cv2.imshow("Pre Image", cv2.cvtColor(measurement_image, cv2.COLOR_BGR2RGB))
                cv2.imshow("Pre Depth", measurement_depth)
                
                cv2.imshow("Fut Image", cv2.cvtColor(measurement_image2, cv2.COLOR_BGR2RGB))
                cv2.imshow("Fut Depth", measurement_depth2)

                cv2.waitKey()
    if 0:
        for i, (images, depths, poses, K) in enumerate(loader):
            for j in range(1, len(images)):
                current_image = images[j]
                current_depth = depths[j].unsqueeze(1)

                previous_image = images[j - 1]
                previous_depth = depths[j - 1].unsqueeze(1)

                print(np.max(current_depth.squeeze(1).numpy()[0]))
                print(np.min(current_depth.squeeze(1).numpy()[0]))
                current_image = (np.transpose(current_image.numpy()[0], (1, 2, 0)) * 255).astype(np.uint8)
                current_depth = (current_depth.squeeze(1).numpy()[0] * 5000).astype(np.uint16)
                measurement_image = (np.transpose(previous_image.numpy()[0], (1, 2, 0)) * 255).astype(np.uint8)
                measurement_depth = (previous_depth.squeeze(1).numpy()[0] * 5000).astype(np.uint16)

                if 1:
                    cv2.imshow("Reference Image", cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
                    cv2.imshow("Reference Depth", current_depth)
                    cv2.imshow("Measurement Image", cv2.cvtColor(measurement_image, cv2.COLOR_BGR2RGB))
                    cv2.imshow("Measurement Depth", measurement_depth)

                    cv2.waitKey()


if __name__ == '__main__':
    main()