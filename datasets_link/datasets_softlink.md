# Datasets Soft Link

## Scannet
### Scannet Training NPZ Data 
Original ScanNet train data is pre-processed and exported as in `*.npz` data.

dataset_type includes 'scannet_mea1_npz', 'scannet_mea2_npz', 'scannet_mea1_npz_sml' etc.
- src: "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/other_exported_train/deepvideomvs/"
- dst: "datasets_link/Scannet/scans_train_npz/"

### Scannet Validation Data 
We use original ScanNet data (validation split), no pre-processing.

dataset_type includes 'scannet_n3_eval_sml', 'scannet_n3_eval', 'scannet_n2_eval' etc.
- src: ""/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_val"
- dst: "datasets_link/Scannet/scans_val/"

### Scannet Test Data 
We use original ScanNet data (test split), no pre-processing.

dataset_type includes 'scannet_n3_eval_sml', 'scannet_n3_eval', 'scannet_n2_eval' etc.
- src: "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_test/"
- dst: "datasets_link/Scannet/scans_test/"

---

## 7Scenes
### dataset_type == '7scenes_n3_eval' or '7scenes_n5_eval':
- src: "/nfs/STG/SemanticDenseMapping/panji/seven-scenes"
- dst: "datasets_link/seven-scenes"

## TUM-RGBD
### dataset_type == 'tumrgbd_n3_eval' or 'tumrgbd_n5_eval':
- src: "/nfs/STG/SemanticDenseMapping/changjiang/data/tum_rgbd/exported_test"
- dst: "datasets_link/tum_rgbd/exported_test"

## RGBD-Scenes V2
### dataset_type == 'rgbdscenesv2_n3_eval' or 'rgbdscenesv2_n5_eval':
- src: "/nfs/STG/SemanticDenseMapping/changjiang/data/rgbd-scenes-v2/exported_test"
- dst: "datasets_link/rgbd-scenes-v2/exported_test"

## DTU

### dataset_type == 'dtu_yao':
- src: "/nfs/STG/SemanticDenseMapping/changjiang/data/DTU/dtu_patchmatchnet"
- dst: "datasets_link/DTU/dtu_patchmatchnet"
                
### dataset_type == 'dtu_yao_eval':
- src: "/nfs/STG/SemanticDenseMapping/changjiang/data/DTU/dtu_patchmatchnet_test"
- dst: "datasets_link/DTU/dtu_patchmatchnet_test/" 


----

## ETH3D (Not Used Yet)

### dataset_type == 'eth3d':
- src: "/nfs/STG/SemanticDenseMapping/changjiang/data/eth3d_multi_view/eth3d_undistorted_ours/"
- dst: 'datasets_link/eth3d_multi_view/eth3d_undistorted_ours/' 

### dataset_type == 'eth3d_yao' or 'eth3d_yao_eval':
- src: "/nfs/STG/SemanticDenseMapping/changjiang/data/eth3d_multi_view/eth3d_mvsnet_yao/"
- dst: 'datasets_link/eth3d_multi_view/eth3d_mvsnet_yao/' 


---

## KITTI (Not Used Yet)
### dataset_type == 'kitti':
- src: "/nfs/STG/SemanticDenseMapping/panji/kitti_raw/"
- dst: 'datasets_link/kitti_raw/' 

### dataset_type == 'vkt2':
- src: "/mnt/Data/changjiang/Virtual-KITTI-V2/"
- dst: "datasets_link/Virtual-KITTI-V2"
