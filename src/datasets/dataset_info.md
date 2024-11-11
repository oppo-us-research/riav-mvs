# 1. ScanNet Dataset Exported
We follow the dataset format used in [DeepVideoMVS](https://github.com/ardaduz/deep-video-mvs).

## 1.1 Test Set Format
- Set dir in the [data export file](src/datasets/scannet_export.py) as below:

```python
input_path = "/nfs/SHARE/dataset/scannet_data/scans_test"
## 1) for test and benchmark
output_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_test"
# and flash dir
output_path = "/nfs/flash/STG/project/SemanticDenseMapping/Scannet/scans_test"

## 2) for validation
output_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_val"
# and flash dir
output_path = "/nfs/flash/STG/project/SemanticDenseMapping/Scannet/scans_val"
```

- Then set `is_sanity_check = False`, and run this code via:

```bash
cd src/datasets 
python scannet_export.py
```

- It will take few hours to finish the parsing and saving.
- After that, set `is_sanity_check = True`, and rerun this code for sanity check.
- If everything goes well, you will see the exported test sets.
- See data at 
```
DATA_PATH="/mnt/Backup2/changjiang/data/Scannet/scans_test"
# now the data is moved to new dirs:
DATA_PATH = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_test"
# and flash dir for quick I/O;
output_path = "/nfs/flash/STG/project/SemanticDenseMapping/Scannet/scans_test"
```
The structure is:
```
| /nfs/flash/STG/project/SemanticDenseMapping/Scannet/scans_test/
--- | scene0707_00  
-------| frames  
-----------| depth  
--------------- 000000.png  000088.png  000176.png ...   
-----------| color
--------------- 000000.png  000088.png  000176.png ...   
-----------| intrinsic
--------------- intrinsic_depth.txt
-----------| pose
--------------- 000000.txt 000001.png  000710.png ...

--- | scene0720_00
...
and so on

```

## 1.2 ScanNet Validation Set We used

- See data at 
```
DATA_PATH = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/scans_val"
# and flash dir for quick I/O;
output_path = "/nfs/flash/STG/project/SemanticDenseMapping/Scannet/scans_val"
```
Its structure is the same as the above.

## 1.3 Train Set Format
- Set the [data export file](src/datasets/scannet_export.py).

```python
is_train = True
input_path = "/nfs/SHARE/dataset/scannet_data/scans"
output_path = "/nfs/STG/SemanticDenseMapping/changjiang/data/Scannet/other_exported_train/deepvideomvs"
# and flash dir
output_path = "/nfs/flash/STG/project/SemanticDenseMapping/Scannet/other_exported_train/deepvideomvs"
```

- Run and get the exported train set:
```bash
cd src/datasets
python scannet-export.py
```
See the instruction from [DeepVideoMVS](https://github.com/ardaduz/deep-video-mvs):

> During training, the system expects each scene to be placed in a folder, and color image and depth image for a time step to be packed inside a zipped numpy archive (.npz). See the code here. We use frame_skip=4 while exporting the ScanNet training and validation scenes due to the large amount of data. The training/validation split of unique scenes which are used during this work is also provided here, one may replace the randomly generated ones with these two.

- Save the selected samples to txt files:
  - `train`: saved in the file [train_files.txt](splits/scannetv2/scannet_mea1_npz/train_files.txt), and passed the Sanity check!
  - `val`: saved in the file [val_files.txt](splits/scannetv2/scannet_mea1_npz/val_files.txt), and passed the Sanity check!

