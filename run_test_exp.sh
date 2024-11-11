#!/bin/bash

#t=18000
#t=1
#echo "Hi, I'm sleeping for $t seconds..."
#sleep ${t}s

#-------
#NOTE:
# 1) run: chmod +x this_file_name
# 2) run: ./this_file_name
#-------

#---------------
# utility function
#---------------
function makeDir () {
    dstDir="$1"
    if [ ! -d $dstDir  ]; then
        mkdir -p $dstDir
        echo "mkdir $dstDir"
    else
        echo "$dstDir exists"
    fi
}

PROJECT_DIR=${HOME}/code/proj-riav-mvs

NUM_EPOCHS=5
NUM_EPOCHS=20
CKPT_DIR="checkpoints_nfs/saved/released" 


###-----------change those parameters --------#
MODEL_NAME=${1:-'OUR_RIAV_MVS_FULL'}
DATASET=${2:-'scannet_n3_eval'}
GPU_ID=${3:-'0'}

## ------- Our models -------- ##
#MODEL_NAME='OUR_RIAV_MVS_FULL'
#MODEL_NAME='OUR_RIAV_MVS_POSE'
#MODEL_NAME='OUR_RIAV_MVS_BASE'


# --- Baselines ----- #
#MODEL_NAME='BASELINE_PAIRNET'
#MODEL_NAME='BASELINE_ESTD'
#MODEL_NAME='BASELINE_MVSNET'
#MODEL_NAME='BASELINE_ITERMVS'

DEFAULT_YAML_FILE="./config/default.yaml"
EXTRA_YAML_FILE=""
EXP_NAME="exp-release-tmp"

#----------------------- our model (full) -------#
if [ $MODEL_NAME == 'OUR_RIAV_MVS_FULL' ]; then
    EXTRA_YAML_FILE="config/riavmvs_full_test.yaml"
    LOAD_WEIGHTS_PATH="$CKPT_DIR/riavmvs_full_epoch_007.pth.tar"

#----------------------- our model (base) -------#
elif [ $MODEL_NAME == 'OUR_RIAV_MVS_BASE' ]; then
    EXTRA_YAML_FILE="config/riavmvs_base_test.yaml"
    LOAD_WEIGHTS_PATH="$CKPT_DIR/riavmvs_base_epoch_002.pth.tar"

#----------------------- our model (base + pose) -------#
elif [ $MODEL_NAME == 'OUR_RIAV_MVS_POSE' ]; then
    EXTRA_YAML_FILE="config/riavmvs_pose_test.yaml"
    LOAD_WEIGHTS_PATH="$CKPT_DIR/riavmvs_pose_epoch_003.pth.tar"

#----------------------- baseline pairnet -------#
elif [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
    EXTRA_YAML_FILE="config/bl_pairnet/bl_pairnet_test.yaml"
    BATCH_SIZE=8
    LOAD_WEIGHTS_PATH="$CKPT_DIR/bl_pairnet_epoch_002.pth.tar"

#----------------------- baseline iter-mvs -------#
elif [ $MODEL_NAME == 'BASELINE_ITERMVS' ]; then
    EXTRA_YAML_FILE="config/bl_itermvs/bl_itermvs_test.yaml"
    LOAD_WEIGHTS_PATH="$CKPT_DIR/bl_pairnet_epoch_002.pth.tar"

#----------------------- baseline mvsnet -------#
elif [ $MODEL_NAME == 'BASELINE_MVSNET' ]; then
    EXTRA_YAML_FILE="config/bl_mvsnet/bl_mvsnet_test.yaml"
    LOAD_WEIGHTS_PATH="$CKPT_DIR/bl_mvsnet_epoch_003.pth.tar"

else
    echo "Wrong MODEL_NAME! $MODEL_NAME" 
    exit

fi


NUM_WORKERS=8
MACHINE_NAME="$HOSTNAME"
echo "[**] MACHINE_NAME=$MACHINE_NAME, NUM_WORKERS=$NUM_WORKERS"

M_2_MM_SCALE_LOSS=1.0

##------Possible Datasets ---------------##
#DATASET='scannet_n3_eval' # test
#DATASET='scannet_n3_eval_sml' # test

#DATASET='scannet' # validation
#DATASET='scannet_n2_eval' # test

#DATASET='dtu_yao'
#DATASET='dtu_yao_eval'

#DATASET='7scenes_n3_eval' # test
#DATASET='7scenes_n5_eval' # test
#DATASET='tumrgbd_n3_eval' # test
#DATASET='tumrgbd_n5_eval' # test
#DATASET='rgbdscenesv2_n3_eval' # test
#DATASET='rgbdscenesv2_n5_eval' # test

if [ "$DATASET" = 'scannet_n3_eval' ] || [ "$DATASET" = 'scannet_n3_eval_sml' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    WIDTH=320
    
elif [ "$DATASET" = 'dtu_yao_eval' ]; then
    if [ $MODEL_NAME == 'OUR_RIAV_MVS_FULL' ]; then
        LOAD_WEIGHTS_PATH="$CKPT_DIR/riavmvs_full_dtu_epoch_03.pth.tar"
    elif [ $MODEL_NAME == 'OUR_RIAV_MVS_BASE' ]; then
        EXTRA_YAML_FILE="config/riavmvs_base_dtu_test.yaml"
        LOAD_WEIGHTS_PATH="$CKPT_DIR/riavmvs_base_dtu_epoch_04.pth.tar"
    elif [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
        LOAD_WEIGHTS_PATH="$CKPT_DIR/bl_pairnet_dtu_epoch_005.pth.tar"
    elif [ $MODEL_NAME == 'BASELINE_ITERMVS' ]; then
        LOAD_WEIGHTS_PATH="$CKPT_DIR/bl_itermvs_dtu_epoch_001.pth.tar"
    elif [ $MODEL_NAME == 'BASELINE_MVSNET' ]; then
        LOAD_WEIGHTS_PATH="$CKPT_DIR/bl_mvsnet_dtu_epoch_006.pth.tar"
    fi
    
    
    M_2_MM_SCALE_LOSS=1.0
    D_MIN=0.425 # meter
    D_MAX=0.935 # meter
     
    HEIGHT=512
    WIDTH=640
    if [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
        HEIGHT=640
        WIDTH=800
    fi


elif [ "$DATASET" = '7scenes_n3_eval' ] || [ "$DATASET" = '7scenes_n3_eval_sml' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    WIDTH=320
    #HEIGHT=480
    #WIDTH=640
     
    if [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
        HEIGHT=256
        WIDTH=320
    fi

elif [ "$DATASET" = '7scenes_n5_eval' ] || [ "$DATASET" = '7scenes_n5_eval_sml' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    WIDTH=320
    if [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
        HEIGHT=256
        WIDTH=320
    fi


elif [ "$DATASET" = 'tumrgbd_n3_eval' ] || [ "$DATASET" = 'tumrgbd_n3_eval_sml' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    WIDTH=320
    #HEIGHT=480
    #WIDTH=640
    if [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
        HEIGHT=256
        WIDTH=320
    fi

elif [ "$DATASET" = 'tumrgbd_n5_eval' ] || [ "$DATASET" = 'tumrgbd_n5_eval_sml' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    WIDTH=320
    if [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
        HEIGHT=256
        WIDTH=320
    fi

elif [ "$DATASET" = 'rgbdscenesv2_n3_eval' ] || [ "$DATASET" = 'rgbdscenesv2_n3_eval_sml' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    WIDTH=320
    #HEIGHT=480
    #WIDTH=640
    SCANNET_EVAL_SAMPLING_TYPE='e-s10n3'
    if [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
        HEIGHT=256
        WIDTH=320
    fi

elif [ "$DATASET" = 'rgbdscenesv2_n5_eval' ] || [ "$DATASET" = 'rgbdscenesv2_n5_eval_sml' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    WIDTH=320
    SCANNET_EVAL_SAMPLING_TYPE='e-s10n5'
    if [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
        HEIGHT=256
        WIDTH=320
    fi


elif [ "$DATASET" = 'scannet_n2_eval' ] || [ "$DATASET" = 'scannet_n2_eval_sml' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    #WIDTH=320
    WIDTH=352
    if [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
        HEIGHT=256
        WIDTH=320
    fi

elif [ "$DATASET" = 'scannet_mea2_npz' ] || [ "$DATASET" = 'scannet_mea2_npz_sml' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    WIDTH=320

elif [ "$DATASET" = 'scannet_mea1_npz' ] || [ "$DATASET" = 'scannet_mea1_npz_sml' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    WIDTH=320
else
    echo "Wrong DATASET TYPE! $DATASET" 
    exit
fi



# ------ run --------- # 
cd ${PROJECT_DIR}

BATCH_SIZE=8
#BATCH_SIZE=16
#BATCH_SIZE=4
#BATCH_SIZE=1
        
if [ $DATASET = 'dtu_yao_eval' ] || [ $DATASET = 'eth3d' ]; then
    BATCH_SIZE=1
    NUM_WORKERS=4
fi

##---- raft iter in test mode ---
RAFT_ITERS=24
#RAFT_ITERS=12
#RAFT_ITERS=8
#RAFT_ITERS=48
#RAFT_ITERS=64
#RAFT_ITERS=8,16

EPO_TEST=0

##------change this part for different experiments ----##
echo "Loading ckpt =" $LOAD_WEIGHTS_PATH
echo "Batch_size=" $BATCH_SIZE
#exit

CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m main \
    --default_yaml_file=${DEFAULT_YAML_FILE} \
    --extra_yaml_file=${EXTRA_YAML_FILE} \
    --dataset=${DATASET} \
    --num_epochs=${NUM_EPOCHS} \
    --batch_size=${BATCH_SIZE} \
    --num_workers=$NUM_WORKERS \
    --height=${HEIGHT} \
    --width=${WIDTH} \
    --load_weights_path=${LOAD_WEIGHTS_PATH} \
    --min_depth=${D_MIN} \
    --max_depth=${D_MAX} \
    --mode='test' \
    --raft_iters=${RAFT_ITERS} \
    --eval_epoch=$EPO_TEST \
    --machine_name=${MACHINE_NAME} \
    --exp_idx=$EXP_NAME \
    --m_2_mm_scale_loss=${M_2_MM_SCALE_LOSS} \
    --eval_gpu_id=${GPU_ID}
