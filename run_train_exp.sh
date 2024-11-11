#!/bin/bash

#-------
#NOTE:
# 1) run: chmod +x this_file_name
# 2) run: ./this_file_name
#-------

#t=6500
#t=1
#echo "Hi, I'm sleeping for $t seconds..."
#sleep ${t}s


NUM_EPOCHS=5
#NUM_EPOCHS=20

NUM_WORKERS=8
#NUM_WORKERS=16

#BATCH_SIZE=32 #RTX-A6000, 4GPUs
BATCH_SIZE=4 #RTX-A6000, 1GPU


#PRINT_FREQ=300
PRINT_FREQ=20

LOG_FREQUENCY=20
#LOG_FREQUENCY=100 # For A6000 machine, for scannet_npz

MODEL_NAME=${1:-'RIAV_MVS_FULL'}
DATASET=${2:-'scannet_mea2_npz'}
GPU_IDS=${3:-'0'}

# --- Ours modules; ----- #
# MODEL_NAME='RIAV_MVS'
# MODEL_NAME='RIAV_MVS_FULL'
# MODEL_NAME='RIAV_MVS_BASE'
# MODEL_NAME='RIAV_MVS_POSE'

# --- Baselines ----- #
# MODEL_NAME='BASELINE_PAIRNET'
# MODEL_NAME='BASELINE_MVSNET'
# MODEL_NAME='BASELINE_ITERMVS'

# --- Baseline + Ours modules; ----- #
# MODEL_NAME='BASELINE_ITERMVS_POSE'
# MODEL_NAME='BASELINE_ITERMVS_ATTEN'
# MODEL_NAME='BASELINE_MVSNET_POSE'
# MODEL_NAME='BASELINE_MVSNET_ATTEN'

# --- Dataset for training ----- #
# DATASET='dtu_yao'
# DATASET='scannet_mea1_npz'
# DATASET='scannet_mea2_npz'


DEFAULT_YAML_FILE="./config/default.yaml"
EXTRA_YAML_FILE=""

#----------------------- our model (full) -------#
if [ $MODEL_NAME == 'RIAV_MVS' ] || [ $MODEL_NAME == 'RIAV_MVS_FULL' ]; then
    EXTRA_YAML_FILE="config/riavmvs_full_train.yaml"
#----------------------- our model (base) -------#
elif [ $MODEL_NAME == 'RIAV_MVS_BASE' ]; then
    EXTRA_YAML_FILE="config/riavmvs_base_train.yaml"

#----------------------- our model (base + pose) -------#
elif [ $MODEL_NAME == 'RIAV_MVS_POSE' ]; then
    EXTRA_YAML_FILE="config/riavmvs_pose_train.yaml"

#----------------------- baseline pairnet -------#
elif [ $MODEL_NAME == 'BASELINE_PAIRNET' ]; then
    EXTRA_YAML_FILE="config/bl_pairnet/bl_pairnet_train.yaml"

#----------------------- baseline iter-mvs -------#
elif [ $MODEL_NAME == 'BASELINE_ITERMVS' ]; then
    EXTRA_YAML_FILE="config/bl_itermvs/bl_itermvs_train.yaml"
#----------------------- baseline iter-mvs + pose -------#
elif [ $MODEL_NAME == 'BASELINE_ITERMVS_POSE' ]; then
    EXTRA_YAML_FILE="./config/bl_itermvs/bl_itermvs_pose_train.yaml"
#----------------------- baseline iter-mvs + atten -------#
elif [ $MODEL_NAME == 'BASELINE_ITERMVS_ATTEN' ]; then
    EXTRA_YAML_FILE="./config/bl_itermvs/bl_itermvs_atten_train.yaml"


#----------------------- baseline estdepth -------#
elif [ $MODEL_NAME == 'BASELINE_ESTD' ]; then
    EXTRA_YAML_FILE=""

#----------------------- baseline mvsnet -------#
elif [ $MODEL_NAME == 'BASELINE_MVSNET' ]; then
    NUM_EPOCHS=20
    EXTRA_YAML_FILE="./config/bl_mvsnet/bl_mvsnet_train.yaml"
#----------------------- baseline mvsnet + pose -------#
elif [ $MODEL_NAME == 'BASELINE_MVSNET_POSE' ]; then
    EXTRA_YAML_FILE="./config/bl_mvsnet/bl_mvsnet_pose_warmup_train.yaml"
    #EXTRA_YAML_FILE="./config/bl_mvsnet/bl_mvsnet_pose_train.yaml"
#----------------------- baseline mvsnet + atten -------#
elif [ $MODEL_NAME == 'BASELINE_MVSNET_ATTEN' ]; then
    EXTRA_YAML_FILE="./config/bl_mvsnet/bl_mvsnet_atten_train.yaml"

fi

LEARNING_RATE=1.0e-4 # based on Exp60C, ckpt Epo2;
#LR_SCHEDULER="constant"
LR_SCHEDULER="piecewise_epoch"
LR_EPOCH_STEP="4-8-15" # epochs#5

RAFT_ITERS=8


if [ $MODEL_NAME == 'BASELINE_ITERMVS' ]; then
    LEARNING_RATE=1e-3
    LR_SCHEDULER="piecewise_epoch"
    LR_EPOCH_STEP="4-8-12" # epochs#20
fi

if [ $LR_SCHEDULER == 'constant' ]; then
    let DUMMY_STEP1=$NUM_EPOCHS+2
    let DUMMY_STEP2=$NUM_EPOCHS+5
    LR_EPOCH_STEP="$DUMMY_STEP1-$DUMMY_STEP2" # actually, disable this LR_EPOCH_STEP;
    echo "constant learning rate:, reset LR_EPOCH_STEP to an inactive larger number, ${LR_EPOCH_STEP}"
fi


MACHINE_NAME="rtxA6ks3"
MACHINE_NAME="$HOSTNAME"
echo "[**] MACHINE_NAME=$MACHINE_NAME, NUM_WORKERS=$NUM_WORKERS"


#---------------------
M_2_MM_SCALE_LOSS=1.0



if [ "$DATASET" = 'dtu_yao' ]; then
    #HEIGHT=256
    #WIDTH=512
    HEIGHT=256
    WIDTH=320
    D_MIN=0.425 # meter
    D_MAX=0.935 # meter
    M_2_MM_SCALE_LOSS=1.0 # inverse depth loss, no need to x1000.0

elif [ "$DATASET" = 'scannet_mea1_npz' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    WIDTH=512

elif [ "$DATASET" = 'scannet_mea2_npz' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    #WIDTH=512
    WIDTH=256
    
    if [ $MODEL_NAME == 'BASELINE_ITERMVS' ]; then
        HEIGHT=256
        WIDTH=320
    elif [ $MODEL_NAME == 'BASELINE_ITERMVS_ATTEN' ]; then
        HEIGHT=256
        WIDTH=320
    elif [ $MODEL_NAME == 'BASELINE_ITERMVS_POSE' ]; then
        HEIGHT=256
        WIDTH=320
    elif [ $MODEL_NAME == 'BASELINE_ITERMVS_ATTEN_POSE' ]; then
        HEIGHT=256
        WIDTH=320
    fi

elif [ "$DATASET" = 'scannet_mea4_npz' ]; then
    D_MIN=0.25
    D_MAX=20
    HEIGHT=256
    #WIDTH=512
    WIDTH=256

    if [ $MODEL_NAME == 'BASELINE_ITERMVS' ]; then
        HEIGHT=480
        WIDTH=640
    elif [ $MODEL_NAME == 'BASELINE_ESTD' ]; then
        HEIGHT=128 # due to time limit, to speed up the training;
        WIDTH=160
        D_MIN=0.10
        D_MAX=10
    fi

fi


CKPT_DIR="checkpoints_nfs/" 

EXP_NAME="exp-release-traintmp"
#EXP_NAME="exp01-release-train"
# exp77A-ddp-raftpsQr4f1a4G3pairspf-Dil1inv-scanpz2-Z0.25to20-D64-epo20-LR1e-4-p-epo4-8-15-gam0.5-bs56-h256xw256-task-20220417191800-57718-grtwm

#LOAD_WEIGHTS_PATH="${CKPT_DIR}/exp77A-ddp-raftpsQr4f1a4G3pairspf-Dil1inv-scanpz2-Z0.25to20-D64-epo20-LR1e-4-p-epo4-8-15-gam0.5-bs56-h256xw256-task-20220417191800-57718-grtwm/ckpt_epoch_007.pth.tar"
LOAD_WEIGHTS_PATH=""

RESUME_PATH=""
#RESUME_PATH="${CKPT_DIR}/expcvr3B-bl_mvsnetp-Dll1-dtu-Z0.42to0.935-D64-epo10-LR1e-4-p-epo4-8-gam0.5-bs32-h256xw320-rtxa6ks10/ckpt_epoch_002.pth.tar"

if [ "$RESUME_PATH" = ''  ]; then
    TRAIN_TEST_MODE='train'
else
    TRAIN_TEST_MODE='resume'
fi


echo "[***] model_name=${MODEL_NAME}"
echo "[***] loading yamls=${DEFAULT_YAML_FILE} and ${EXTRA_YAML_FILE}"
echo "[***] loading ckpt=${LOAD_WEIGHTS_PATH}"

#GPU_IDS=$1
# change IFS to comma: the IFS variable is set to a comma, 
# which means that the read command will use the comma 
# as the delimiter when splitting the string into an array;
IFS=','
GPU_NUM=$(echo "$GPU_IDS" | tr "$IFS" '\n' | wc -l)
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$GPU_IDS
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, nproc_per_node=$GPU_NUM"

#exit
## Single-node multi-worker
#python3 -m main \
torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM -m main \
    --default_yaml_file=${DEFAULT_YAML_FILE} \
    --extra_yaml_file=${EXTRA_YAML_FILE} \
    --dataset=${DATASET} \
    --batch_size=${BATCH_SIZE} \
    --num_epochs=${NUM_EPOCHS} \
    --height=${HEIGHT} \
    --width=${WIDTH} \
    --min_depth=${D_MIN} \
    --max_depth=${D_MAX} \
    --load_weights_path=${LOAD_WEIGHTS_PATH} \
    --learning_rate=${LEARNING_RATE} \
    --lr_scheduler=${LR_SCHEDULER} \
    --scheduler_step_size=${LR_EPOCH_STEP} \
    --print_freq=${PRINT_FREQ} \
    --resume_path=${RESUME_PATH} \
    --mode=${TRAIN_TEST_MODE} \
    --num_workers=$NUM_WORKERS \
    --machine_name=${MACHINE_NAME} \
    --exp_idx=${EXP_NAME} \
    --m_2_mm_scale_loss=${M_2_MM_SCALE_LOSS} \
    --raft_iters=${RAFT_ITERS} \
    --log_frequency=${LOG_FREQUENCY}
