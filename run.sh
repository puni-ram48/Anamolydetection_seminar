#!/bin/bash
set -e  # Stop script if any command fails

# -------------------------------------------------
# Change to project directory
# Users should replace this with their own path
# -------------------------------------------------
PROJECT_DIR="<PATH_TO_PROJECT>"
cd $PROJECT_DIR

# -------------------------------------------------
# Dataset path
# Users should replace this with their own dataset path
# -------------------------------------------------
DATASET_PATH="<PATH_TO_CHEST_XRAY_DATASET>"

# -------------------------------------------------
# Common parameters (shared across runs)
# -------------------------------------------------
GPU=0
SEED=42
BACKBONE=resnet18
LAYER=layer3
PATCHSIZE=3
GAN_EPOCHS=4
PRE_PROJ=1
NUM_WORKERS=4
RESIZE=256
IMAGESIZE=256
DATASET_NAME=chest_xray

# =================================================
# EXPERIMENT 1 — baseline 
# =================================================
python3 main.py \
  --gpu $GPU \
  --seed $SEED \
  --results_path ./test_1_results \
  --run_name final_gpu0 \
  net \
  -b $BACKBONE \
  -le $LAYER \
  --pretrain_embed_dimension 128 \
  --target_embed_dimension 64 \
  --patchsize $PATCHSIZE \
  --meta_epochs 30 \
  --embedding_size 256 \
  --gan_epochs $GAN_EPOCHS \
  --noise_std 0.015 \
  --dsc_hidden 128 \
  --dsc_layers 2 \
  --dsc_margin .5 \
  --pre_proj $PRE_PROJ \
  dataset \
  --batch_size 32 \
  --resize $RESIZE \
  --imagesize $IMAGESIZE \
  --num_workers $NUM_WORKERS \
  --augment \
  -d NORMAL \
  $DATASET_NAME $DATASET_PATH


# =================================================
# EXPERIMENT 2 — lower noise
# =================================================
python3 main.py \
  --gpu $GPU \
  --seed $SEED \
  --results_path ./test_2_results \
  --run_name final_gpu0 \
  net \
  -b $BACKBONE \
  -le $LAYER \
  --pretrain_embed_dimension 128 \
  --target_embed_dimension 64 \
  --patchsize $PATCHSIZE \
  --meta_epochs 30 \
  --embedding_size 256 \
  --gan_epochs $GAN_EPOCHS \
  --noise_std 0.005 \
  --dsc_hidden 128 \
  --dsc_layers 2 \
  --dsc_margin .5 \
  --pre_proj $PRE_PROJ \
  dataset \
  --batch_size 32 \
  --resize $RESIZE \
  --imagesize $IMAGESIZE \
  --num_workers $NUM_WORKERS \
  --augment \
  -d NORMAL \
  $DATASET_NAME $DATASET_PATH
