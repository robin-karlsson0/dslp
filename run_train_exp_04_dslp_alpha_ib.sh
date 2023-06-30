#!/bin/bash

python train.py \
    --accelerator gpu \
    --devices 1 \
    --precision 32 \
    --num_workers 4 \
    --profiler simple \
    --max_epochs 1000 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --enc_str 1x16,1x16,1x32,1x32,1x64,1x64,1x128,1x256 \
    --sla_dec_str 1x64,1x64,1x32,1x32,1x16,1x16,1x8,1x8 \
    --da_dec_str 1x64,1x64,1x32,1x32,1x16,1x16,1x8,1x8 \
    --input_ch 5 \
    --out_feat_ch 32 \
    --num_angs 36 \
    --sla_head_layers 1 \
    --da_head_layers 1 \
    --base_channels 32 \
    --dropout_prob 0 \
    --batch_size 16 \
    --train_data_dir ./data/bev_nuscenes_256px_v01_boston_seaport_gt_preproc_train \
    --val_data_dir ./data/bev_nuscenes_256px_v01_boston_seaport_unaug_gt_eval_preproc \
    --test_data_dir ./data/bev_nuscenes_256px_v01_boston_seaport_unaug_gt_eval_preproc \
    --gradient_clip_val 35 \
    --check_val_every_n_epoch 1 \
    --num_sanity_val_steps=0 \
    --viz_dir ./data/bev_nuscenes_256px_viz \
    --do_augmentation \
