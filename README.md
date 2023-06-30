# Learning to Predict Navigational Patterns from Partial Observations

Code accompanying the paper "Learning to Predict Navigational Patterns from Partial Observations" (RA-L 2023)

Paper link: [Predictive World Models from Real-World Partial Observations](https://arxiv.org/abs/2304.13242)

Video presentation link: TODO

Shared public data (incl. pretrained models): [Google Drive directory](https://drive.google.com/drive/folders/1ylLDDdaxGEOZOJ9b6YXumRtXbOGTncVi?usp=sharing)

**TODO: Overview image**


# Installation

The code is tested with Python 3.9 on Ubuntu 22.04.

Download all submodules
```
git submodule update --init --recursive
```

The submodules are used for the following tasks

1. predictive-world-models: Predictive world model repository
2. vdvae: Code for implementing the predictive world model. Fork of the original VDVAE repository modified to a dual encoder posterior matching HVAE model.

## Install dependencies

Follow README instructions in `predictive-world-models/`

Downgrade Pytorch Lightning --> 1.9.0 (for CLI implementation to work)
```
pip uninstall pytorch-lightning
pip install pytorch-lightning==1.9.0
```


## Evaluation data

```
dslp/
└───data/
    |   bev_nuscenes_256px_v01_boston_seaport_unaug_gt_full_eval_preproc.tar.gz
    |   bev_nuscenes_256px_v01_boston_seaport_unaug_gt_eval_preproc.tar.gz
```

Evaluate on partial observations: [bev_nuscenes_256px_v01_boston_seaport_unaug_gt_eval_preproc.tar.gz](https://drive.google.com/file/d/16M4y5Hu9-c5jXMi9anViCOfzudSHpIgE/view?usp=drive_link)

Evaluate on full observations: [bev_nuscenes_256px_v01_boston_seaport_unaug_gt_full_eval_preproc.tar.gz](https://drive.google.com/file/d/1g_wysAgmMryLTq4svXg8hzmg-BlcXs0r/view?usp=sharing)


## Training data
```
dslp/
└───data/
    |   bev_nuscenes_256px_v01_boston_seaport_gt_full_preproc_train
    |   bev_nuscenes_256px_v01_boston_seaport_gt_preproc_train
        bev_nuscenes_256px_viz.tar.gz
```

Train on partial observations:
[bev_nuscenes_256px_v01_boston_seaport_gt_preproc_train.tar.gz](https://drive.google.com/file/d/1p4zpkLiSJxDACB9EQKboGQr89dBIBA-g/view?usp=drive_link)

Train on full observations:
[bev_nuscenes_256px_v01_boston_seaport_gt_full_preproc_train.tar.gz]()

Static set of visualization samples used to monitor progress (required for running the code!):
[bev_nuscenes_256px_viz.tar.gz](https://drive.google.com/file/d/1JMIQ48yr5tSGxgYRCMGXsEhl8N05-ieJ/view?usp=drive_link)


## Checkpoint files

| Experiment                | NLL | IoU |
|---------------------------|------------|--|
| [exp_04_dslp_alpha_ib.ckpt](exp_04_dslp_alpha_ib.ckpt) | 12.325   | 0.442 |


# Evaluation

Run the evaluation script to recompute the main experiment results. The script assumes the datasets and checkpoints are set up as instructed.

```
sh run_eval_exp_04_dslp_alpha_ib.sh
```

# Training

Run the training script to recreate the main experiment DSLP model. The script assumes the datasets and checkpoints are set up as instructed.

```
sh run_train_exp_04_dslp_alpha_ib.sh
```
