# Learning to Predict Navigational Patterns from Partial Observations

Code accompanying the paper "Learning to Predict Navigational Patterns from Partial Observations" (RA-L 2023).

The paper presents a self-supervised method to learn navigational patterns in structured environments from partial observations of other agents. The navigational patterns are represented as a directional soft lane probability (DSLP) field. We also present a method for inferring the most likely discrete path or lane graph based on the predicted DSLP field.

Paper link: [Predictive World Models from Real-World Partial Observations](https://arxiv.org/abs/2304.13242)

Video presentation link: TODO

Data directory link: [Google Drive directory](https://drive.google.com/drive/folders/1ylLDDdaxGEOZOJ9b6YXumRtXbOGTncVi?usp=sharing)

![Overview image](https://github.com/robin-karlsson0/dslp/assets/34254153/e9daf5be-05fa-4736-8d3e-a4cd9d6a5d1b)

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

Download and extract the following compressed directories into the local `data/` directory.

[Data directory](https://drive.google.com/drive/folders/1ylLDDdaxGEOZOJ9b6YXumRtXbOGTncVi?usp=sharing)

```
dslp/
└───data/
    |   bev_nuscenes_256px_v01_boston_seaport_unaug_gt_full_eval_preproc.tar.gz
    |   bev_nuscenes_256px_v01_boston_seaport_unaug_gt_eval_preproc.tar.gz
```

Evaluate on partial observations: [bev_nuscenes_256px_v01_boston_seaport_unaug_gt_eval_preproc.tar.gz](https://drive.google.com/file/d/16M4y5Hu9-c5jXMi9anViCOfzudSHpIgE/view?usp=drive_link)

Evaluate on full observations: [bev_nuscenes_256px_v01_boston_seaport_unaug_gt_full_eval_preproc.tar.gz](https://drive.google.com/file/d/1g_wysAgmMryLTq4svXg8hzmg-BlcXs0r/view?usp=sharing)


## Training data

Download and extract the following compressed directories into the local `data/` directory.

_Note: The training datasets are 33 and 35 GB in size._

[Data directory](https://drive.google.com/drive/folders/1ylLDDdaxGEOZOJ9b6YXumRtXbOGTncVi?usp=sharing)

```
dslp/
└───data/
    |   bev_nuscenes_256px_v01_boston_seaport_gt_full_preproc_train
    |   bev_nuscenes_256px_v01_boston_seaport_gt_preproc_train
    |   bev_nuscenes_256px_viz.tar.gz
```

Train on partial observations:
[bev_nuscenes_256px_v01_boston_seaport_gt_preproc_train.tar.gz](https://drive.google.com/file/d/1p4zpkLiSJxDACB9EQKboGQr89dBIBA-g/view?usp=drive_link)

Train on full observations:
[bev_nuscenes_256px_v01_boston_seaport_gt_full_preproc_train.tar.gz]()

Static set of visualization samples used to monitor progress (required for running the code!):
[bev_nuscenes_256px_viz.tar.gz](https://drive.google.com/file/d/1JMIQ48yr5tSGxgYRCMGXsEhl8N05-ieJ/view?usp=drive_link)


## Checkpoint files

Download checkpoint files into the local `checkpoints/` directory.

[Data directory](https://drive.google.com/drive/folders/1ylLDDdaxGEOZOJ9b6YXumRtXbOGTncVi?usp=sharing)

```
dslp/
└───checkpoints/
    |   ...
```

| Experiment                | NLL | IoU |
|---------------------------|------------|--|
| [exp_04_dslp_alpha_ib.ckpt](https://drive.google.com/file/d/1oy1RlmDJojKdJg8-LDUWbpUeaOJWoQqk/view?usp=drive_link) | **12.325**   | 0.442 |
| [exp_08_dslp_region_1.ckpt](https://drive.google.com/file/d/1Z00VNKtLvj1-GBQa8peNAD8WmzKEl1Th/view?usp=drive_link) | 13.174   | 0.423 |
| [exp_09_dslp_region_1_2.ckpt](https://drive.google.com/file/d/1pry0prA-QKcOBbHtoA1p1O6HYOreWQWM/view?usp=drive_link) | 12.557   | **0.444** |

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

# Experimental results

All quantitative results incl. visualizations for experiments are uploaded.

[Data directory](https://drive.google.com/drive/folders/1ylLDDdaxGEOZOJ9b6YXumRtXbOGTncVi?usp=sharing)

```
dslp/
└───results/
    └───exp_01_dsla/
    |   |   eval.txt <--Evaluation log
    |   |   results.txt <-- Evaluation summary
    |   |   viz_000.png <-- Output visualizations
    |   |   ...
    |
    └───exp_02_dslp_const_alpha/
    └───exp_03_dslp_mean_alpha_ib/
    └───exp_04_dslp_alpha_ib/ <-- Main result
    └───exp_05_dslp_full_obs/
    └───exp_06_dslp_no_world_model/
    └───exp_07_dslp_no_aug/
    └───exp_08_dslp_region_1/
    └───exp_09_dslp_region_1_2/
```