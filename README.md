# Parameter-Efficient Image-to-Video Adaptation for Action Spotting of Illegal Waste Dumping

Code for the WasteVision 2027 workshop paper of the same title (IWDD 2027 challenge). It contains the two models of the
paper:

- the Per-Step Spotter (`tools/iwdd/train_loc.py`): an image ViT, frozen, adapted or fully fine-tuned, scores every
  step of a whole video with a temporal head; it also trains on stored frozen per-frame features;
- the Window Classifier (`tools/train.py`): a ViT-B/16 with a temporal head classifies 3.2 s windows.

Backbones: ViT-B/16 pretrained on ImageNet-21k (`pevit_base_patch16_224.orig_in21k`), CLIP LAION-2B
(`pevit_base_patch16_clip_224.laion2b`), DeiT-3 (`pedeit3_base_patch16_224.fb_in22k_ft_in1k`) and MAE
(`pevit_base_patch16_224.mae`). Methods: frozen backbone, full fine-tuning, AdaptFormer, ST-Adapter, AIM and
Frame2Freq.


## Install

```bash
conda create -n pei2v python=3.10
conda activate pei2v
pip install -r requirements.txt
```

`requirements.txt` pins torch 2.5.1+cu121, timm 0.9.12 and the rest. Run every script from the repo root. Data
preparation also needs `ffmpeg` on `PATH`.


## Data

Download the IWDD 2027 training and public test sets from the challenge organizers into `$D`, as
`$D/train/{videos,labels}` and `$D/public_test/{videos,labels}`. Then extract frames (5 fps, short side 256) and write
the video and window CSVs with 5 stratified folds:

```bash
D=/path/to/iwdd
python tools/iwdd/prepare_iwdd.py $D
```

`data/iwdd_scene_folds.csv` holds the scene-grouped folds of the paper's Per-Step Spotter runs, one row per video:
`video_id, split, scene, fold, dumping, test_scene_in_train`. Videos whose opening frames match share a scene, and a
scene's videos share a fold (0-4); the 100 public test videos have fold -1. Pass it to `train_loc` as
`--loc_folds data/iwdd_scene_folds.csv` (the path is relative to the working directory, here the repo root). Without
`--loc_folds`, `train_loc` uses the stratified `fold` column of `train_val.csv`.


## Per-Step Spotter, end to end

One grid cell: CLIP backbone with AdaptFormer, fold 0 (`--loc_fold -1` trains on all training videos and scores the
public test set). The other methods replace the `--cfg ... --adapter_dim 32` flags with:

| Method | Flags |
|---|---|
| Frozen | `--cfg configs/iwdd_ft.yaml --freeze_backbone` |
| Full fine-tuning | `--cfg configs/iwdd_ft.yaml` |
| AdaptFormer | `--cfg configs/iwdd_ft.yaml configs/methods/adapter_adaptformer.yaml --adapter_dim 32` |
| ST-Adapter | `--cfg configs/iwdd_ft.yaml --freeze_backbone --adapter st_adapter --adapter_dim 384` |
| AIM | `--cfg configs/iwdd_ft.yaml --freeze_backbone --adapter aim --adapter_dim 192 --adapter_scale 0.5 --sd 0.2` |
| Frame2Freq | `--cfg configs/iwdd_ft.yaml --freeze_backbone --adapter frame2freq_ms --adapter_dim 192` |

Only the CLIP backbone takes `--custom_mean_std clip`.

```bash
python -u -m tools.iwdd.train_loc --serial 100101 --entity <wandb_entity> --project_name <wandb_project> \
  --dataset_root_path $D --loc_folds data/iwdd_scene_folds.csv --loc_fps 2 --loc_head conv --loc_depth 2 \
  --image_size 224 --n_frames 32 --weight_decay 0.05 --epochs 15 --lr 1e-4 --loc_backbone_lr 1e-5 --loc_accum 8 \
  --cpu_workers 16 --seed 0 --cfg configs/iwdd_ft.yaml configs/methods/adapter_adaptformer.yaml --adapter_dim 32 \
  --model_name pevit_base_patch16_clip_224.laion2b --custom_mean_std clip --loc_fold 0 --loc_out results_loc/fold0.csv
```

The output CSV holds one score per 0.5 s step: `split, seed, video_id, t, score`. Add `--offline` to run without
wandb.


## Per-Step Spotter on stored features

Extract frozen per-frame features (class token and mean patch token) at 2 fps, then train the head on them:

```bash
python -u tools/iwdd/extract_frame_features.py $D vit_base_patch16_clip_224.laion2b feats/vit_base_patch16_clip_224.laion2b_2fps.npz 2
python -u -m tools.iwdd.train_loc --serial 100227 --cfg configs/iwdd_ft.yaml --model_name vit_b16 --freeze_backbone \
  --dataset_root_path $D --loc_fps 2 --batch_size 16 --epochs 60 --lr 1e-3 --weight_decay 0.05 \
  --loc_features feats/vit_base_patch16_clip_224.laion2b_2fps.npz --loc_head conv --loc_depth 2 \
  --loc_folds data/iwdd_scene_folds.csv --seed 0 --loc_fold 0 --loc_out results_loc/feat_fold0.csv
```

`--loc_head transformer --loc_depth 1` gives the transformer head.


## Window Classifier

Fold 0 (`--train_trainval` holds out `win_val_f0.csv`); without `--train_trainval --df_train --df_val` the run trains on
`win_train_val.csv` and predicts `win_test.csv`:

```bash
python -m tools.train --serial 100011 --entity <wandb_entity> --project_name <wandb_project> --model_name vit_b16 \
  --image_size 112 --n_frames 16 --batch_size 32 --cpu_workers 20 --warmup_steps 0 --warmup_epochs 2 --epochs 20 \
  --eval_freq 1 --cfg configs/iwdd_ft.yaml --freeze_backbone --adapter adapter --adapter_dim 32 --video_model temporal \
  --base_lr 0.1 --train_trainval --df_train win_train_f0.csv --df_val win_val_f0.csv --results_dir results_train
```

Full fine-tuning drops `--freeze_backbone --adapter adapter --adapter_dim 32` and uses `--base_lr 0.01`. Each run writes
its window predictions to `results_train/<run name>/ind_preds.csv`.


## Acknowledgements

The code is built on FGIR-ViT by the PCS Lab, National Yang Ming Chiao Tung University, and on
[timm](https://github.com/huggingface/pytorch-image-models/). Experiments were tracked with
[Weights and Biases](https://wandb.ai/).


## Citation

```bibtex
@inproceedings{rios2027pei2v,
  title={Parameter-Efficient Image-to-Video Adaptation for Action Spotting of Illegal Waste Dumping},
  author={Rios, Edwin Arkel and Hu, Min-Chun and Lai, Bo-Cheng},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
  year={2027}
}

@inproceedings{greco2027evolution,
  title={The Evolution of Illegal Waste Dumping Detection in the Age of Foundation Models},
  author={Greco, Antonio and Ricciardi, Andrea Vincenzo and Sansone, Carlo and Vento, Bruno},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
  year={2027}
}
```
