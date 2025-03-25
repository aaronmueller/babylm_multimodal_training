# BabyLM 2024 Multimodal Training Pipeline

This repository contains code for training Flamingo and GIT models, following the procedures used to train the 2024 BabyLM multimodal baselines. Note that the code and README in this repository were written almost entirely by Chengxu Zhuang.

## Environment

Python 3.9, transformer package in huggingface, and datasets package in huggingface.

And also install: https://github.com/chengxuz/pt_framework

Install the current repo using `pip install .` or `pip install -e .`.

### Where to put data

First, define the environment variable `BABYLM_ROOT_DIR` to be where your models and data will live.
The downloaded data should be put at `${BABYLM_ROOT_DIR}/datasets/` so that this folder contains the following four subfolders: `babylm_100M`, `babylm_10M`, `babylm_dev`, and `babylm_test`.
The trained models will be put at `${BABYLM_ROOT_DIR}/models/` and the records will be put at `${BABYLM_ROOT_DIR}/model_recs/`.

## Training Command for 2024 Visual-text Baselines

### Folder structure for training from raw images

Specify the environment variable `BABYLM_DATASET_ROOT_DIR` to be where the visual datasets are stored.

### Image data

Under this folder, the CC-3M should be structured like this:
```
ls ${BABYLM_DATASET_ROOT_DIR}/Conceptual-3M:
downloaded_training_report.tsv  downloaded_validation_report.tsv  Train_GCC-training.tsv  training  validation  Validation_GCC-1.1.0-Validation.tsv
```
The training and validation are folders containing the raw images downloaded using this [repo](https://github.com/igorbrigadir/DownloadConceptualCaptions).

The Localized-Narratives should be structured like this under the same folder (note the repetition of the name `LocalNarratives`):
```
ls ${BABYLM_DATASET_ROOT_DIR}/LocalNarratives/LocalNarratives/
all_anno  MSCOCO  OpenImages
```
The `all_anno` folder contains the json files for the annotations, where the filenames are like this:
```
mscoco_00000000.json
open_images_00096010.json
open_images_test_00126010.json
```
The number after the last `_` must be a unique identifier.

The `MSCOCO` folder directly contains `{image_id:012}.jpg`, and the OpenImages folder directly contains `{image_id}.jpg`.

### Text data
The `train_50M.zip` file downloaded from [OSF](https://osf.io/ad7qg/) should be unzipped so that the structure is like this:
```
ls ${BABYLM_DATASET_ROOT_DIR}/babylm_vis_text_2024
bnc_spoken.train  childes.train  gutenberg.train  open_subtitles.train  simple_wiki.train  switchboard.train
```

### GIT and Flamingo models

The Flamingo training:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29123 general_train.py --setting "BabyLM/exp_txt_vis_new.py:base_flmg_1vd25_s1"
```

Changing the `base_flmg_1vd25_s1` to `base_git_1vd25_s1` in this command will start the training of the GIT model.

# Training Command for 2023 Baselines

## OPT-125M
Run the following command under the `scripts` folder.
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29123 general_train.py --setting "BabyLM/exp_strict.py:opt125m_s1"
```

This command will load a training setting specified by function `opt125m_s1` at `src/babylm_baseline_train/configs/BabyLM/exp_strict.py`.

## RoBERTa-Base
Run the following command under the `scripts` folder.
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29123 general_train.py --setting "BabyLM/exp_strict_mask.py:roberta_s1"
```

### SBATCH running for 100M
Run the following command under the `scripts` folder.
```
sbatch --export=SETTING="BabyLM/exp_strict_mask_100M.py:roberta_s1" sb_scripts/train.sh
```

# Where important parameters are defined

Learning rate schedule is defined at function `get_learning_rate_params` in script `basic_param_setter.py` under `src/babylm_baseline_train` folder.

Optimizer is in the `scripts/general_train.py` script inside the `get_key_params` funciton.

# How to load the pretrained models

See the functions in `src/babylm_baseline_train/models/ckpt_loader.py`.

# Questions?

Feel free to open issues here. Or just contact us through Slack/emails.
