import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import re
import numpy as np
import json
import ipdb

from ..env_vars import DATASET_ROOT_DIR


class ConceptualCaptions3M(Dataset):
    def __init__(
            self, image_processor_func,
            root_dir=os.path.join(
                DATASET_ROOT_DIR, 
                'Conceptual-3M'),
            with_img=True,
            split='training',
            ):
        self.image_processor = image_processor_func()
        self.root_dir = root_dir
        self.with_img = with_img
        self.split = split

        self.get_valid_idxs()

    def clean_cap(self, cap):
        punc_to_fix = [
                '.', ':', '!',
                ',', '?', ';',
                ]
        for _punc in punc_to_fix:
            cap = cap.replace(f' {_punc}', _punc)
        return cap

    def get_valid_idxs(self):
        if self.split == 'training':
            cap_meta_path = os.path.join(
                    self.root_dir,
                    'Train_GCC-training.tsv')
            valid_download = os.path.join(
                    self.root_dir,
                    'downloaded_training_report.tsv')
        elif self.split == 'validation':
            cap_meta_path = os.path.join(
                    self.root_dir,
                    'Validation_GCC-1.1.0-Validation.tsv')
            valid_download = os.path.join(
                    self.root_dir,
                    'downloaded_validation_report.tsv')
        else:
            raise NotImplementedError
        with open(valid_download, 'r') as fin:
            all_lines = fin.readlines()
        with open(cap_meta_path, 'r') as fin:
            all_caps = fin.readlines()

        valid_fpath_cap = []
        num_not_jpgs = 0
        for cap, line in zip(all_caps, all_lines):
            line = line.strip()
            line = line.split('\t')
            img_path = line[0]
            file_format = line[2]
            if img_path == self.split:
                continue
            if 'jpeg' not in file_format:
                num_not_jpgs += 1
                continue
            cap = cap.strip().split('\t')[0]
            cap = self.clean_cap(cap)
            valid_fpath_cap.append((img_path, cap))
        self.valid_fpath_cap = valid_fpath_cap

    def __len__(self):
        return len(self.valid_fpath_cap)

    def __getitem__(self, idx):
        img_fpath, cap = self.valid_fpath_cap[idx]
        if not self.with_img:
            return {'text': cap}
        img_fpath = os.path.join(self.root_dir, img_fpath)
        if self.image_processor is not None:
            try:
                img = Image.open(img_fpath)
                img = img.convert(mode='RGB')
                img = self.image_processor(
                        images=[img], return_tensors="pt")['pixel_values'][0]
            except:
                img = Image.fromarray(
                        np.ones((3, 256, 256)) * 255,
                        'RGB')
                img = self.image_processor(
                        images=[img], return_tensors="pt")['pixel_values'][0]
        else:
            img = None
        return {'pixel_values': img, 
                'text': cap}
