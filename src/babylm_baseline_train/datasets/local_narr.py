import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import re
import numpy as np
import json
import ipdb

from ..env_vars import DATASET_ROOT_DIR


class LocalizedNarrativesBase(Dataset):
    def __init__(
            self, image_processor_func,
            root_dir=os.path.join(
                DATASET_ROOT_DIR, 'LocalNarratives',
                'LocalNarratives'),
            ):
        self.image_processor = image_processor_func()
        self.root_dir = root_dir

        self.anno_dir = os.path.join(root_dir, 'all_anno')
        self.fns = os.listdir(self.anno_dir)
        self.anno_pre_to_img_folder = dict(
                open_images='OpenImages',
                open_images_test='OpenImages_test',
                mscoco='MSCOCO',
                )
        self.anno_pre_to_img_pat = dict(
                open_images='{image_id}.jpg',
                open_images_test='{image_id}.jpg',
                mscoco='{image_id:012}.jpg',
                )

    def __len__(self):
        return len(self.fns)

    def get_img_path_pat(self, curr_idx):
        anno_path = self.fns[curr_idx]
        anno_prefix = '_'.join(anno_path.split('_')[:-1])
        assert anno_prefix in self.anno_pre_to_img_folder
        img_dir = self.anno_pre_to_img_folder[anno_prefix]
        img_path_pat = os.path.join(
                self.root_dir, img_dir, 
                self.anno_pre_to_img_pat[anno_prefix])
        return img_path_pat

    def clean_caption(self, caption):
        caption = caption.replace(u'\xa0', u' ')
        caption = ' '.join(caption.split())
        return caption

    def load_raw_anno(self, idx):
        anno_path = os.path.join(self.anno_dir, self.fns[idx])
        with open(anno_path, 'r') as fin:
            anno = json.load(fin)
        return anno

    def get_img_path(self, anno, idx):
        img_path_pat = self.get_img_path_pat(idx)
        image_id = anno['image_id']
        if 'MSCOCO' in img_path_pat:
            image_id = int(image_id)
        img_path = img_path_pat.format(
                image_id=image_id)
        return img_path

    def load_raw_img_anno(self, idx):
        anno = self.load_raw_anno(idx)
        img_path = self.get_img_path(anno, idx)
        try:
            img = Image.open(img_path)
            img = img.convert(mode='RGB')
        except:
            img = None
        return anno, img

    def __getitem__(self, idx):
        raise NotImplementedError


class LocalizedNarrativesImgTxt(LocalizedNarrativesBase):
    def __init__(
            self,
            with_img=True,
            *args, **kwargs):
        self.with_img = with_img
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        if not self.with_img:
            anno = self.load_raw_anno(idx)
            caption = self.clean_caption(anno['caption'])
            return {'text': caption}
        else:
            anno, img = self.load_raw_img_anno(idx)
            caption = self.clean_caption(anno['caption'])
            if self.image_processor is not None:
                try:
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
                    'text': caption}
