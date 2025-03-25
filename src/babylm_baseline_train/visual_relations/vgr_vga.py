import ipdb
import copy
import argparse
from PIL import Image
import functools
import os
import re
import numpy as np
import pickle
from tqdm import tqdm
import logging
from itertools import chain, product, combinations
import torch
import numpy as np
from torch.nn import CrossEntropyLoss

from babylm_baseline_train.perplexity import eval_img_cap_ppl
from babylm_baseline_train.datasets.babyLM_txt_vis import tk_pad_collate_fn
from babylm_baseline_train.train import tk_funcs
from babylm_baseline_train.env_vars import ROOT_DIR_FREQ, DATASET_ROOT_DIR, DEBUG
from babylm_baseline_train.image_cls.zero_shot_classifier import ImgClsRunner
from babylm_baseline_train.datasets import aro_datasets
RESULT_DIR = os.path.join(
        ROOT_DIR_FREQ,
        'llm_devo_visual_relations_results')


def get_parser():
    parser = argparse.ArgumentParser(
            description='Get zero-shot image classification from models')
    parser.add_argument(
            '--ckpt_path', default=None, type=str, action='store')
    parser.add_argument(
            '--pretrained', default=None, type=str, action='store')
    parser.add_argument(
            '--setting', default=None, type=str, action='store')
    parser.add_argument(
            '--all_ckpts', default=False, action='store_true')
    parser.add_argument(
            '--overwrite', default=False, action='store_true')
    parser.add_argument(
            '--high_level_task', default='vgr',
            type=str, action='store')
    parser.add_argument(
            '--prompt_style', default='None',
            type=str, action='store')
    parser.add_argument(
            '--store_right_flgs', default=False,
            action='store_true')
    return parser


class VisualRelationRunner(ImgClsRunner):
    def __init__(self, args, result_dir=RESULT_DIR):
        super().__init__(args, result_dir)

    def get_task_in_res(self):
        return self.args.high_level_task\
                + f'_pt{self.args.prompt_style}'

    def load_datasets(self):
        datasets = {}
        if self.args.high_level_task == 'vgr':
            datasets['main'] = aro_datasets.VG_Relation(
                    image_preprocess=self.image_processor)
        elif self.args.high_level_task == 'vga':
            datasets['main'] = aro_datasets.VG_Attribution(
                    image_preprocess=self.image_processor)
        else:
            raise NotImplementedError
        self.datasets = datasets

    def construct_trials(self):
        self.all_trials = self.now_dataset

    def test_trials(self):
        accurate_nums = 0
        all_len = len(self.all_trials)
        if self.args.store_right_flgs:
            all_right_flags = []
        #all_len = 100
        for sta_idx in tqdm(
                range(0, all_len, self.batch_size),
                desc='Digest dataset'):
            end_idx = min(all_len, sta_idx + self.batch_size)
            now_data = []
            for _idx in range(sta_idx, end_idx):
                now_text_annos = self.all_trials[_idx]['caption_options']
                now_img = self.all_trials[_idx]['image_options'][0]['pixel_values'][0]
                for _text in now_text_annos:
                    if self.args.prompt_style != 'None':
                        if self.args.prompt_style == 'Context':
                            _text = 'In this image, {}.'.format(_text)
                        else:
                            raise NotImplementedError
                    now_data.append(
                            {'pixel_values': now_img,
                             'text': _text})
            batch_data = self.batch_list_data(now_data)
            for key in batch_data:
                batch_data[key] = batch_data[key].cuda()
            if not self.is_pretrained_image_models():  
                batch_data.pop('pixel_values')
            with torch.no_grad():
                logits = self.lm.model(**batch_data)['logits']
            per_elm_loss = self.get_per_elm_loss(
                    logits, batch_data['input_ids'])
            per_elm_loss = torch.mean(per_elm_loss, dim=-1)
            if DEBUG:
                ipdb.set_trace()
            for _idx in range(sta_idx, end_idx):
                targ_idx = (_idx - sta_idx) * 2 + 1
                if per_elm_loss[targ_idx-1] <= per_elm_loss[targ_idx]:
                    right_flag = 0
                else:
                    right_flag = 1
                accurate_nums += right_flag
                if self.args.store_right_flgs:
                    all_right_flags.append(right_flag)
        accuracy = accurate_nums / all_len
        print(getattr(self, 'curr_ckpt', None), accuracy)
        ret_res = {'accuracy': accuracy}
        if self.args.store_right_flgs:
            ret_res['all_right_flags'] = all_right_flags
        return ret_res


def main():
    parser = get_parser()
    args = parser.parse_args()

    runner = VisualRelationRunner(args)
    runner.eval_all()


if __name__ == '__main__':
    main()
