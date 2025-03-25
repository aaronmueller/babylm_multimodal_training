import ipdb
import copy
import json
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

from datasets import load_dataset
from babylm_baseline_train.datasets.babyLM_txt_vis import tk_pad_collate_fn
from babylm_baseline_train.env_vars import ROOT_DIR_FREQ, DATASET_ROOT_DIR, DEBUG
from babylm_baseline_train.image_cls.zero_shot_classifier import ImgClsRunner
RESULT_DIR = os.path.join(
        ROOT_DIR_FREQ,
        'llm_devo_vqa_zero_shot_results')


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
            '--high_level_task', default='vqa_v2',
            type=str, action='store')
    parser.add_argument(
            '--prompt_style', default='Base',
            type=str, action='store')
    parser.add_argument(
            '--num_distractors', default=4,
            type=int, action='store')
    parser.add_argument(
            '--source_dist', default='random',
            type=str, action='store')
    return parser


class VisualQARunner(ImgClsRunner):
    def __init__(self, args, result_dir=RESULT_DIR):
        super().__init__(args, result_dir)

    def get_task_in_res(self):
        task_in_res = self.args.high_level_task\
                + f'_nd{self.args.num_distractors}'\
                + f'_pt{self.args.prompt_style}'
        if self.args.source_dist != 'random':
            task_in_res += f'_sd{self.args.source_dist}'
        return task_in_res

    def load_datasets(self):
        datasets = {}
        if self.args.high_level_task == 'vqa_v2':
            datasets['main'] = load_dataset(
                    "HuggingFaceM4/VQAv2",
                    split='validation')
        else:
            raise NotImplementedError
        self.datasets = datasets
        self.get_dataset_meta()

    def get_dataset_meta(self, subset_idxs=7000):
        train_idxs = json.load(open('./mscoco_train_idxs.json', 'r'))
        valid_idxs = []
        all_answers = []
        all_target_ans = []
        all_image_ids = self.datasets['main']['image_id']
        all_raw_answers = self.datasets['main']['answers']
        for idx in range(len(all_image_ids)):
            now_img_id = all_image_ids[idx]
            if str(now_img_id) in train_idxs:
                continue
            valid_idxs.append(idx)
            now_answers = all_raw_answers[idx]
            now_answers = [_ans['answer'] for _ans in now_answers]
            all_target_ans.append(np.unique(now_answers)[0])
            all_answers.extend(now_answers)
        #self.valid_idxs = valid_idxs
        if subset_idxs is not None:
            self.valid_idxs = valid_idxs[:subset_idxs] # save some time
        else:
            self.valid_idxs = valid_idxs # save some time
        self.all_target_ans = all_target_ans
        self.all_answers = np.unique(all_answers)
        self.answer_to_idx_map = {}
        for idx, _ans in enumerate(self.all_answers):
            self.answer_to_idx_map[_ans] = idx

    def init_for_get_distractors(self):
        if self.args.source_dist == 'random':
            self.perm_all_answers = np.random.permutation(
                    self.all_answers)
            self.now_perm_idx = 0
        elif self.args.source_dist.startswith('sim_ans_'):
            if self.args.source_dist.startswith('sim_ans_opt125m_'):
                similar_answer_source = 'facebook_opt_125m'
            elif self.args.source_dist.startswith('sim_ans_opt2d7b_'):
                similar_answer_source = 'facebook_opt_2.7b'
            else:
                raise NotImplementedError
            pkl_path = os.path.join(
                    self.result_dir, 'vqa_v2_na100', 'pretrained',
                    f'{similar_answer_source}.pkl')
            similar_answers = pickle.load(open(pkl_path, 'rb'))
            which_ly = int(self.args.source_dist.split('_')[-1])
            self.similar_answers = similar_answers[which_ly]
        elif self.args.source_dist == 'saved_json':
            with open('vqa_distractors_info.json', 'r') as fin:
                self.saved_distractors = json.load(fin)
        else:
            raise NotImplementedError

    def get_curr_distractors(self, now_answers, target_ans):
        num_dt = self.args.num_distractors
        if self.args.source_dist == 'random':
            distractors = self.perm_all_answers[self.now_perm_idx : (self.now_perm_idx + num_dt)]
            self.now_perm_idx += num_dt
            if self.now_perm_idx + num_dt > len(self.perm_all_answers):
                self.perm_all_answers = np.random.permutation(self.all_answers)
                self.now_perm_idx = 0
            return distractors
        elif self.args.source_dist.startswith('sim_ans_'):
            target_idx = self.answer_to_idx_map[target_ans]
            now_sim_ans_idx = self.similar_answers[target_idx]
            now_sim_ans_idx = now_sim_ans_idx[::-1]
            now_sim_ans = [self.all_answers[_idx] for _idx in now_sim_ans_idx]
            filtered_ans = []
            for _ans in now_sim_ans:
                if _ans not in now_answers:
                    filtered_ans.append(_ans)
                    if len(filtered_ans) == num_dt:
                        break
            return filtered_ans
        elif self.args.source_dist == 'saved_json':
            dist_info = self.saved_distractors[self.now_idx_for_valid_idx]
            return dist_info['distractors']
        else:
            raise NotImplementedError

    def construct_trials(self):
        if len(getattr(self, 'all_trials', [])) > 0:
            return
        self.all_trials = []
        all_dataset_answers = self.now_dataset['answers']
        all_questions = self.now_dataset['question']
        self.init_for_get_distractors()
        for idx_for_valid_idx, valid_idx\
                in enumerate(self.valid_idxs):
            now_answers = [
                    _ans['answer']\
                    for _ans in all_dataset_answers[valid_idx]]
            unique_ans, _cts = np.unique(now_answers, return_counts=True)
            target_ans = unique_ans[np.argmax(_cts)]

            self.now_idx_for_valid_idx = idx_for_valid_idx
            self.now_valid_idx = valid_idx
            distractors = self.get_curr_distractors(
                    now_answers, target_ans)
            now_trial = (valid_idx, target_ans,
                         distractors, all_questions[valid_idx])
            self.all_trials.append(now_trial)

    def add_prompt(self, ans, question):
        if self.args.prompt_style == 'Base':
            return f'Question: {question} Answer: {ans}'
        else:
            raise NotImplementedError

    def expand_trials(self, now_data, now_trial):
        pixel_values = self.image_processor(
                images=self.now_dataset[now_trial[0]]['image'].convert(mode='RGB'),
                return_tensors='pt',
                )['pixel_values'][0]
        now_data.append(
                {'pixel_values': pixel_values,
                 'text': self.add_prompt(
                     now_trial[1], now_trial[3])})
        for text in now_trial[2]:
            now_data.append(
                    {'pixel_values': pixel_values,
                     'text': self.add_prompt(
                         text, now_trial[3])})
        return now_data


def main():
    parser = get_parser()
    args = parser.parse_args()

    runner = VisualQARunner(args)
    runner.eval_all()


if __name__ == '__main__':
    main()
