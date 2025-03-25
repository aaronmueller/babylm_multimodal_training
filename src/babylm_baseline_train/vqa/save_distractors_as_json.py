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
import babylm_baseline_train.vqa.zero_shot_cls as vqa


class SaveDistJson(vqa.VisualQARunner):
    def run(self):
        self.now_dataset = self.datasets['main']
        self.construct_trials()
        self.add_more_info_to_trials()
        self.save_trials()

    def add_more_info_to_trials(self):
        new_trials = []
        all_image_ids = self.now_dataset['image_id']
        all_question_ids = self.now_dataset['question_id']

        for _trial in self.all_trials:
            valid_idx = _trial[0]
            target_ans = _trial[1]
            distractors = _trial[2]
            now_question = _trial[3]

            now_trial = {
                    'idx_in_hf_dataset': valid_idx,
                    'target_ans': target_ans,
                    'distractors': distractors,
                    'question': now_question,
                    'image_id': all_image_ids[valid_idx],
                    'question_id': all_question_ids[valid_idx],
                    }
            new_trials.append(now_trial)
        self.new_trials = new_trials

    def save_trials(self):
        save_path = os.path.join(
                '/nese/mit/group/evlab/u/chengxuz/babyLM_related/eval_related',
                'vqav2_distractors_info.json')
        with open(save_path, 'w') as fout:
            json.dump(self.new_trials, fout, indent=4)

    def load_datasets(self):
        datasets = {}
        if self.args.high_level_task == 'vqa_v2':
            datasets['main'] = load_dataset(
                    "HuggingFaceM4/VQAv2",
                    split='validation')
        else:
            raise NotImplementedError
        self.datasets = datasets
        self.get_dataset_meta(subset_idxs=None)


def main():
    parser = vqa.get_parser()
    args = parser.parse_args()

    runner = SaveDistJson(args)
    runner.run()


if __name__ == '__main__':
    main()
