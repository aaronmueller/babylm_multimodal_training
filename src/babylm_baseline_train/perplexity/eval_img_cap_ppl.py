import ipdb
import copy
import argparse
import functools
import os
import re
import numpy as np
import pickle
from tqdm import tqdm
import logging
from itertools import chain, product, combinations
from transformers import AutoTokenizer

import torch
import numpy as np

from babylm_baseline_train.analysis import use_lm_eval
from babylm_baseline_train.datasets import cc_3M
from babylm_baseline_train.datasets.babyLM_txt_vis import tk_pad_collate_fn
from babylm_baseline_train.train import tk_funcs
from babylm_baseline_train.env_vars import ROOT_DIR_FREQ
DEBUG = int(os.environ.get(
        'DEBUG',
        '0')) == 1
RESULT_DIR = os.path.join(
        ROOT_DIR_FREQ,
        'llm_devo_perplexity_results')


def get_parser():
    parser = argparse.ArgumentParser(
            description='Get perplexity from models')
    parser.add_argument(
            '--ckpt_path', default=None, type=str, action='store')
    parser.add_argument(
            '--setting', default=None, type=str, action='store')
    parser.add_argument(
            '--all_ckpts', default=False, action='store_true')
    parser.add_argument(
            '--overwrite', default=False, action='store_true')
    parser.add_argument(
            '--high_level_task', default='cc3m',
            type=str, action='store')
    parser.add_argument(
            '--ctx_len', default=4,
            type=int, action='store')
    return parser


class ImgCapPPLRunner(use_lm_eval.LMEvalRunner):
    def __init__(self, args, result_dir=RESULT_DIR):
        self.args = args
        self.batch_size = 16
        self.result_dir = RESULT_DIR
        self.get_key_params()
        self.get_lm_model()
        self.get_all_ckpts()
        self.load_datasets()
        self.setup_tk_proc()

    def get_task_in_res(self):
        return self.args.high_level_task + f'_ctx{self.args.ctx_len}'

    def load_datasets(self):
        datasets = {}
        if self.args.high_level_task == 'cc3m':
            datasets['main'] = cc_3M.ConceptualCaptions3M(
                    image_processor_func=lambda: None,
                    with_img=True,
                    split='validation')
        else:
            raise NotImplementedError
        self.datasets = datasets

    def need_img_prfx(self):
        return ('_flmg_' in self.args.setting)

    def model_taking_noimg(self):
        return ('_noimg_' in self.args.setting)

    def setup_tk_proc(self):
        self.collate_fn = tk_pad_collate_fn
        if self.args.setting is not None:
            if ('get_dataset_func' in self.key_params):
                get_dataset_func = self.key_params['get_dataset_func']
                train_dataset = get_dataset_func()
                try:
                    self.tokenizer = train_dataset.tokenizer
                except:
                    self.tokenizer = get_dataset_func.keywords['tokenizer']
                self.datasets['main'].image_processor\
                        = get_dataset_func.keywords['processor_func']()
                if self.need_img_prfx():
                    self.collate_fn = functools.partial(
                            tk_pad_collate_fn,
                            add_image_pfx=True)
                if self.model_taking_noimg():
                    self.datasets['main'].with_img = False
            else:
                raise NotImplementedError
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained)
        except:
            raise NotImplementedError

    def batch_list_data(self, now_data):
        batch_data = self.collate_fn(
                now_data,
                self.tokenizer)
        return batch_data

    def do_now_dataset(self):
        nlls = 0
        all_valid_tks = 0
        all_len = len(self.now_dataset)
        for sta_idx in tqdm(range(
                0, all_len,
                self.batch_size), desc='Digest dataset'):
            end_idx = min(all_len, sta_idx + self.batch_size)
            now_data = [
                    self.now_dataset[_idx]
                    for _idx in range(sta_idx, end_idx)]
            batch_data = self.batch_list_data(now_data)
            batch_data['labels'] = batch_data['labels'].clone()
            if not self.need_img_prfx():
                batch_data['labels'][:, :self.args.ctx_len] = -100
            else:
                batch_data['labels'][:, :(self.args.ctx_len + 1)] = -100
            batch_data['labels'][batch_data['labels'] == self.tokenizer.pad_token_id] = -100
            num_valid_tks = torch.sum(batch_data['labels'] != -100).cpu().numpy()
            for key in batch_data:
                batch_data[key] = batch_data[key].cuda()
            with torch.no_grad():
                outputs = self.lm.model(**batch_data)
            loss = outputs.loss
            neg_log_likelihood = loss.cpu().numpy() * num_valid_tks
            nlls += neg_log_likelihood
            all_valid_tks += num_valid_tks
        ppl = np.exp(nlls / all_valid_tks)
        print(getattr(self, 'curr_ckpt', None), ppl)
        return ppl

    def do_one_eval(self, results):
        for which_ds in tqdm(self.datasets, desc='All datasets'):
            if which_ds in results:
                continue
            self.now_dataset = self.datasets[which_ds]
            results[which_ds] = self.do_now_dataset()
        return results

    def get_result_path(self):
        self.has_finished = False
        args = self.args
        task_in_res = self.get_task_in_res()
        pretrained = getattr(args, 'pretrained', None)
        if pretrained is not None:
            fname = pretrained.replace('/', '_').replace('-', '_')
            result_path = os.path.join(
                    self.result_dir, task_in_res,
                    'pretrained', f'{fname}.pkl')
        else:
            fname = self.exp_id
            result_path = os.path.join(
                    self.result_dir, task_in_res,
                    self.col_name, f'{fname}.pkl')
        self.result_path = result_path

    def eval_all(self):
        args = self.args

        self.get_result_path()
        result_path = self.result_path

        now_results = {}
        os.system('mkdir -p ' + os.path.dirname(result_path))
        if (not args.overwrite) and (os.path.exists(result_path)):
            now_results = pickle.load(open(result_path, 'rb'))

        pretrained = getattr(args, 'pretrained', None)
        if pretrained is not None:
            now_results = self.do_one_eval(now_results)
            pickle.dump(now_results, open(result_path, 'wb'))
            return

        for _ckpt in tqdm(self.all_ckpts):
            if _ckpt not in now_results:
                now_results[_ckpt] = {}
            self.curr_ckpt = _ckpt
            self.load_ckpt(os.path.join(self.exp_folder, _ckpt))
            now_results[_ckpt] = self.do_one_eval(now_results[_ckpt])
            pickle.dump(now_results, open(result_path, 'wb'))


def main():
    parser = get_parser()
    args = parser.parse_args()

    runner = ImgCapPPLRunner(args)
    runner.eval_all()


if __name__ == '__main__':
    main()
