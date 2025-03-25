import torch
import pdb
import argparse
import functools
import os
import ipdb
import re
import numpy as np
import pickle
from tqdm import tqdm

import pt_framework.checkpoint as checkpoint

from ..models import helper
from ..train.tk_funcs import get_tokenizer_func
from ..train.utils import get_setting_func
from ..train.env_params import MODEL_SAVE_FOLDER
import lm_eval


def get_parser():
    parser = argparse.ArgumentParser(
            description='Build model interface with lm_eval')
    parser.add_argument(
            '--ckpt_path', default=None, type=str, action='store')
    parser.add_argument(
            '--setting', default=None, type=str, action='store')
    parser.add_argument(
            '--all_ckpts', default=False, action='store_true')
    parser.add_argument(
            '--overwrite', default=False, action='store_true')
    parser.add_argument(
            '--which_ckpt', default=None, 
            type=str, action='store')
    return parser


class LMEvalRunner:
    def __init__(self, args):
        self.args = args
        self.get_key_params()
        self.get_lm_model()
        self.get_all_ckpts()
        self.update_all_ckpts()

    def get_key_params(self):
        setting = self.args.setting
        if setting is not None:
            setting_func = get_setting_func(setting)
            key_params = setting_func({})
        else:
            key_params = {}
        self.key_params = key_params

    def get_lm_model(self):
        args = self.args
        self.lm = lm_eval.get_model(
                'hf-causal',
                pretrained='gpt2',
                batch_size=16)
        pretrained = getattr(args, 'pretrained', None)
        if pretrained is not None:
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
        else:
            if 'get_model_func' in self.key_params:
                model = self.key_params['get_model_func']()
            else:
                model = helper.get_opt_func(opt_model_size='125m')

            if 'get_dataset_func' in self.key_params:
                dataset_not_used = self.key_params['get_dataset_func']()
                tokenizer = self.key_params['get_dataset_func'].keywords['tokenizer']
            else:
                tokenizer = get_tokenizer_func()
        self.lm.model = model.to(self.lm.model.device)
        self.lm.tokenizer = tokenizer

        if self.lm.tokenizer.eos_token_id is None:
            self.lm.tokenizer.eos_token_id = self.lm.tokenizer.sep_token_id
        if args.ckpt_path is not None:
            self.load_ckpt(args.ckpt_path)

    def load_ckpt(self, ckpt_path):
        if torch.cuda.is_available():
            checkpoint.load_checkpoint(self.lm.model, ckpt_path)
        else:
            checkpoint.load_checkpoint(
                    self.lm.model, ckpt_path,
                    map_location='cpu')

    def get_ckpts_from_exp_folder(self, exp_folder):
        if not os.path.exists(exp_folder):
            return []
        all_ckpts = os.listdir(exp_folder)
        all_ckpts = list(filter(lambda x: x.startswith('epoch_') and x.endswith('pth'), all_ckpts))
        return all_ckpts

    def update_all_ckpts(self):
        which_ckpt = getattr(self.args, 'which_ckpt', None)
        if which_ckpt is None:
            return
        wanted_ckpts = self.args.which_ckpt.split(',')
        self.all_ckpts = list(filter(
                lambda x: x in wanted_ckpts,
                self.all_ckpts))

    def update_has_eos_token(self):
        if getattr(self, 'has_eos_token', None) is None:
            self.has_eos_token = False
        try:
            empty_tks = self.lm.tokenizer(' ').input_ids[-1]
            self.has_eos_token = self.lm.tokenizer.decode(empty_tks) == self.lm.tokenizer.eos_token
        except:
            pass

    def get_all_ckpts(self):
        args = self.args

        key_params = self.key_params
        self.exp_id = key_params.get('exp_id', 'test_train')
        self.col_name = key_params.get('col_name', 'miniBERTa')
        if not args.all_ckpts:
            return
        self.exp_folder = os.path.join(
                MODEL_SAVE_FOLDER, self.col_name, self.exp_id)
        all_ckpts = self.get_ckpts_from_exp_folder(self.exp_folder)
        self.all_ckpts = all_ckpts
