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
from transformers import AutoProcessor, AutoTokenizer
from torch.nn import CrossEntropyLoss

from babylm_baseline_train.perplexity import eval_img_cap_ppl
from babylm_baseline_train.datasets.babyLM_txt_vis import tk_pad_collate_fn
from babylm_baseline_train.train import tk_funcs
from babylm_baseline_train.env_vars import ROOT_DIR_FREQ, DATASET_ROOT_DIR, DEBUG
RESULT_DIR = os.path.join(
        ROOT_DIR_FREQ,
        'llm_devo_image_cls_results')
IMAGENET_DIR = os.path.join(
        DATASET_ROOT_DIR,
        'MiniImageNet')


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
            '--high_level_task', default='imagenet',
            type=str, action='store')
    parser.add_argument(
            '--num_distractors', default=1,
            type=int, action='store')
    parser.add_argument(
            '--num_trials', default=5,
            type=int, action='store')
    parser.add_argument(
            '--prompt_style', default='base',
            type=str, action='store')
    return parser


def load_imagenet_data(image_processor=None):
    # from this link: https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
    class_name_path = 'data/imgnt_class_names.txt'
    with open(class_name_path, 'r') as fin:
        all_lines = fin.readlines()
    class_name_map = {}
    for line in all_lines:
        line = line.strip()
        class_id, _, class_name = line.split()
        class_name = class_name.replace('_', ' ')
        class_name_map[class_id] = class_name

    dataset = {}
    val_classes = os.listdir(
            os.path.join(IMAGENET_DIR, 'val'))
    val_classes = list(sorted(val_classes))
    for val_class in tqdm(val_classes, desc='Loading MiniImageNet val'):
        now_folder = os.path.join(
                IMAGENET_DIR, 'val', val_class)
        image_paths = os.listdir(now_folder)
        image_paths = list(sorted(image_paths))
        pil_images = []
        for _path in image_paths:
            now_image = Image.open(
                    os.path.join(now_folder, _path))
            now_image = now_image.convert(mode='RGB')
            if image_processor is not None:
                now_image = image_processor(
                        now_image,
                        return_tensors='pt')['pixel_values'][0]
            pil_images.append(now_image)
        dataset[val_class] = {
                'image_paths': image_paths,
                'pil_images': pil_images,
                'class_name': class_name_map[val_class],
                }
    return dataset


class ImgClsRunner(eval_img_cap_ppl.ImgCapPPLRunner):
    def __init__(self, args, result_dir=RESULT_DIR):
        self.args = args
        self.batch_size = 16
        self.result_dir = RESULT_DIR
        self.get_key_params()
        self.get_lm_model()
        self.get_all_ckpts()
        self.setup_tk_proc()
        self.load_datasets()
        self.loss_fct = CrossEntropyLoss(reduction='none').cuda()

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
                self.image_processor = get_dataset_func.keywords['processor_func']()
                if self.need_img_prfx():
                    self.collate_fn = functools.partial(
                            tk_pad_collate_fn,
                            add_image_pfx=True)
            else:
                raise NotImplementedError
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.is_pretrained_image_models():
            self.image_processor = AutoProcessor.from_pretrained(
                    self.args.pretrained)
        else:
            self.image_processor = AutoProcessor.from_pretrained(
                    'microsoft/git-large')

    def is_pretrained_image_models(self):
        if self.args.pretrained is None:
            return True
        if 'git' in self.args.pretrained:
            return True
        if 'flamingo' in self.args.pretrained:
            return True
        return False

    def get_task_in_res(self):
        return self.args.high_level_task\
                + f'_nd{self.args.num_distractors}'\
                + f'_nt{self.args.num_trials}'\
                + f'_pt{self.args.prompt_style}'

    def load_datasets(self):
        datasets = {}
        if self.args.high_level_task == 'imagenet':
            datasets['main'] = load_imagenet_data(
                    self.image_processor)
        else:
            raise NotImplementedError
        self.datasets = datasets

    def get_classification_text(self, val_class):
        if self.args.prompt_style == 'base':
            class_name = self.now_dataset[val_class]['class_name']
            text = f'This is an image of {class_name}'
        else:
            raise NotImplementedError
        return text

    def construct_trials(self):
        self.all_trials = []
        for val_class in self.now_dataset:
            now_text = self.get_classification_text(val_class)
            all_imgs = self.now_dataset[val_class]['pil_images']
            other_classes = list(self.now_dataset.keys())
            other_classes.remove(val_class)
            for img, _ in product(all_imgs, range(self.args.num_trials)):
                distractors = []
                for _ in range(self.args.num_distractors):
                    dist_class = np.random.choice(other_classes)
                    dist_image_idx = np.random.choice(
                            len(self.now_dataset[dist_class]['pil_images']))
                    dist_image = self.now_dataset[dist_class]['pil_images'][dist_image_idx]
                    distractors.append(dist_image)
                now_trial = ([img,] + distractors, now_text)
                self.all_trials.append(now_trial)

    def img_feat_in_logits(self):
        if self.args.pretrained is None:
            return ('_git_' in self.args.setting)
        else:
            return ('git' in self.args.pretrained)

    def get_per_elm_loss(
            self, logits, input_ids):
        if not self.img_feat_in_logits():
            shift_logits = logits[..., :-1, :].contiguous()
        else:
            if self.args.pretrained is None:
                num_image_tokens = self.lm.model.base_model.git.encoder.layer[0].attention.self.image_patch_tokens
            else:
                num_image_tokens = self.lm.model.git.encoder.layer[0].attention.self.image_patch_tokens
            shift_logits = logits[:, num_image_tokens:-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
        # Flatten the tokens
        per_elm_loss = self.loss_fct(shift_logits.view(-1, logits.shape[-1]), shift_labels.view(-1))
        per_elm_loss = per_elm_loss.reshape(logits.shape[0], -1)
        per_elm_loss = per_elm_loss
        per_elm_loss = per_elm_loss.detach().cpu()
        return per_elm_loss

    def expand_trials(self, now_data, now_trial):
        for _img in now_trial[0]:
            now_data.append(
                    {'pixel_values': _img,
                     'text': now_trial[1]})
        return now_data

    def test_trials(self):
        accurate_nums = 0
        all_len = len(self.all_trials)
        #all_len = 100
        for sta_idx in tqdm(
                range(0, all_len, self.batch_size),
                desc='Digest dataset'):
            end_idx = min(all_len, sta_idx + self.batch_size)
            now_data = []
            for _idx in range(sta_idx, end_idx):
                now_data = self.expand_trials(
                        now_data,
                        self.all_trials[_idx])
            batch_data = self.batch_list_data(now_data)
            if not self.is_pretrained_image_models():  
                batch_data.pop('pixel_values')
            for key in batch_data:
                batch_data[key] = batch_data[key].cuda()
            with torch.no_grad():
                logits = self.lm.model(**batch_data)['logits']
            per_elm_loss = self.get_per_elm_loss(
                    logits, batch_data['input_ids'])
            per_elm_loss = torch.mean(per_elm_loss, dim=-1)
            for _idx in range(sta_idx, end_idx):
                targ_idx = (_idx - sta_idx) * (self.args.num_distractors + 1)
                right_flag = 1
                for dist_idx in range(self.args.num_distractors):
                    dist_idx += targ_idx + 1
                    if per_elm_loss[dist_idx] < per_elm_loss[targ_idx]:
                        right_flag = 0
                accurate_nums += right_flag
                if DEBUG:
                    ipdb.set_trace()
        accuracy = accurate_nums / all_len
        print(getattr(self, 'curr_ckpt', None), accuracy)
        return {'accuracy': accuracy}

    def do_now_dataset(self):
        np.random.seed(0)
        self.construct_trials()
        return self.test_trials()


def main():
    parser = get_parser()
    args = parser.parse_args()

    runner = ImgClsRunner(args)
    runner.eval_all()


if __name__ == '__main__':
    main()
