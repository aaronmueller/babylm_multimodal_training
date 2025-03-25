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

from babylm_baseline_train.datasets.babyLM_txt_vis import tk_pad_collate_fn
from babylm_baseline_train.env_vars import ROOT_DIR_FREQ, DATASET_ROOT_DIR, DEBUG
from babylm_baseline_train.image_cls.zero_shot_classifier import ImgClsRunner
from babylm_baseline_train.vqa.zero_shot_cls import VisualQARunner
RESULT_DIR = os.path.join(
        ROOT_DIR_FREQ,
        'llm_devo_vqa_zero_shot_results')


def get_parser():
    parser = argparse.ArgumentParser(
            description='Get similar ans from a pretrained model')
    parser.add_argument(
            '--ckpt_path', default=None, type=str, action='store')
    parser.add_argument(
            '--pretrained', required=True,
            default=None, type=str, action='store')
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
            '--num_ans', default=20,
            type=int, action='store')
    return parser


class VisualQAAnsGen(VisualQARunner):
    def get_task_in_res(self):
        return self.args.high_level_task\
                + f'_na{self.args.num_ans}'

    def get_all_model_inputs(self, string_list):
        tokenizer = self.lm.tokenizer
        all_input_ids = []
        all_attention_masks = []
        for string in string_list:
            input_str = [' ' + string]
            if self.args.pretrained == 'gpt2':
                input_str = [tokenizer.bos_token + input_str[0]]
            inputs = tokenizer(
                    input_str, return_tensors="pt",
                    add_special_tokens=True)
            all_input_ids.append(inputs.input_ids)
            if 'attention_mask' in inputs:
                all_attention_masks.append(inputs.attention_mask)
        return all_input_ids, all_attention_masks

    def get_all_embeddings(
            self, input_ids, attention_masks,
            additional_forward_params=None):
        model = getattr(self.lm, 'gpt2', None) # the legacy name for the loaded model
        if model is None:
            model = self.lm.model
        single_extra_forward_kwargs = getattr(
                self.lm, 'extra_forward_kwargs', {})
        input_lens = [_id.shape[1] for _id in input_ids]
        input_lens = np.asarray(input_lens)
        sorted_idxs = np.argsort(input_lens)

        returned_embeddings = []
        diff_lens = np.unique(input_lens)
        all_idxs = []
        for _len in tqdm(diff_lens, desc='All len'):
            now_idx = np.where(input_lens == _len)[0]
            all_idxs.append(now_idx)
            new_input_ids = torch.cat([input_ids[_idx] for _idx in now_idx])
            if len(attention_masks) > 0:
                new_att_masks = torch.cat([attention_masks[_idx] for _idx in now_idx])
            else:
                new_att_masks = None

            for sta_idx in range(0, len(now_idx), self.batch_size):
                end_idx = min(sta_idx + self.batch_size, len(now_idx))
                _input_ids = new_input_ids[sta_idx : end_idx]
                if new_att_masks is not None:
                    _att_masks = new_att_masks[sta_idx : end_idx]

                extra_forward_kwargs = copy.copy(single_extra_forward_kwargs)
                if additional_forward_params is not None:
                    for key, value in additional_forward_params.items():
                        now_value = [
                                value[_idx]
                                for _idx in now_idx[sta_idx : end_idx]]
                        now_value = torch.stack(now_value, dim=0).to(self.lm._device)
                        extra_forward_kwargs[key] = now_value

                for key in extra_forward_kwargs:
                    if extra_forward_kwargs[key].size(0) != end_idx - sta_idx:
                        now_ts = extra_forward_kwargs[key]
                        rep_sizes = [1] * now_ts.ndim
                        rep_sizes[0] = end_idx - sta_idx
                        extra_forward_kwargs[key] = now_ts.repeat(*rep_sizes)
                
                with torch.no_grad():
                    if new_att_masks is not None:
                        text_features = model(
                                input_ids=_input_ids.to(self.lm._device), 
                                attention_mask=_att_masks.to(self.lm._device),
                                output_hidden_states=True,
                                **extra_forward_kwargs)
                    else:
                        text_features = model(
                                input_ids=_input_ids.to(self.lm._device), 
                                output_hidden_states=True,
                                **extra_forward_kwargs)

                if 'hidden_states' in text_features:
                    hidden_states = text_features['hidden_states']
                elif 'multimodal_output' in text_features: # special case for Flava models
                    hidden_states = []
                    for _state in text_features['text_output']['hidden_states']:
                        hidden_states.append(_state)
                    num_txt_tks = text_features['text_embeddings'].shape[1]
                    for _state in text_features['multimodal_output']['hidden_states']:
                        hidden_states.append(_state[:, -num_txt_tks:, :])
                elif 'text_output' in text_features: # special case for Flava models
                    hidden_states = text_features['text_output']['hidden_states']
                else:
                    hidden_states = text_features[self.hidden_states_pos]
                embeddings = []
                for layer_vectors in hidden_states:
                    if not isinstance(layer_vectors, np.ndarray):
                        layer_vectors = layer_vectors.cpu().numpy()
                    representation = layer_vectors[:, self.extraction_id]
                    embeddings.append(representation)
                returned_embeddings.append(embeddings)
        all_idxs = np.concatenate(all_idxs)
        no_layers = len(returned_embeddings[0])
        new_returned_embeddings = []
        for idx in range(no_layers):
            _embds = np.concatenate(
                    [returned_embeddings[inner_idx][idx]
                    for inner_idx in range(len(returned_embeddings))])
            new_embds = np.zeros_like(_embds)
            for curr_idx, new_idx in enumerate(all_idxs):
                new_embds[new_idx] = _embds[curr_idx]
            new_returned_embeddings.append(new_embds)
        return new_returned_embeddings

    def get_similar_ans(self):
        all_similar_ans = []
        all_len = len(self.all_answers)
        for layer_idx in tqdm(range(len(self.all_embds))):
            now_all_targets = np.transpose(
                    self.all_embds[layer_idx],
                    [1, 0])
            now_similar_ans = []
            for sta_idx in tqdm(
                    range(0, all_len, self.batch_size),
                    desc=f'Layer {layer_idx}'):
                end_idx = min(all_len, sta_idx + self.batch_size)
                now_embds = self.all_embds[layer_idx][sta_idx : end_idx]
                now_sims = np.matmul(
                        now_embds, now_all_targets)
                max_sim_idxs = np.argsort(now_sims, axis=-1)
                max_sim_idxs = max_sim_idxs[:, -(self.args.num_ans+1):]
                now_similar_ans.append(max_sim_idxs)
            now_similar_ans = np.concatenate(
                    now_similar_ans, axis=0)
            all_similar_ans.append(now_similar_ans)
        self.all_similar_ans = all_similar_ans

    def run(self):
        self.batch_size = 16
        self.hidden_states_pos = 2
        input_ids, att_masks = self.get_all_model_inputs(self.all_answers)

        try:
            empty_tks = self.lm.tokenizer(' ').input_ids[-1]
            self.has_eos_token = self.lm.tokenizer.decode(empty_tks) == self.lm.tokenizer.eos_token
        except:
            self.has_eos_token = False

        self.extraction_id = -2
        if not self.has_eos_token:
            self.extraction_id = -1

        self.get_result_path()
        result_path = self.result_path
        os.system('mkdir -p ' + os.path.dirname(result_path))
        print(result_path)

        self.all_embds = self.get_all_embeddings(input_ids, att_masks)
        self.get_similar_ans()
        pickle.dump(
                self.all_similar_ans,
                open(result_path, 'wb'))



def main():
    parser = get_parser()
    args = parser.parse_args()

    runner = VisualQAAnsGen(args)
    runner.run()


if __name__ == '__main__':
    main()
