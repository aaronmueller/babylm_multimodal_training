import torch
from abc import ABC, abstractmethod
import ipdb
import numpy as np
from itertools import product


def tk_pad_collate_fn(
        all_data, tokenizer,
        add_image_pfx=False):
    keys = list(all_data[0].keys())
    assert 'text' in keys
    all_texts = [_data['text'] for _data in all_data]
    if add_image_pfx:
        all_texts = [
                '<image> ' + _data['text']\
                for _data in all_data]
    all_texts = tokenizer(
            all_texts, padding='longest', return_tensors="pt",
            truncation=True, max_length=128)
    all_texts = all_texts.input_ids
    ret_dict = dict(
            input_ids=all_texts,
            labels=all_texts,
            )
    keys.remove('text')
    for other_key in keys:
        all_other_value = [_data[other_key] for _data in all_data]
        all_other_value = torch.stack(all_other_value, 0)
        ret_dict[other_key] = all_other_value
    return ret_dict


class PerItemCombineDataset:
    def __init__(
            self, datasets,
            per_item_nums=[1,1],
            rd_ep_idx_order=True):
        self.datasets = datasets
        self.per_item_nums = per_item_nums
        assert len(per_item_nums) == len(datasets)
        data_len = len(datasets[0]) // per_item_nums[0]
        for now_dataset, now_num in zip(datasets, per_item_nums):
            now_len = len(now_dataset) // now_num
            if now_len > data_len:
                data_len = now_len
        self.dataset_len = data_len
        self.rd_ep_idx_order = rd_ep_idx_order
        self.set_epoch(0)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        whole_item_list = []
        for _dataset, now_num, now_idxs\
                in zip(self.datasets, self.per_item_nums, self.dataset_idxs):
            now_dataset_list = []
            sta_idx = idx * now_num
            end_idx = (idx+1) * now_num
            for _idx in range(sta_idx, end_idx):
                if not self.rd_ep_idx_order:
                    if _idx >= len(_dataset):
                        _idx = _idx % len(_dataset)
                now_dataset_list.append(
                        _dataset[int(now_idxs[_idx])])
            whole_item_list.append(tuple(now_dataset_list))
        return tuple(whole_item_list)

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        dataset_idxs = []
        for which_dataset, _dataset in enumerate(self.datasets):
            if not self.rd_ep_idx_order:
                dataset_idxs.append(
                        np.random.permutation(len(_dataset)))
            else:
                now_idxs = []
                now_idx_len = 0
                max_len = self.dataset_len * self.per_item_nums[which_dataset]
                while now_idx_len < max_len:
                    now_idxs.append(
                            np.random.permutation(len(_dataset)))
                    now_idx_len += len(_dataset)
                dataset_idxs.append(np.concatenate(now_idxs))
        self.dataset_idxs = dataset_idxs


class CombineCollate:
    def __init__(
            self, tokenizer,
            name_prefixs=['', 'noimg_'],
            add_image_pfx=False):
        self.tokenizer = tokenizer
        self.name_prefixs = name_prefixs
        self.add_image_pfx = add_image_pfx

    def collect_examples(self, examples, idx):
        now_list = []
        for _example in examples:
            now_list.extend(_example[idx])
        return tuple(now_list)

    def simple_stack(self, examples):
        keys = list(examples[0].keys())
        ret_dict = {}
        for other_key in keys:
            if isinstance(examples[0][other_key], list):
                all_other_value = [
                        torch.LongTensor(_data[other_key])
                        for _data in examples]
            else:
                all_other_value = [_data[other_key] for _data in examples]
            all_other_value = torch.stack(all_other_value, 0)
            ret_dict[other_key] = all_other_value
        return ret_dict

    def __call__(self, examples):
        num_datasets = len(examples[0])
        assert num_datasets == len(self.name_prefixs)
        all_results = {}
        for idx in range(num_datasets):
            tmp_examples = self.collect_examples(examples, idx)
            if 'text' in tmp_examples[0]:
                now_result_dict = tk_pad_collate_fn(
                        tmp_examples, self.tokenizer,
                        add_image_pfx=self.add_image_pfx)
            else:
                now_result_dict = self.simple_stack(tmp_examples)
            new_result_dict = {}
            for key, value in now_result_dict.items():
                new_key = self.name_prefixs[idx] + key
                new_result_dict[new_key] = value
                assert new_key not in all_results
            all_results.update(new_result_dict)
        return all_results


def add_comb_collate_fn(key_params, tokenizer):
    if 'add_train_loader_kwargs' not in key_params:
        key_params['add_train_loader_kwargs'] = {}
    key_params['add_train_loader_kwargs'].update(
            {'collate_fn': CombineCollate(tokenizer=tokenizer)})
    return key_params

def add_comb_collate_fn_wimg(key_params, tokenizer):
    if 'add_train_loader_kwargs' not in key_params:
        key_params['add_train_loader_kwargs'] = {}
    key_params['add_train_loader_kwargs'].update(
            {'collate_fn': CombineCollate(
                tokenizer=tokenizer,
                add_image_pfx=True)})
    return key_params


def get_babyLM_txt_vis(
        tokenizer=None,
        processor_func=None,
        txt_kwargs={},
        vis_kwargs={},
        use_old_data=True,
        **kwargs):
    from babylm_baseline_train.datasets import local_narr, cc_3M, babyLM
    if use_old_data:
        txt_data = babyLM.get_babyLM_50M(
                tokenizer=tokenizer,
                **txt_kwargs)
    else:
        txt_data = babyLM.get_babyLM_text_vis_2024(
                tokenizer=tokenizer,
                **txt_kwargs)

    cc_data = cc_3M.ConceptualCaptions3M(
            image_processor_func=processor_func,
            **vis_kwargs)
    ln_data = local_narr.LocalizedNarrativesImgTxt(
            image_processor_func=processor_func,
            **vis_kwargs)
    vis_data = torch.utils.data.ConcatDataset([cc_data, ln_data])
    cmb_dataset = PerItemCombineDataset(
            [vis_data, txt_data],
            **kwargs)
    return cmb_dataset
