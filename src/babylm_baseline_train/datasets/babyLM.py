from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
import pdb

from .utils import Group_Texts
from .base import BaseGroupDataset
from . import hf_loaders

BABYLM_2024_NAMEs = ['babyLM-Vis-Text']

class BabyLM(BaseGroupDataset):
    def __init__(
            self, 
            seq_len=128, tokenizer=None,
            name='babyLM-10M',
            ):
        super().__init__(seq_len, tokenizer)
        self.name = name

    def get_dataset(self):
        if self.name not in BABYLM_2024_NAMEs:
            self.dataset = hf_loaders.get_babyLM(
                    name=self.name,
                    split="train")
        else:
            self.dataset = hf_loaders.get_babyLM_2024(
                    name=self.name,
                    split="train")


def get_babyLM_10M(seq_len=128, tokenizer=None, just_dataset=False):
    dataset_builder = BabyLM(
            seq_len=seq_len,
            tokenizer=tokenizer,
            name='babyLM-10M',
            )
    return dataset_builder.get_group_dataset(just_dataset=just_dataset)


def get_babyLM_100M(seq_len=128, tokenizer=None, just_dataset=False):
    dataset_builder = BabyLM(
            seq_len=seq_len,
            tokenizer=tokenizer,
            name='babyLM-100M',
            )
    return dataset_builder.get_group_dataset(just_dataset=just_dataset)


def get_babyLM_50M(seq_len=128, tokenizer=None, just_dataset=False):
    # Used for Multi-modality track
    dataset_builder = BabyLM(
            seq_len=seq_len,
            tokenizer=tokenizer,
            name='babyLM-50M',
            )
    return dataset_builder.get_group_dataset(just_dataset=just_dataset)


def get_babyLM_text_vis_2024(seq_len=128, tokenizer=None, just_dataset=False):
    # Used for Multi-modality track
    dataset_builder = BabyLM(
            seq_len=seq_len,
            tokenizer=tokenizer,
            name='babyLM-Vis-Text',
            )
    return dataset_builder.get_group_dataset(just_dataset=just_dataset)
