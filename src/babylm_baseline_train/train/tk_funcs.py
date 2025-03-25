import os
import pdb
import setuptools
import torch

from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
from transformers import PreTrainedTokenizerFast
import babylm_baseline_train


def get_gpt2_tokenizer_func(model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return tokenizer


def get_roberta_tokenizer_func(model_name="roberta-base"):
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    return tokenizer


def get_tokenizer_func(opt_model_size='125m'):
    model_name = f"facebook/opt-{opt_model_size}"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_bos_token = False
    tokenizer.add_special_tokens(
            {
                'bos_token': '<s>', 
                'unk_token': '<unk>',
                'additional_special_tokens': [
                    '<image>', '</c>', 
                    '<PERSON>', # C-12M for person names
                    ]
            })
    return tokenizer


def get_pretrained_tokenizer_func(
        tk_name='tokenizer-babylm100M-32768',
        add_sp_tk=True):
    json_path = os.path.join(
            babylm_baseline_train.__path__[0],
            f'../../tokenizers/{tk_name}/tokenizer.json')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=json_path)
    if add_sp_tk:
        tokenizer.add_special_tokens(
                {
                    'pad_token': '<pad>',
                    'additional_special_tokens': [
                        '<image>',
                        '<PERSON>',
                        ],
                })
    return tokenizer
