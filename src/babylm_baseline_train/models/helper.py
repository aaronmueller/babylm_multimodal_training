import os
import pdb
import setuptools
import torch
import ipdb
import copy
import functools
from itertools import product

from transformers import OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTConfig
from .git import get_dino_git_func, get_noimg_more_lyrs_git_func
from .multi_comb import get_multi_comb_model
from .flamingo_hf_opt import get_dino_flamingo_model_func
DEBUG = int(os.environ.get(
        'DEBUG',
        '0')) == 1


def get_opt_func(opt_model_size='125m'):
    model_name = f"facebook/opt-{opt_model_size}"
    config = OPTConfig.from_pretrained(model_name)
    model = OPTForCausalLM(config=config)
    return model


def get_roberta_func(model_name="roberta-base", tokenizer=None):
    from transformers import RobertaConfig, RobertaForMaskedLM
    config = RobertaConfig.from_pretrained(model_name)
    model = RobertaForMaskedLM(config)
    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer))
    return model


def get_ltg_bert(
        model_name='norbert3-base',
        tokenizer=None,
        large_vocab=False):
    from .ltgbert import modeling_ltgbert, configuration_ltgbert
    config = configuration_ltgbert.LtgBertConfig(model_name)
    if tokenizer is None:
        from ..train import tk_funcs
        tokenizer = tk_funcs.get_roberta_tokenizer_func()
    config.vocab_size = tokenizer.vocab_size
    if large_vocab:
        config.vocab_size += 100
    config.pad_token_id = tokenizer.pad_token_id
    model = modeling_ltgbert.LtgBertForMaskedLM(config)
    return model



MODEL_NAME_TO_FUNC = {
        'base_noimg_git': functools.partial(
            get_noimg_more_lyrs_git_func,
            num_layers=12,
            output_with_bias=False,
            tie_output=True),
        'base_dino_tie_git': functools.partial(
            get_dino_git_func,
            num_layers=12, tie_output=True),
        }


cmb_exp_suffix_to_mixw_map = {
        '_1v1': [1, 1],
        '_1v2': [1, 2],
        '_1vd5': [1, 0.5],
        '_1vd25': [1, 0.25],
        '_1vd125': [1, 0.125],
        '_1vd0625': [1, 0.0625],
        '_1vd03125': [1, 0.03125],
        }
cmb_exp_name_to_func_map = {
        'base_noimg': functools.partial(
            get_multi_comb_model,
            base_model_func=MODEL_NAME_TO_FUNC['base_noimg_git'],
            add_vis=False),
        'base_git': functools.partial(
            get_multi_comb_model,
            base_model_func=MODEL_NAME_TO_FUNC['base_dino_tie_git'],
            add_vis='only_pixel'),
        'base_flmg': functools.partial(
            get_multi_comb_model,
            base_model_func=get_dino_flamingo_model_func,
            add_vis='only_pixel'),
        }

for _exp_name, _exp_suffix in product(
        list(cmb_exp_name_to_func_map.keys()),
        list(cmb_exp_suffix_to_mixw_map.keys())):
    MODEL_NAME_TO_FUNC[f'cmb_{_exp_name}{_exp_suffix}'] = functools.partial(
            cmb_exp_name_to_func_map[_exp_name],
            mix_weights=cmb_exp_suffix_to_mixw_map[_exp_suffix])
