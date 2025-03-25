import babylm_baseline_train.datasets.babyLM_txt_vis as babyLM_txt_vis
from babylm_baseline_train.configs.general import\
        add_func_in_general, get_general_data_func,\
        get_seq_funcs_to_update_key_params, add_set_epoch
import functools
from itertools import product
import babylm_baseline_train.train.tk_funcs as tk_funcs
import babylm_baseline_train.datasets.babyLM_txt_vis as babyLM_txt_vis
from transformers import ViTFeatureExtractor

tokenizer = tk_funcs.get_pretrained_tokenizer_func(
        tk_name='tokenizer-babylmvis2024-32768',
        )
base_post_func = get_seq_funcs_to_update_key_params(
        [add_set_epoch,
         functools.partial(
             babyLM_txt_vis.add_comb_collate_fn,
             tokenizer=tokenizer)])

KWARGS = dict(
        all_things=globals(),
        specify_iter=[],
        specify_epoch=[5, 10, 20, 25],
        post_func=base_post_func)
DATA_KWARGS = dict(
        max_epochs=30, ckpt_save_interval=15,
        col_name='babyLM_txt_vis_new',
        tokenizer=tokenizer)


def add_exp_seeds(
        exp_names, seeds, data_func,
        model_name=None,
        ):
    for exp_name, seed in zip(exp_names, seeds):
        add_func_in_general(
                func_name=exp_name,
                data_func=get_general_data_func(
                    data_func,
                    **DATA_KWARGS),
                seed=seed,
                model_name=model_name,
                **KWARGS)

processor_func = lambda: ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
txt_only_data_func = functools.partial(
        babyLM_txt_vis.get_babyLM_txt_vis,
        processor_func=processor_func,
        vis_kwargs={'with_img': False},
        use_old_data=False,
        )
wimg_data_func = functools.partial(
        babyLM_txt_vis.get_babyLM_txt_vis,
        processor_func=processor_func,
        use_old_data=False,
        )

txt_only_model_keys = [
        'base_noimg_1v1',
        'base_noimg_1vd5',
        'base_noimg_1vd25',
        'base_noimg_1vd125',
        ]
for model_key in txt_only_model_keys:
    add_exp_seeds(
            exp_names=[
                f'{model_key}_s1',
                f'{model_key}_s2',
                ], 
            seeds=[1, 2], 
            data_func=txt_only_data_func,
            model_name=f'cmb_{model_key}')


git_model_keys = [
        'base_git_1v1',
        'base_git_1vd5',
        'base_git_1vd25',
        'base_git_1vd125',
        ]
for model_key in git_model_keys:
    add_exp_seeds(
            exp_names=[
                f'{model_key}_s1',
                f'{model_key}_s2',
                ], 
            seeds=[1, 2], 
            data_func=wimg_data_func,
            model_name=f'cmb_{model_key}')


flmg_post_func = get_seq_funcs_to_update_key_params(
        [add_set_epoch,
         functools.partial(
             babyLM_txt_vis.add_comb_collate_fn_wimg,
             tokenizer=tokenizer)])
FLMG_KWARGS = dict(
        all_things=globals(),
        specify_iter=[],
        specify_epoch=[5, 10, 20, 25],
        post_func=flmg_post_func)

def add_flmg_exp_seeds(
        exp_names, seeds, data_func,
        model_name=None,
        ):
    for exp_name, seed in zip(exp_names, seeds):
        add_func_in_general(
                func_name=exp_name,
                data_func=get_general_data_func(
                    data_func,
                    **DATA_KWARGS),
                seed=seed,
                model_name=model_name,
                **FLMG_KWARGS)

flmg_model_keys = [
        'base_flmg_1v1',
        'base_flmg_1vd5',
        'base_flmg_1vd25',
        'base_flmg_1vd125',
        ]
for model_key in flmg_model_keys:
    add_flmg_exp_seeds(
            exp_names=[
                f'{model_key}_s1',
                f'{model_key}_s2',
                ], 
            seeds=[1, 2], 
            data_func=wimg_data_func,
            model_name=f'cmb_{model_key}')
