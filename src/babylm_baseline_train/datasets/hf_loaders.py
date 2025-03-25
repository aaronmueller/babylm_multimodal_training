from datasets import load_dataset
from ..env_vars import ROOT_DIR, DATASET_ROOT_DIR
import os
import babylm_baseline_train

repo_path = babylm_baseline_train.__path__[0]


def get_babyLM(name, split):
    dataset = load_dataset(
            path=os.path.join(
                repo_path, 'datasets', "babyLM_for_hf.py"),
            name=name,
            split=split)
    return dataset

get_babyLM_2023 = get_babyLM


def get_babyLM_2024(name, split):
    dataset = load_dataset(
            path=os.path.join(
                repo_path, 'datasets', "babyLM_2024_for_hf.py"),
            name=name,
            split=split)
    return dataset
