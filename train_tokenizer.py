import argparse
import datasets
import os
from transformers import T5Config
from typing import Iterator, List, Union
import json

from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing


class SentencePieceUnigramTokenizer(BaseTokenizer):
    """
    This class is a copy of `DeDLOC's tokenizer implementation <https://github.com/yandex-research/DeDLOC/blob/main/sahajbert/tokenizer/tokenizer_model.py>`__ .

    Custom SentencePiece Unigram Tokenizer with NMT, NKFC, spaces and lower-casing characters normalization
    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(
        self,
        replacement: str = "▁",
        add_prefix_space: bool = True,
        unk_token: Union[str, AddedToken] = "<unk>",
        eos_token: Union[str, AddedToken] = "</s>",
        pad_token: Union[str, AddedToken] = "<pad>",
        mask_token: Union[str, AddedToken] = "<mask>",
    ):
        self.special_tokens = {
            "pad": {"id": 0, "token": pad_token},
            "eos": {"id": 1, "token": eos_token},
            "unk": {"id": 2, "token": unk_token},
            "mask": {"id": 3, "token": mask_token},
        }

        self.special_tokens_list = [None] * len(self.special_tokens)
        for token_dict in self.special_tokens.values():
            self.special_tokens_list[token_dict["id"]] = token_dict["token"]

        tokenizer = Tokenizer(Unigram())

        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Nmt(),
                normalizers.NFKC(),
                normalizers.Replace(Regex(" {2,}"), " "),
                normalizers.Lowercase(),
            ]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.Punctuation(),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

        tokenizer.post_processor = TemplateProcessing(
            single=f"$A {self.special_tokens['eos']['token']}",
            special_tokens=[(self.special_tokens["eos"]["token"], self.special_tokens["eos"]["id"])],
        )

        parameters = {
            "model": "SentencePieceUnigram",
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given files"""

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )

        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

        self.add_unk_id()
        self.add_mask_id()

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given iterator"""

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )

        self._tokenizer.train_from_iterator(iterator, trainer=trainer)

        self.add_unk_id()

    def add_unk_id(self):
        tokenizer_json = json.loads(self._tokenizer.to_str())

        tokenizer_json["model"]["unk_id"] = self.special_tokens["unk"]["id"]

        self._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

    def add_mask_id(self):
        tokenizer_json = json.loads(self._tokenizer.to_str())

        tokenizer_json["model"]["mask_id"] = self.special_tokens["mask"]["id"]

        self._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
        
# get dataset path from argument
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", "-d", type=str, default="/scratch/myh2014/pre-pre-training/data/babylm_data/babylm_10M/")
parser.add_argument("--vocabsize", "-v", type=int, default=6144)
parser.add_argument("--dataname", "-n", type=str)
args = parser.parse_args()

vocab_size = args.vocabsize
input_sentence_size = None

# Initialize a dataset
if args.dataname != 'babylmvis2024':
    dataset = datasets.load_dataset("text", data_dir=args.datadir, split="train")
else:
    from babylm_baseline_train.datasets import hf_loaders
    dataset = hf_loaders.get_babyLM_2024('babyLM-Vis-Text', 'train')

tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
    batch_length = 128
    for i in range(0, input_sentence_size, batch_length):
        yield dataset[i: i + batch_length]["text"]

# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(input_sentence_size=input_sentence_size),
    vocab_size=vocab_size,
    show_progress=True,
)

# Save tokenizer to directory
dataname = args.dataname
file_pth = f"./tokenizer-{dataname}/"
if not os.path.exists(file_pth):
    os.mkdir(file_pth)

tokenizer.save(file_pth + "tokenizer.json")
