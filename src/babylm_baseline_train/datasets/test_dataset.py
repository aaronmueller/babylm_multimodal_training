from transformers import ViTFeatureExtractor
from babylm_baseline_train.datasets.local_narr import LocalizedNarrativesImgTxt
from babylm_baseline_train.datasets.cc_3M import ConceptualCaptions3M
import babylm_baseline_train.datasets.babyLM as babyLM
import babylm_baseline_train.datasets.babyLM_txt_vis as babyLM_txt_vis
from babylm_baseline_train.train.tk_funcs import get_pretrained_tokenizer_func
import ipdb
from tqdm import tqdm
import re


def test_local_narr():
    processor_func = lambda: ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    dataset = LocalizedNarrativesImgTxt(
            image_processor_func=processor_func)
    ipdb.set_trace()


def test_cc3M():
    processor_func = lambda: ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    dataset = ConceptualCaptions3M(
            image_processor_func=processor_func)
    one_item = dataset[0]
    ipdb.set_trace()


def test_cc3M_valid():
    processor_func = lambda: ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    dataset = ConceptualCaptions3M(
            image_processor_func=processor_func,
            split='validation')
    one_item = dataset[0]
    ipdb.set_trace()


def count_no_words_local_narr():
    processor_func = lambda: ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    dataset = LocalizedNarrativesImgTxt(
            image_processor_func=processor_func,
            with_img=False)
    all_count = 0
    for idx in tqdm(list(range(len(dataset)))):
        anno = dataset.load_raw_anno(idx)
        caption = dataset.clean_caption(anno['caption'])
        count = len(re.findall(r'\w+', caption))
        all_count += count
    print(all_count)


def count_no_words_cc3M():
    processor_func = lambda: ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    dataset = ConceptualCaptions3M(
            image_processor_func=processor_func)
    all_count = 0
    for idx in tqdm(list(range(len(dataset)))):
        _, cap = dataset.valid_fpath_cap[idx]
        count = len(re.findall(r'\w+', cap))
        all_count += count
    print(all_count)


def test_multi_modal():
    #babyLM_50M = babyLM.get_babyLM_50M()
    processor_func = lambda: ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    tokenizer = get_pretrained_tokenizer_func()
    dataset = babyLM_txt_vis.get_babyLM_txt_vis(
            processor_func=processor_func,
            tokenizer=tokenizer)
    collator = babyLM_txt_vis.CombineCollate(tokenizer=tokenizer)
    examples = []
    for batch_idx in tqdm(list(range(238, 250))):
        offset_idx = batch_idx * 128
        for idx in range(128):
            examples.append(dataset[idx + offset_idx])
    ipdb.set_trace()


if __name__ == '__main__':
    #test_local_narr()
    #test_cc3M()
    count_no_words_local_narr()
    #count_no_words_cc3M()
    #test_multi_modal()
    #test_cc3M_valid()
