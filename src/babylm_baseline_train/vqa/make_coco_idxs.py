import os
import json


folder = '/om2/group/evlab/llm_dataset/LocalNarratives/LocalNarratives/MSCOCO/'
all_files = os.listdir(folder)
valid_idxs = [int(_idx[:-4]) for _idx in all_files]

valid_dict = {}
for _idx in valid_idxs:
    valid_dict[_idx] = 1

json.dump(valid_dict, open('mscoco_train_idxs.json', 'w'))
