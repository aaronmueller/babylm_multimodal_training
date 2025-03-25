# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Union

import transformers.models.opt.configuration_opt as configuration_opt


class FlamingoConfig(configuration_opt.OPTConfig, dict):
    model_type = "flamingo"
    def __init__(
        self,
        cross_attn_every=2,
        vocab_size=32778,
        media_token_id=32768,
        **kwargs,
    ):
        configuration_opt.OPTConfig.__init__(
                self, vocab_size=vocab_size, **kwargs)
        self.media_token_id = media_token_id
        self.cross_attn_every = cross_attn_every
        dict.__init__(self, **self.__dict__)
