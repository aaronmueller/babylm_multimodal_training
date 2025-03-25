import transformers
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
from typing import List, Optional, Tuple, Union
import warnings
import ipdb
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from itertools import product
import numpy as np
import transformers.models.git.modeling_git as modeling_git
import transformers.models.vit.modeling_vit as modeling_vit
from transformers.models.opt.modeling_opt import OPTConfig
import transformers.models.opt.modeling_opt as hg_opt
import transformers.models.clip.modeling_clip as modeling_clip


class GitForCausalLM(modeling_git.GitForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.output
        self.output = nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False)
        self.post_init()

        del self.git.image_encoder
        self.git.image_encoder = ViTModel.from_pretrained('facebook/dino-vitb16')
        dino_cfg = self.git.image_encoder.config
        config = self.git.config
        config.vision_config.hidden_size = dino_cfg.hidden_size

        del self.git.visual_projection
        self.git.visual_projection = modeling_git.GitProjection(config)
        num_tks = (dino_cfg.image_size // dino_cfg.patch_size) ** 2 + 1
        self.git.encoder.layer[0].attention.self.image_patch_tokens = num_tks
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_git.CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.git(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.output(sequence_output)

        loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            if pixel_values is not None:
                num_image_tokens = self.git.encoder.layer[0].attention.self.image_patch_tokens
            else:
                num_image_tokens = 0
            shifted_logits = logits[:, num_image_tokens:-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shifted_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return modeling_git.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
