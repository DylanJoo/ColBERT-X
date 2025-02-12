import json
import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from collections import OrderedDict
from transformers.utils import cached_file  # from_pretrained() uses this internally
from transformers import PreTrainedModel, XLMRobertaConfig
from safetensors import safe_open
from safetensors.torch import load_model
from transformers.activations import gelu
from copy import deepcopy

class StaticEmbedding(PreTrainedModel):
    config_class = XLMRobertaConfig
    base_model_prefix = "glove"

    @classmethod
    def from_pretrained(cls, name_or_path, colbert_config, **kwargs):
        config = AutoConfig.from_pretrained(colbert_config.model_name)
        config._name_or_path = name_or_path
        config.hidden_size = colbert_config.lite_hidden_size
        config.num_hidden_layers = colbert_config.lite_num_hidden_layers
        config.num_attention_heads = colbert_config.lite_num_attention_heads

        if colbert_config.lite_encoder_init is not None:
            file_path = cached_file(name_or_path, "pytorch_model.bin")
            model = torch.load(file_path)
            instance = cls(config, colbert_config)
            # if cls.__class__ == StaticEmbedding: 
            #     instance.embeddings.load_state_dict({'weight': model['embeddings.weight']})
            #     if model.get('projection.weight', None) is not None:
            #         instance.projection.load_state_dict({'weight': model['projection.weight']})
            #     if model.get('projection.bias', None) is not None:
            #         instance.projection.load_state_dict({'bias': model['projection.bias']})
            #     return instance
            # else:
            instance.glove.embeddings.load_state_dict({'weight': model['embeddings.weight']})
            if model.get('projection.weight', None) is not None:
                instance.glove.projection.load_state_dict({'weight': model['projection.weight']})
            if model.get('projection.bias', None) is not None:
                instance.glove.projection.load_state_dict({'bias': model['projection.bias']})
            return instance
        else:
            instance = cls(config, colbert_config)
            load_model(instance, f"{name_or_path}/model.safetensors")
            return instance

class Glove(StaticEmbedding):

    def __init__(self, config, colbert_config=None):
        super().__init__(config, colbert_config)
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        output_hidden_size = 1024 if colbert_config is None else colbert_config.lite_hidden_size
        self.embeddings = nn.Embedding(len(self.tokenizer), 300)
        self.projection = nn.Linear(300, output_hidden_size)

    # [modified]
    def forward(self, input_ids, attention_mask=None):
        input_embeds = self.embeddings(input_ids)

        if attention_mask is not None:
            input_embeds = input_embeds.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )

        avg_embeds = input_embeds.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        input_embeds[:, 0, :] = avg_embeds
        input_embeds = self.projection(input_embeds)

        if attention_mask is not None:
            input_embeds = input_embeds.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )

        return (input_embeds, )

    def get_input_embeddings(self):
        return self.embeddings

    def init_weights(self):
        pass
