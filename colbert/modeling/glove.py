import json
import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from collections import OrderedDict
from transformers.utils import cached_file  # from_pretrained() uses this internally

class StaticEmbedding(nn.Module):
    base_model_prefix = "static_embs"

    def __init__(self, config, dim=300):
        super().__init__()
        self.config = config 
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self.static_embs = nn.Sequential(OrderedDict([
            ('embeddings', nn.Embedding(len(self.tokenizer), dim)),
            ('projection', nn.Linear(dim, config.hidden_size, bias=True)),
        ]))

        # self.stop_words = data.get('stop_words', [])
        # self.do_lower_case = data.get('do_lower_case', False)

    def forward(self, input_ids, attention_mask=None):
        input_embeds = self.static_embs(input_ids)
        if attention_mask is not None:
            input_embeds = input_embeds.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
        return input_embeds

    @classmethod
    def from_pretrained(cls, name_or_path, colbert_config, lite_encoder: bool = False):
        """load pretrained static embeddings."""
        config = AutoConfig.from_pretrained(colbert_config.model_name)
        instance = cls(config, colbert_config)  # create an instance first

        file_path = cached_file(name_or_path, "pytorch_model.bin")
        embedding_matrix = torch.load(file_path)
        print(embedding_matrix)
        instance.static_embs.static_embs.embeddings.load_state_dict({'weight': embedding_matrix})
        # instance.static_embs.load_state_dict({'weight': embedding_matrix})
        return instance 

    def get_input_embeddings(self):
        return self.static_embs.embeddings

    def init_weights(self):
        pass

