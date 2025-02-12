import json
import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.bert.modeling_bert import BertModel

class MiniLM(PreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, colbert_config):
        super().__init__(config, add_pooling_layer=False)
        self.config = config
        self.bert = BertModel(config)
        self.proj = nn.Linear(config.hidden_size, colbert_config.lite_hidden_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()
        print('minilm', self)

    def forward(self, inputs):
        output = super().forward(inputs)[0]
        output = self.proj(output)
        return output
