import os
import json
import torch
from huggingface_hub import hf_hub_download, login
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from huggingface_hub import HfApi
from torch import load
from transformers.utils import cached_file  # from_pretrained() uses this internally
from transformers import AutoTokenizer

# download files
hf_hub_download(
    repo_id="sentence-transformers/average_word_embeddings_glove.6B.300d", 
    filename="0_WordEmbeddings/pytorch_model.bin", 
    local_dir='glove.6B.300d'
)
hf_hub_download(
    repo_id="sentence-transformers/average_word_embeddings_glove.6B.300d", 
    filename="0_WordEmbeddings/whitespacetokenizer_config.json", 
    local_dir='glove.6B.300d'
)

def get_whitespace_tokenizer(dir):

    config_path = os.path.join(
        dir, '0_WordEmbeddings/whitespacetokenizer_config.json'
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab_size = len(data['vocab'])  # Ensure vocab is properly loaded
    vocab = {v: i for i, v in enumerate(data['vocab'])}
    vocab["[UNK]"] = len(vocab) 
    vocab["[unused0]"] = len(vocab) 
    vocab["[unused1]"] = len(vocab) 
    vocab["[Q]"] = len(vocab) 
    vocab["[D]"] = len(vocab) 

    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Wrap it with Hugging Face tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="PADDING_TOKEN",
        mask_token="[MASK]"
    )

    # Customize stopword removal (this would not be included)
    # stop_words = data['stop_words']
    # def custom_split(text):
    #     tokens = text.split()
    #     return [(word, (i, i+len(word))) for i, word in enumerate(tokens) if word.lower() not in stop_words]
    # hf_tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(custom_split)
    return hf_tokenizer

tokenizer = get_whitespace_tokenizer('glove.6B.300d')
tokenizer.push_to_hub(repo_id='glove.6B.300d')

# push models
# embedding_matrix = torch.load("glove.6B.300d/0_WordEmbeddings/pytorch_model.bin", map_location="cpu")['emb_layer.weight']
# print(len(tokenizer))
# print(embedding_matrix.shape)
# v, d = embedding_matrix.shape
# initializer_range = 0.02 
# embedding_matrix = torch.cat(
#     [embedding_matrix, (torch.rand(len(tokenizer)-v, d) * 2 - 1) * initializer_range], 
#     dim=0
# )
# print(embedding_matrix.shape)
# torch.save({
#     "embeddings.weight": torch.tensor(embedding_matrix),
#     "projection.weight": None,
#     "projection.bias": None,
# }, "glove.6B.300d/0_WordEmbeddings/new_model.pt")
#
# api = HfApi()
# api.upload_file(
#     path_or_fileobj="glove.6B.300d/0_WordEmbeddings/new_model.pt",
#     path_in_repo="pytorch_model.bin",
#     repo_id="DylanJHJ/glove.6B.300d",
#     repo_type="model"
# )
# file_path = cached_file("DylanJHJ/glove.6B.300d", "pytorch_model.bin")
# embedding_matrix = load(file_path)
# print(embedding_matrix['embeddings.weight'].shape)

from colbert.modeling.glove import StaticEmbedding
from colbert.infra.config import ColBERTConfig
baseline = 'hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng'
config = ColBERTConfig.load_from_checkpoint(baseline)
model = StaticEmbedding.from_pretrained('DylanJHJ/glove.6B.300d', colbert_config=config)
model.save_pretrained('glove.6B.300d3')
model = StaticEmbedding.from_pretrained('glove.6B.300d3', colbert_config=config)
print(model)

