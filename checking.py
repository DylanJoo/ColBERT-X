from safetensors import safe_open
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra.config import ColBERTConfig
from transformers import AutoConfig

## DEBUG1: safetensors vs. loaded
## >> the output are the same
# baseline = 'experiments/colbert-lite-q-L24/none/baseline/12bat.6way/checkpoints/colbert-100000'
# tensors = {}
# with safe_open(f"{baseline}/model.safetensors", framework="pt", device='cpu') as f:
#     print(len(f.keys()))
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)
#
# print('linear', tensors['linear.weight'])
# print('linear_lite', tensors_lite['linear.weight'])
#
# config = ColBERTConfig.load_from_checkpoint(baseline)
# checkpoint = Checkpoint(baseline, colbert_config=config)
# print('linear', checkpoint.model.linear.weight)
# print('linear_lite', checkpoint.model_lite.linear.weight)

## DEBUG2: baseline document encoder vs. saved frozen document encoder
## >> the output are the same
# baseline = 'hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng'
# config = ColBERTConfig.load_from_checkpoint(baseline)
# checkpoint = Checkpoint(baseline, colbert_config=config)
# print('linear', checkpoint.model.linear.weight)
# print(checkpoint.docFromText(['this is a testing doc'])[:, :10, :10])
# print(checkpoint.queryFromText(['this is a testing query'])[:, :10, :10])
# baseline = 'experiments/q-L0-xlm-roberta-large/none/frozen/64bat.6way/checkpoints/colbert'
# config = ColBERTConfig.load_from_checkpoint(baseline)
# checkpoint = Checkpoint(baseline, colbert_config=config)
# print('linear', checkpoint.model.linear.weight)
# print(checkpoint.docFromText(['this is a testing doc'])[:, :10, :10])
# print(checkpoint.queryFromText(['this is a testing query'])[:, :10, :10])

## DEBUG3: check the embedding weights are correct
# from huggingface_hub import hf_hub_download, login
# from transformers.utils import cached_file
# import torch
# hf_hub_download(
#     repo_id="sentence-transformers/average_word_embeddings_glove.6B.300d", 
#     filename="0_WordEmbeddings/pytorch_model.bin", 
#     local_dir='glove.6B.300d'
# )
# model = torch.load("glove.6B.300d/0_WordEmbeddings/pytorch_model.bin")
# print(model.keys())
# print(model['emb_layer.weight'])
# print(model['emb_layer.weight'].shape)
# from colbert.modeling.colbert import ColBERT
# baseline = 'hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng'
# config = ColBERTConfig(
#     lite_query_encoder=True,
#     lite_document_encoder=False,
#     lite_encoder_init='DylanJHJ/glove.6B.300d',
#     model_lite_name='DylanJHJ/glove.6B.300d',
#     freeze_document_encoder=True,
#     freeze_query_word_embeddings=True,
#     lite_num_hidden_layers=0,
#     lite_num_attention_heads=-1,
#     shared_linear_lite=True,
#     lite_hidden_size=1024
# )
# colbert = ColBERT(baseline, colbert_config=config)
# print(colbert.model_lite.static_embs.static_embs.embeddings.weight[:400001, :])
# print(colbert.model_lite.static_embs.static_embs.embeddings.weight.shape)
#
# for n, p in colbert.named_parameters():
#     if p.requires_grad:
#         print(n, 'activated')

## DEBUG4: check the shared-learned linear layer is working
## >> the output are the same
# baseline = 'experiments/q-L6-minilm/none/frozen/384bat.6way/checkpoints/colbert'
# config = ColBERTConfig.load_from_checkpoint(baseline)
# checkpoint = Checkpoint(baseline, colbert_config=config)
# print('linear', checkpoint.model.linear.weight)

# baseline = 'experiments/q-L6-minilm/none/frozen/384bat.6way/checkpoints/colbert'
# config = ColBERTConfig.load_from_checkpoint(baseline)
# checkpoint = Checkpoint(baseline, colbert_config=config)
# print('linear', checkpoint.model_lite.linear.weight)
# print('linear', checkpoint.model.linear.weight)
# print('bridge', checkpoint.model_lite.bridge.weight)

## DEBUG5: check static embedding can be loaded with hfcolbert
from colbert.modeling.colbert import ColBERT
baseline = 'hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng'
config = ColBERTConfig(
    lite_query_encoder=True,
    lite_document_encoder=False,
    lite_encoder_init='DylanJHJ/glove.6B.300d',
    model_lite_name='DylanJHJ/glove.6B.300d',
    lite_num_hidden_layers=0,
    lite_num_attention_heads=3,
    lite_hidden_size=1024
)
colbert = ColBERT(baseline, colbert_config=config)
print(colbert.raw_tokenizer_lite)
# print(model.queryFromText(['apple banana cat dog'])[:, :8, :5])
# colbert.save('hello')
# print(colbert.model_lite.glove.projection.weight)
# baseline = 'hello'
# config = ColBERTConfig.load_from_checkpoint(baseline)
# model = Checkpoint(baseline, colbert_config=config)
# print(model.model_lite.glove.projection.weight)
# print(model.queryFromText(['apple banana cat dog'])[:, :8, :5])

## DEBUG6: check glove query embeddings
# from colbert.modeling.colbert import ColBERT
# baseline = 'hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng'
# config = ColBERTConfig(
#     lite_query_encoder=True,
#     lite_document_encoder=False,
#     lite_encoder_init='DylanJHJ/glove.6B.300d',
#     model_lite_name='DylanJHJ/glove.6B.300d',
#     lite_num_hidden_layers=0,
#     lite_num_attention_heads=1,
# )
# model = Checkpoint(baseline, colbert_config=config)
# print(model.model_lite)
# print('e', model.model_lite.static_embs.static_embs.embeddings.weight)
# print('l', model.model_lite.linear.weight)
# print('q', model.queryFromText(['this is a testing query'])[:, :8, :5])
# print(model.queryFromText(['apple banana cat dog'])[:, :8, :5])
# print(model.queryFromText(['cat pig canada'])[:, :8, :5])

## DEBUG7: check minilm embeddings
# from colbert.modeling.colbert import ColBERT
# baseline = 'hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng'
# config = ColBERTConfig(
#     lite_query_encoder=True,
#     lite_document_encoder=False,
#     lite_encoder_init='sentence-transformers/all-MiniLM-L6-v2',
#     model_lite_name='sentence-transformers/all-MiniLM-L6-v2',
#     lite_num_hidden_layers=6,
#     lite_num_attention_heads=12,
# )
# colbert = ColBERT(baseline, colbert_config=config)
# print(model.model_lite)

# model = Checkpoint(baseline, colbert_config=config)
# print(model.model_lite)
# print('q', model.queryFromText(['this is a testing query'])[:, :8, :5])

