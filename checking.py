from safetensors import safe_open
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra.config import ColBERTConfig

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
baseline = 'hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng'
config = ColBERTConfig.load_from_checkpoint(baseline)
checkpoint = Checkpoint(baseline, colbert_config=config)
print('linear', checkpoint.model.linear.weight)
print(checkpoint.docFromText(['this is a testing doc'])[:, :10, :10])
print(checkpoint.queryFromText(['this is a testing query'])[:, :10, :10])

baseline = 'experiments/q-L0-xlm-roberta-large/none/frozen/16bat.6way/checkpoints/colbert'
config = ColBERTConfig.load_from_checkpoint(baseline)
checkpoint = Checkpoint(baseline, colbert_config=config)
print('linear', checkpoint.model.linear.weight)
print(checkpoint.docFromText(['this is a testing doc'])[:, :10, :10])
print(checkpoint.queryFromText(['this is a testing query'])[:, :10, :10])
