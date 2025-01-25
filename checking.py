from colbert.modeling.checkpoint import Checkpoint
from colbert.infra.config import ColBERTConfig

from safetensors import safe_open
baseline = 'experiments/colbert-lite/none/fromscratch/8bat.6way/checkpoints/colbert/lite'
tensors_lite = {}
with safe_open(f"{baseline}/model.safetensors", framework="pt", device='cpu') as f:
    print(len(f.keys()))
    for k in f.keys():
        tensors_lite[k] = f.get_tensor(k)
print(tensors_lite)

baseline = 'experiments/colbert-lite/none/fromscratch/8bat.6way/checkpoints/colbert'
tensors = {}
with safe_open(f"{baseline}/model.safetensors", framework="pt", device='cpu') as f:
    print(len(f.keys()))
    for k in tensors_lite.keys():
        tensors[k] = f.get_tensor(k)
print(tensors)

# baseline = 'experiments/colbert-lite/none/fromscratch/8bat.6way/checkpoints/colbert'
# baseline = 'experiments/colbert-full/none/baseline/8bat.6way/checkpoints/colbert'
# checkpoint_config = ColBERTConfig.load_from_checkpoint(baseline)
# checkpoint = Checkpoint(baseline, colbert_config=checkpoint_config)
