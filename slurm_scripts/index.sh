#!/bin/sh
#SBATCH --job-name=indexing
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err

. /home/dju/miniconda3/etc/profile.d/conda.sh
conda activate plaid

cd ~/ColBERT-X

dataset=/home/dju/datasets/neuclir-csl/csl.tsv
# checkpoint=experiments/q-L0-xlm-roberta-large/none/frozen/16bat.6way/checkpoints/colbert

rm -r experiments/test/indexes/neuclir-csl
for step in prepare encode finalize; do
python -m colbert.scripts.index \
--coll_dir ${dataset} \
--index_name neuclir-csl \
--dataset_name neuclir-csl \
--nbits 1 \
--step $step \
--checkpoint ${checkpoint} \
--experiment test  
done
