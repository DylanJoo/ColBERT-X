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
# Index with document encoder

dataset=/home/dju/datasets/neuclir-csl/csl.tsv
checkpoint=experiments/colbert-lite-q-L24/none/baseline/12bat.6way/checkpoints/colbert/

for step in prepare encode finalize; do
python -m colbert.scripts.index \
--coll_dir ${dataset} \
--index_name neuclir-csl-test \
--dataset_name neuclir-csl \
--nbits 1 \
--step $step \
--checkpoint ${checkpoint} \
--experiment test  
done
