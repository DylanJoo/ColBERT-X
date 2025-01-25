#!/bin/sh
#SBATCH --job-name=indexing
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err

. /home/jju/temp/miniconda3/etc/profile.d/conda.sh
conda activate plaid

cd ~/ColBERT-X

checkpoint=experiments/colbert-lite/none/fromscratch/8bat.6way/checkpoints/colbert
for step in prepare encode finalize; do
python -m colbert.scripts.index \
--coll_dir /home/jju/datasets/neuclir-csl/csl.tsv \
--index_name neuclir-csl \
--dataset_name test_coll \
--nbits 1 \
--step $step \
--checkpoint ${checkpoint} \
--experiment test  
done

# checkpoint=experiments/colbert-lite/none/baseline/8bat.6way/checkpoints/colbert # this is the same as `hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng`
# for step in prepare encode finalize; do
# python -m colbert.scripts.index \
# --coll_dir /home/jju/datasets/neuclir-csl/csl.tsv \
# --index_name neuclir-csl \
# --dataset_name test_coll \
# --nbits 1 \
# --step $step \
# --checkpoint ${checkpoint} \
# --experiment plaidx-zho  
# done
