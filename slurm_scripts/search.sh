#!/bin/sh
#SBATCH --job-name=eval
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err

. /home/dju/miniconda3/etc/profile.d/conda.sh
conda activate plaid

cd ~/ColBERT-X


# [reproduce] [plaidx]
# python -m colbert.scripts.search \
# --index_name neuclir-csl-plaidx \
# --passage_mapping /home/dju/datasets/neuclir-csl/csl_mapping.tsv \
# --query_file neuclir/neuclir-2023-technical_topics.0719.tsv  \
# --metrics nDCG@20 R@100 \
# --qrel neuclir/tech_final_qrels.txt  \
# --experiment test

# [experiments] [reindex -- shared linear + frozen doc LM]
checkpoint=experiments/frozen_plaidx/none/q-L1-d-L24/128bat.6way/checkpoints/colbert-50000
python -m colbert.scripts.search \
--index_name neuclir-csl \
--checkpoint_path  ${checkpoint} \
--passage_mapping /home/dju/datasets/neuclir-csl/csl_mapping.tsv \
--query_file neuclir/neuclir-2023-technical_topics.0719.tsv  \
--metrics nDCG@20 R@100 \
--qrel neuclir/tech_final_qrels.txt  \
--experiment test

# [experiments] [frozen]
# checkpoint=experiments/q-L12-plaidx/none/frozen/128bat.6way/checkpoints/colbert
# python -m colbert.scripts.search \
# --index_name neuclir-csl-plaidx \
# --checkpoint_path  ${checkpoint} \
# --passage_mapping /home/dju/datasets/neuclir-csl/csl_mapping.tsv \
# --query_file neuclir/neuclir-2023-technical_topics.0719.tsv  \
# --metrics nDCG@20 R@100 \
# --qrel neuclir/tech_final_qrels.txt  \
# --experiment test
