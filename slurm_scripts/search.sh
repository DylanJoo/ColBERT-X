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


# [reproduce] load checkpoint from index
# hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng: hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng {nDCG@20: 0.35747562328164084}

# [experiments] [frozen]
# checkpoint=experiments/q-L12-xlm-roberta-large/none/frozen/64bat.6way/checkpoints/colbert/
# checkpoint=experiments/q-L0-xlm-roberta-large/none/frozen/64bat.6way/checkpoints/colbert-50000/
checkpoint=experiments/q-L0-plaid/none/frozen/64bat.6way/checkpoints/colbert-100000/
# checkpoint=experiments/q-L0-minilm/none/frozen/64bat.6way/checkpoints/colbert/
python -m colbert.scripts.search \
--index_name neuclir-csl-plaidx \
--checkpoint_path  ${checkpoint} \
--passage_mapping /home/dju/datasets/neuclir-csl/csl_mapping.tsv \
--query_file neuclir/neuclir-2023-technical_topics.0719.tsv  \
--metrics nDCG@20 \
--qrel neuclir/tech_final_qrels.txt  \
--experiment test

# checkpoint=experiments/colbert-full/none/baseline/8bat.6way/checkpoints/colbert
# python -m colbert.scripts.search \
# --index_name neuclir-csl \
# --checkpoint_path  ${checkpoint} \
# --passage_mapping /home/jju/datasets/neuclir-csl/csl_mapping.tsv \
# --query_file neuclir/neuclir-2023-technical_topics.0719.tsv  \
# --metrics nDCG@20 \
# --qrel neuclir/tech_final_qrels.txt  \
# --experiment plaidx-zho
# [Baseline]
# {nDCG@20: 0.1569494826603382}
# {nDCG@20: 0.24299381072171614}

