#!/bin/sh
#SBATCH --job-name=eval
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err

. /home/jju/temp/miniconda3/etc/profile.d/conda.sh
conda activate plaid

cd ~/ColBERT-X

# [reproduce] load checkpoint from index
# python -m colbert.scripts.search \
# --index_name neuclir-csl \
# --passage_mapping /home/jju/datasets/neuclir-csl/csl_mapping.tsv \
# --query_file neuclir/neuclir-2023-technical_topics.0719.tsv  \
# --metrics nDCG@20 MAP R@100 R@1000 Judged@10 \
# --qrel neuclir/tech_final_qrels.txt  \
# --experiment test

# checkpoint=experiments/colbert-full/none/baseline/8bat.6way/checkpoints/colbert/lite
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

checkpoint=experiments/colbert-lite/none/fromscratch/8bat.6way/checkpoints/colbert/lite
python -m colbert.scripts.search \
--index_name neuclir-csl \
--checkpoint_path  ${checkpoint} \
--passage_mapping /home/jju/datasets/neuclir-csl/csl_mapping.tsv \
--query_file neuclir/neuclir-2023-technical_topics.0719.tsv  \
--metrics nDCG@20 \
--qrel neuclir/tech_final_qrels.txt  \
--experiment test
# [Baseline]
# {nDCG@20: 0.021239207788448348}
# {nDCG@20: 0.018698787642032057}
